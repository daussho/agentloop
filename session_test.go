package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestSessionRetainsMessagesBetweenRuns(t *testing.T) {
	provider := &fakeProvider{responses: []Response{{Content: "first"}, {Content: "second"}}}
	session := (Agent{Provider: provider, Model: "test"}).NewSession()
	if _, err := session.Run(context.Background(), "one"); err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(context.Background(), "two"); err != nil {
		t.Fatal(err)
	}
	if got := provider.requests[1].Messages; len(got) != 3 || got[0].Content != "one" || got[2].Content != "two" {
		t.Fatalf("second request messages = %+v", got)
	}
}

func TestSessionEmitsEvents(t *testing.T) {
	provider := &fakeProvider{responses: []Response{
		{ToolCalls: []ToolCall{{ID: "call_1", Name: "double", Arguments: []byte(`{"value":2}`)}}},
		{Content: "done"},
	}}
	var events []EventType
	session := (Agent{Provider: provider, Model: "test", Tools: []Tool{testTool{}}, EventHandler: func(event Event) {
		events = append(events, event.Type)
	}}).NewSession()
	if _, err := session.Run(context.Background(), "run"); err != nil {
		t.Fatal(err)
	}
	want := []EventType{EventModelRequest, EventModelResponse, EventToolCall, EventToolResult, EventModelRequest, EventModelResponse, EventCompleted}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
}

type retryError struct{}

func (retryError) Error() string   { return "temporary failure" }
func (retryError) Retryable() bool { return true }

type retryProvider struct{ calls int }

func (p *retryProvider) Complete(_ context.Context, _ Request) (Response, error) {
	p.calls++
	if p.calls == 1 {
		return Response{}, retryError{}
	}
	return Response{Content: "done"}, nil
}

func TestSessionRetriesTransientProviderError(t *testing.T) {
	provider := &retryProvider{}
	result, err := (Agent{Provider: provider, Model: "test", MaxRetries: 1}).Run(context.Background(), "run")
	if err != nil || result.Output != "done" || provider.calls != 2 {
		t.Fatalf("result = %+v, err = %v, calls = %d", result, err, provider.calls)
	}
}

type contextWaitProvider struct{}

func (contextWaitProvider) Complete(ctx context.Context, _ Request) (Response, error) {
	<-ctx.Done()
	return Response{}, ctx.Err()
}

func TestSessionAppliesRequestTimeout(t *testing.T) {
	_, err := (Agent{Provider: contextWaitProvider{}, Model: "test", RequestTimeout: 10 * time.Millisecond}).Run(context.Background(), "run")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v", err)
	}
}

type contextWaitTool struct{}

func (contextWaitTool) Definition() ToolDefinition { return ToolDefinition{Name: "wait"} }

func (contextWaitTool) Execute(ctx context.Context, _ json.RawMessage) (any, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestSessionAppliesToolTimeout(t *testing.T) {
	provider := &fakeProvider{responses: []Response{{ToolCalls: []ToolCall{{Name: "wait"}}}}}
	_, err := (Agent{Provider: provider, Model: "test", ToolTimeout: 10 * time.Millisecond, Tools: []Tool{contextWaitTool{}}}).Run(context.Background(), "run")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v", err)
	}
}

type blockingProvider struct {
	started chan struct{}
	release chan struct{}
	mu      sync.Mutex
	calls   int
}

func (p *blockingProvider) Complete(_ context.Context, _ Request) (Response, error) {
	p.mu.Lock()
	p.calls++
	call := p.calls
	p.mu.Unlock()
	if call == 1 {
		close(p.started)
		<-p.release
	}
	return Response{Content: "done"}, nil
}

func (p *blockingProvider) Calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

func TestSessionSessionIDIsStableAcrossStepsAndRuns(t *testing.T) {
	provider := &fakeProvider{responses: []Response{
		{ToolCalls: []ToolCall{{ID: "call_1", Name: "double", Arguments: json.RawMessage(`{"value":2}`)}}},
		{Content: "done"},
		{Content: "again"},
	}}
	session := (Agent{Provider: provider, Model: "test", Tools: []Tool{testTool{}}}).NewSession()
	if session.sessionID == "" {
		t.Fatal("NewSession did not generate a session ID")
	}
	if _, err := session.Run(context.Background(), "one"); err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(context.Background(), "two"); err != nil {
		t.Fatal(err)
	}
	for i, request := range provider.requests {
		if request.SessionID != session.sessionID {
			t.Fatalf("request %d session ID = %q, want %q", i, request.SessionID, session.sessionID)
		}
	}
}

func TestNewSessionUsesProvidedSessionID(t *testing.T) {
	session := (Agent{}).NewSession(WithSessionID("conversation-1"))
	if session.sessionID != "conversation-1" {
		t.Fatalf("session ID = %q, want conversation-1", session.sessionID)
	}
}

func TestResumeSessionRestoresHistoryAndCopiesInput(t *testing.T) {
	history := []Message{
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Name: "double", Arguments: json.RawMessage(`{"value":21}`)}}},
		{Role: "tool", Content: `{"result":42}`, ToolCallID: "call_1"},
	}
	provider := &fakeProvider{responses: []Response{{Content: "done"}}}
	session, err := (Agent{Provider: provider, Model: "test"}).ResumeSession("resumed-1", history)
	if err != nil {
		t.Fatal(err)
	}
	history[0].ToolCalls[0].Arguments = json.RawMessage(`{"value":999}`)
	history[0].ToolCalls = append(history[0].ToolCalls, ToolCall{ID: "call_2", Name: "other"})

	if _, err := session.Run(context.Background(), "continue"); err != nil {
		t.Fatal(err)
	}
	got := provider.requests[0].Messages
	if len(got) != 3 || got[0].Role != "assistant" || len(got[0].ToolCalls) != 1 || string(got[0].ToolCalls[0].Arguments) != `{"value":21}` || got[1].Role != "tool" || got[1].ToolCallID != "call_1" || got[2].Content != "continue" {
		t.Fatalf("resumed messages = %+v", got)
	}
	if got := provider.requests[0].SessionID; got != "resumed-1" {
		t.Fatalf("resumed session ID = %q, want resumed-1", got)
	}
}

func TestResumeSessionValidatesSessionID(t *testing.T) {
	if _, err := (Agent{Provider: &fakeProvider{}, Model: "test"}).ResumeSession("", nil); err == nil {
		t.Fatal("expected error for empty session ID")
	}
	if _, err := (Agent{Provider: &fakeProvider{}, Model: "test"}).ResumeSession(strings.Repeat("x", 257), nil); err == nil {
		t.Fatal("expected error for session ID over 256 characters")
	}
}

func TestSessionSerializesRuns(t *testing.T) {
	provider := &blockingProvider{started: make(chan struct{}), release: make(chan struct{})}
	session := (Agent{Provider: provider, Model: "test"}).NewSession()
	done := make(chan struct{}, 2)
	go func() { _, _ = session.Run(context.Background(), "one"); done <- struct{}{} }()
	<-provider.started
	go func() { _, _ = session.Run(context.Background(), "two"); done <- struct{}{} }()

	<-time.After(25 * time.Millisecond)
	if provider.Calls() != 1 {
		t.Fatalf("provider calls = %d; want 1 while first run is blocked", provider.Calls())
	}
	close(provider.release)
	<-done
	<-done
}
