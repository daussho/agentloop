package agentloop

import (
	"context"
	"reflect"
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
