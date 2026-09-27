package agentloop

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
)

type fakeProvider struct {
	responses []Response
	requests  []Request
}

func (p *fakeProvider) Complete(_ context.Context, request Request) (Response, error) {
	p.requests = append(p.requests, request)
	response := p.responses[0]
	p.responses = p.responses[1:]
	return response, nil
}

type testTool struct{}

func (testTool) Definition() ToolDefinition {
	return ToolDefinition{Name: "double", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (testTool) Execute(_ context.Context, input json.RawMessage) (any, error) {
	var arguments struct{ Value int }
	if err := json.Unmarshal(input, &arguments); err != nil {
		return nil, err
	}
	return map[string]int{"result": arguments.Value * 2}, nil
}

func TestAgentRunsToolLoop(t *testing.T) {
	var events []Event
	provider := &fakeProvider{responses: []Response{
		{ToolCalls: []ToolCall{{ID: "call_1", Name: "double", Arguments: json.RawMessage(`{"value":21}`)}}, Usage: Usage{InputTokens: 10, OutputTokens: 1, CachedTokens: 5, CacheCreationTokens: 2, Cost: 0.001}},
		{Content: "42", Usage: Usage{InputTokens: 3, OutputTokens: 2, CachedTokens: 1, CacheCreationTokens: 4, Cost: 0.002}},
	}}
	result, err := (Agent{Provider: provider, Model: "test", OutputSchema: json.RawMessage(`{"type":"string"}`), Tools: []Tool{testTool{}}, EventHandler: func(event Event) { events = append(events, event) }}).Run(context.Background(), "double 21")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "42" || result.Steps != 2 || result.Usage.InputTokens != 13 {
		t.Fatalf("unexpected result: %+v", result)
	}
	if result.Usage.CachedTokens != 6 || result.Usage.CacheCreationTokens != 6 || result.Usage.Cost != 0.003 {
		t.Fatalf("unexpected cache usage: %+v", result.Usage)
	}
	var lastResponseUsage Usage
	for _, event := range events {
		if event.Type == EventModelResponse {
			lastResponseUsage = event.Usage
		}
	}
	if lastResponseUsage.CachedTokens != 1 || lastResponseUsage.CacheCreationTokens != 4 {
		t.Fatalf("unexpected per-step event usage: %+v", lastResponseUsage)
	}
	if got := provider.requests[1].Messages[2].Content; got != `{"result":42}` {
		t.Fatalf("tool output = %q", got)
	}
	if got := string(provider.requests[1].OutputSchema); got != `{"type":"string"}` {
		t.Fatalf("output schema = %q", got)
	}
}

func TestAgentStopsAtMaxSteps(t *testing.T) {
	provider := &fakeProvider{responses: []Response{{ToolCalls: []ToolCall{{ID: "call_1", Name: "double", Arguments: json.RawMessage(`{"value":1}`)}}}}}
	_, err := (Agent{Provider: provider, Model: "test", Tools: []Tool{testTool{}}, MaxSteps: 1}).Run(context.Background(), "loop")
	if err == nil || err.Error() != "agentloop: reached max steps (1)" {
		t.Fatalf("error = %v", err)
	}
}

func TestAgentFinalResponseAtMaxSteps(t *testing.T) {
	provider := &fakeProvider{responses: []Response{
		{ToolCalls: []ToolCall{{ID: "call_1", Name: "double", Arguments: json.RawMessage(`{"value":21}`)}}},
		{Content: "42"},
	}}
	result, err := (Agent{Provider: provider, Model: "test", Tools: []Tool{testTool{}}, MaxSteps: 2, FinalResponseAtMaxSteps: true}).Run(context.Background(), "double 21")
	if err != nil || result.Output != "42" || result.Steps != 2 {
		t.Fatalf("result = %+v, err = %v", result, err)
	}
	if len(provider.requests) != 2 || len(provider.requests[0].Tools) != 1 || provider.requests[1].Tools != nil {
		t.Fatalf("requests = %+v", provider.requests)
	}
}

func TestAgentRejectsToolCallsOnFinalResponseStep(t *testing.T) {
	provider := &fakeProvider{responses: []Response{{ToolCalls: []ToolCall{{ID: "call_1", Name: "double"}}}}}
	_, err := (Agent{Provider: provider, Model: "test", Tools: []Tool{testTool{}}, MaxSteps: 1, FinalResponseAtMaxSteps: true}).Run(context.Background(), "loop")
	if err == nil || err.Error() != "agentloop: provider returned tool calls without tools" {
		t.Fatalf("error = %v", err)
	}
	if provider.requests[0].Tools != nil {
		t.Fatalf("tools = %+v", provider.requests[0].Tools)
	}
}

func TestAgentRejectsNegativeMaxSteps(t *testing.T) {
	_, err := (Agent{Provider: &fakeProvider{}, Model: "test", MaxSteps: -1}).Run(context.Background(), "run")
	if err == nil || err.Error() != "agentloop: max steps cannot be negative" {
		t.Fatalf("error = %v", err)
	}
}

func TestAgentRejectsInvalidOutputSchema(t *testing.T) {
	_, err := (Agent{Provider: &fakeProvider{}, Model: "test", OutputSchema: []byte(`{`)}).Run(context.Background(), "run")
	if err == nil || err.Error() != "agentloop: output schema must be valid JSON" {
		t.Fatalf("error = %v", err)
	}
}

func TestAgentRejectsUnknownTool(t *testing.T) {
	provider := &fakeProvider{responses: []Response{{ToolCalls: []ToolCall{{Name: "missing"}}}}}
	_, err := (Agent{Provider: provider, Model: "test"}).Run(context.Background(), "run")
	if err == nil || err.Error() != `agentloop: unknown tool "missing"` {
		t.Fatalf("error = %v", err)
	}
}

func TestAgentUsesProviderDefaultModel(t *testing.T) {
	agent := NewOpenAICompatibleAgent(WithBaseURL("http://example.test"), WithModel("gpt-4"))
	provider := agent.Provider.(*OpenAICompatibleProvider)
	provider.Client = &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if !strings.Contains(readBody(request), `"model":"gpt-4"`) {
			return nil, fmt.Errorf("provider model was not used")
		}
		return response(`{"choices":[{"message":{"content":"done"}}]}`), nil
	})}
	result, err := agent.Run(context.Background(), "finish")
	if err != nil || result.Output != "done" {
		t.Fatalf("result = %+v, err = %v", result, err)
	}
}

func TestOpenAICompatibleAgentResilienceDefaults(t *testing.T) {
	agent := NewOpenAICompatibleAgent()
	if agent.RequestTimeout != DefaultRequestTimeout || agent.MaxRetries != DefaultMaxRetries || agent.ToolTimeout != DefaultToolTimeout {
		t.Fatalf("unexpected defaults: %+v", agent)
	}

	agent = NewOpenAICompatibleAgent(WithRequestTimeout(0), WithMaxRetries(0), WithToolTimeout(0), WithFinalResponseAtMaxSteps(true))
	if agent.RequestTimeout != 0 || agent.MaxRetries != 0 || agent.ToolTimeout != 0 || !agent.FinalResponseAtMaxSteps {
		t.Fatalf("expected explicit zero values: %+v", agent)
	}
}

func ExampleAgent_Run() {
	provider := &fakeProvider{responses: []Response{{Content: "done"}}}
	result, _ := (Agent{Provider: provider, Model: "example"}).Run(context.Background(), "finish this")
	fmt.Println(result.Output)
	// Output: done
}
