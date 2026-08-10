package agentloop

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

func readBody(request *http.Request) string {
	body, _ := io.ReadAll(request.Body)
	return string(body)
}

func response(body string) *http.Response {
	return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Body: io.NopCloser(bytes.NewBufferString(body)), Header: make(http.Header)}
}

func TestOpenAICompatibleProviderTranslatesToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if r.URL.Path != "/chat/completions" || !strings.Contains(string(body), `"arguments":"{\"value\":21}"`) || !strings.Contains(string(body), `"response_format":{"type":"json_schema","json_schema":{"name":"response","schema":{"type":"object"},"strict":true}}`) {
			t.Fatalf("unexpected request: %s %s", r.URL, body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"done","tool_calls":[{"id":"call_2","type":"function","function":{"name":"double","arguments":"{\"value\":42}"}}]}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":2}}}`))
	}))
	defer server.Close()

	provider := NewOpenAICompatibleProvider(WithBaseURL(server.URL), WithKey("key"))
	response, err := provider.Complete(context.Background(), Request{Model: "test", OutputSchema: []byte(`{"type":"object"}`), Messages: []Message{{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Name: "double", Arguments: []byte(`{"value":21}`)}}}}})
	if err != nil {
		t.Fatal(err)
	}
	if response.ToolCalls[0].Name != "double" || string(response.ToolCalls[0].Arguments) != `{"value":42}` || response.Usage.InputTokens != 5 {
		t.Fatalf("unexpected response: %+v", response)
	}
	if response.Usage.CachedTokens != 2 {
		t.Fatalf("cached tokens = %d, want 2", response.Usage.CachedTokens)
	}
}

func TestOpenAICompatibleProviderParsesOpenRouterCacheUsage(t *testing.T) {
	provider := NewOpenAICompatibleProvider(WithBaseURL("http://example.test"))
	provider.Client = &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		return response(`{"choices":[{"message":{"content":"done"}}],"usage":{"prompt_tokens":10,"completion_tokens":4,"cache_read_input_tokens":6,"cache_creation_input_tokens":3,"cost":0.00042}}`), nil
	})}
	response, err := provider.Complete(context.Background(), Request{Model: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if response.Usage.InputTokens != 10 || response.Usage.OutputTokens != 4 || response.Usage.CachedTokens != 6 || response.Usage.CacheCreationTokens != 3 || response.Usage.Cost != 0.00042 {
		t.Fatalf("unexpected usage: %+v", response.Usage)
	}
}

func TestOpenAICompatibleProviderRequiresBaseURL(t *testing.T) {
	_, err := NewOpenAICompatibleProvider().Complete(context.Background(), Request{Model: "test"})
	if err == nil || err.Error() != "base URL is required" {
		t.Fatalf("error = %v", err)
	}
}

func TestOpenAICompatibleProviderSessionHeader(t *testing.T) {
	tests := []struct {
		name       string
		openRouter bool
		sessionID  string
		want       string
	}{
		{name: "sends with openrouter", openRouter: true, sessionID: "session-1", want: "session-1"},
		{name: "omits without openrouter", sessionID: "session-1"},
		{name: "omits empty session id", openRouter: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			options := []Option{WithBaseURL("http://example.test")}
			if test.openRouter {
				options = append(options, WithOpenRouter())
			}
			provider := NewOpenAICompatibleProvider(options...)
			var got string
			provider.Client = &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
				got = request.Header.Get("x-session-id")
				return response(`{"choices":[{"message":{"content":"done"}}]}`), nil
			})}
			if _, err := provider.Complete(context.Background(), Request{Model: "test", SessionID: test.sessionID}); err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("x-session-id = %q, want %q", got, test.want)
			}
		})
	}
}
