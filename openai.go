package agentloop

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAICompatibleProvider calls the OpenAI Chat Completions-compatible API.
type OpenAICompatibleProvider struct {
	BaseURL string
	APIKey  string
	Model   string
	Client  *http.Client
}

func NewOpenAICompatibleProvider(options ...Option) *OpenAICompatibleProvider {
	config := applyOptions(options)
	return &OpenAICompatibleProvider{
		BaseURL: strings.TrimRight(config.baseURL, "/"),
		APIKey:  config.apiKey,
		Model:   config.model,
	}
}

func (p *OpenAICompatibleProvider) DefaultModel() string {
	return p.Model
}

func (p *OpenAICompatibleProvider) Complete(ctx context.Context, request Request) (Response, error) {
	if p.BaseURL == "" {
		return Response{}, fmt.Errorf("base URL is required")
	}
	messages := make([]openAIMessage, 0, len(request.Messages)+1)
	if request.SystemPrompt != "" {
		messages = append(messages, openAIMessage{Role: "system", Content: request.SystemPrompt})
	}
	for _, message := range request.Messages {
		messages = append(messages, openAIMessage{Role: message.Role, Content: message.Content, ToolCalls: toOpenAIToolCalls(message.ToolCalls), ToolCallID: message.ToolCallID})
	}
	payload := openAIRequest{Model: request.Model, Messages: messages, ReasoningEffort: request.ReasoningEffort}
	if len(request.OutputSchema) > 0 {
		payload.ResponseFormat = &openAIResponseFormat{
			Type: "json_schema",
			JSONSchema: openAIJSONSchema{
				Name:   "response",
				Schema: request.OutputSchema,
				Strict: true,
			},
		}
	}
	for _, tool := range request.Tools {
		payload.Tools = append(payload.Tools, openAITool{Type: "function", Function: tool})
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return Response{}, fmt.Errorf("encode request: %w", err)
	}
	httpClient := p.Client
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("create request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" {
		httpRequest.Header.Set("Authorization", "Bearer "+p.APIKey)
	}
	httpResponse, err := httpClient.Do(httpRequest)
	if err != nil {
		return Response{}, &retryableError{err}
	}
	defer httpResponse.Body.Close()
	responseBody, err := io.ReadAll(httpResponse.Body)
	if err != nil {
		return Response{}, fmt.Errorf("read response: %w", err)
	}
	if httpResponse.StatusCode < http.StatusOK || httpResponse.StatusCode >= http.StatusMultipleChoices {
		return Response{}, &APIError{StatusCode: httpResponse.StatusCode, Status: httpResponse.Status, Body: strings.TrimSpace(string(responseBody))}
	}
	var payloadResponse openAIResponse
	if err := json.Unmarshal(responseBody, &payloadResponse); err != nil {
		return Response{}, fmt.Errorf("decode response: %w", err)
	}
	if len(payloadResponse.Choices) == 0 {
		return Response{}, fmt.Errorf("decode response: no choices")
	}
	return Response{Content: payloadResponse.Choices[0].Message.Content, ToolCalls: fromOpenAIToolCalls(payloadResponse.Choices[0].Message.ToolCalls), Usage: Usage{InputTokens: payloadResponse.Usage.PromptTokens, OutputTokens: payloadResponse.Usage.CompletionTokens}}, nil
}

// APIError is a non-success response from an OpenAI-compatible API.
type APIError struct {
	StatusCode int
	Status     string
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("API returned %s: %s", e.Status, e.Body)
}

func (e *APIError) Retryable() bool {
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode >= http.StatusInternalServerError
}

type retryableError struct{ error }

func (retryableError) Retryable() bool { return true }

type openAIRequest struct {
	Model           string                `json:"model"`
	Messages        []openAIMessage       `json:"messages"`
	Tools           []openAITool          `json:"tools,omitempty"`
	ReasoningEffort string                `json:"reasoning_effort,omitempty"`
	ResponseFormat  *openAIResponseFormat `json:"response_format,omitempty"`
}

type openAIResponseFormat struct {
	Type       string           `json:"type"`
	JSONSchema openAIJSONSchema `json:"json_schema"`
}

type openAIJSONSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func toOpenAIToolCalls(calls []ToolCall) []openAIToolCall {
	converted := make([]openAIToolCall, 0, len(calls))
	for _, call := range calls {
		var convertedCall openAIToolCall
		convertedCall.ID, convertedCall.Type = call.ID, "function"
		convertedCall.Function.Name, convertedCall.Function.Arguments = call.Name, string(call.Arguments)
		converted = append(converted, convertedCall)
	}
	return converted
}

func fromOpenAIToolCalls(calls []openAIToolCall) []ToolCall {
	converted := make([]ToolCall, 0, len(calls))
	for _, call := range calls {
		converted = append(converted, ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: json.RawMessage(call.Function.Arguments)})
	}
	return converted
}

type openAITool struct {
	Type     string         `json:"type"`
	Function ToolDefinition `json:"function"`
}

type openAIResponse struct {
	Choices []struct {
		Message openAIMessage `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}
