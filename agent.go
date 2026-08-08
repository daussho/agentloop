package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// Provider completes a conversation using a model provider.
type Provider interface {
	Complete(context.Context, Request) (Response, error)
}

// Tool is a function the agent can call.
type Tool interface {
	Definition() ToolDefinition
	Execute(context.Context, json.RawMessage) (any, error)
}

type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type Request struct {
	Model           string
	SystemPrompt    string
	ReasoningEffort string
	Messages        []Message
	Tools           []ToolDefinition
}

type Response struct {
	Content   string
	ToolCalls []ToolCall
	Usage     Usage
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type Result struct {
	Output   string
	Messages []Message
	Steps    int
	Usage    Usage
}

type Agent struct {
	Provider        Provider
	Model           string
	SystemPrompt    string
	ReasoningEffort string
	Tools           []Tool
	MaxSteps        int
}

// Option configures an OpenAI-compatible agent or provider.
type Option func(*config)

type config struct {
	baseURL         string
	apiKey          string
	model           string
	systemPrompt    string
	reasoningEffort string
	tools           []Tool
	maxSteps        int
}

func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.baseURL = baseURL }
}

func WithKey(apiKey string) Option {
	return func(config *config) { config.apiKey = apiKey }
}

func WithModel(model string) Option {
	return func(config *config) { config.model = model }
}

func WithSystemPrompt(prompt string) Option {
	return func(config *config) { config.systemPrompt = prompt }
}

func WithReasoningEffort(effort string) Option {
	return func(config *config) { config.reasoningEffort = effort }
}

func WithTools(tools ...Tool) Option {
	return func(config *config) { config.tools = append(config.tools, tools...) }
}

func WithMaxSteps(maxSteps int) Option {
	return func(config *config) { config.maxSteps = maxSteps }
}

// NewOpenAICompatibleAgent builds an agent backed by an OpenAI-compatible API.
func NewOpenAICompatibleAgent(options ...Option) *Agent {
	config := applyOptions(options)
	return &Agent{
		Provider:        NewOpenAICompatibleProvider(options...),
		SystemPrompt:    config.systemPrompt,
		ReasoningEffort: config.reasoningEffort,
		Tools:           config.tools,
		MaxSteps:        config.maxSteps,
	}
}

func applyOptions(options []Option) config {
	var config config
	for _, option := range options {
		option(&config)
	}
	return config
}

// Run continues until the provider returns a final response or an error occurs.
func (a Agent) Run(ctx context.Context, input string) (Result, error) {
	if a.Provider == nil {
		return Result{}, errors.New("agentloop: provider is required")
	}
	model := a.Model
	if model == "" {
		if configured, ok := a.Provider.(interface{ DefaultModel() string }); ok {
			model = configured.DefaultModel()
		}
	}
	if model == "" {
		return Result{}, errors.New("agentloop: model is required")
	}
	maxSteps := a.MaxSteps
	if maxSteps == 0 {
		maxSteps = 20
	}

	tools := make(map[string]Tool, len(a.Tools))
	definitions := make([]ToolDefinition, 0, len(a.Tools))
	for _, tool := range a.Tools {
		definition := tool.Definition()
		if definition.Name == "" {
			return Result{}, errors.New("agentloop: tool name is required")
		}
		if _, exists := tools[definition.Name]; exists {
			return Result{}, fmt.Errorf("agentloop: duplicate tool %q", definition.Name)
		}
		tools[definition.Name] = tool
		definitions = append(definitions, definition)
	}

	messages := []Message{{Role: "user", Content: input}}
	result := Result{Messages: messages}
	for step := 1; step <= maxSteps; step++ {
		response, err := a.Provider.Complete(ctx, Request{
			Model:           model,
			SystemPrompt:    a.SystemPrompt,
			ReasoningEffort: a.ReasoningEffort,
			Messages:        messages,
			Tools:           definitions,
		})
		if err != nil {
			return result, fmt.Errorf("agentloop: complete: %w", err)
		}
		result.Steps = step
		result.Usage.InputTokens += response.Usage.InputTokens
		result.Usage.OutputTokens += response.Usage.OutputTokens
		messages = append(messages, Message{Role: "assistant", Content: response.Content, ToolCalls: response.ToolCalls})
		result.Messages = messages
		if len(response.ToolCalls) == 0 {
			result.Output, result.Messages = response.Content, messages
			return result, nil
		}
		for _, call := range response.ToolCalls {
			tool, ok := tools[call.Name]
			if !ok {
				return result, fmt.Errorf("agentloop: unknown tool %q", call.Name)
			}
			output, err := tool.Execute(ctx, call.Arguments)
			if err != nil {
				return result, fmt.Errorf("agentloop: tool %q: %w", call.Name, err)
			}
			encoded, err := json.Marshal(output)
			if err != nil {
				return result, fmt.Errorf("agentloop: encode tool %q output: %w", call.Name, err)
			}
			messages = append(messages, Message{Role: "tool", Content: string(encoded), ToolCallID: call.ID})
			result.Messages = messages
		}
	}
	return result, fmt.Errorf("agentloop: reached max steps (%d)", maxSteps)
}
