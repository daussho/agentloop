package agentloop

import (
	"context"
	"encoding/json"
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
	return a.NewSession().Run(ctx, input)
}
