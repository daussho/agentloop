package agentloop

import (
	"context"
	"encoding/json"
	"time"
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
	OutputSchema    json.RawMessage
	Messages        []Message
	Tools           []ToolDefinition
}

type Response struct {
	Content   string
	ToolCalls []ToolCall
	Usage     Usage
}

type Usage struct {
	InputTokens         int     `json:"input_tokens"`
	OutputTokens        int     `json:"output_tokens"`
	CachedTokens        int     `json:"cached_tokens"`               // cache read (prompt_tokens_details.cached_tokens)
	CacheCreationTokens int     `json:"cache_creation_input_tokens"` // cache write / creation
	Cost                float64 `json:"cost"`                        // provider-computed cost in USD, when reported
}

type Result struct {
	Output   string
	Messages []Message
	Steps    int
	Usage    Usage
}

type EventType string

const (
	EventModelRequest  EventType = "model_request"
	EventModelResponse EventType = "model_response"
	EventToolCall      EventType = "tool_call"
	EventToolResult    EventType = "tool_result"
	EventError         EventType = "error"
	EventCompleted     EventType = "completed"
)

// Event describes an agent-loop state transition.
type Event struct {
	Type     EventType
	Step     int
	Content  string
	ToolCall ToolCall
	Usage    Usage
	Err      error
}

// EventHandler receives session events synchronously.
type EventHandler func(Event)

type Agent struct {
	Provider        Provider
	Model           string
	SystemPrompt    string
	ReasoningEffort string
	OutputSchema    json.RawMessage
	Tools           []Tool
	MaxSteps        int
	RequestTimeout  time.Duration
	MaxRetries      int
	ToolTimeout     time.Duration
	EventHandler    EventHandler
}

const (
	ReasoningEffortLow    = "low"
	ReasoningEffortMedium = "medium"
	ReasoningEffortHigh   = "high"

	DefaultRequestTimeout = 2 * time.Minute
	DefaultMaxRetries     = 2
	DefaultToolTimeout    = 30 * time.Second
)

// Option configures an OpenAI-compatible agent or provider.
type Option func(*config)

type config struct {
	baseURL           string
	apiKey            string
	model             string
	systemPrompt      string
	reasoningEffort   string
	outputSchema      json.RawMessage
	tools             []Tool
	maxSteps          int
	requestTimeout    time.Duration
	maxRetries        int
	toolTimeout       time.Duration
	requestTimeoutSet bool
	maxRetriesSet     bool
	toolTimeoutSet    bool
	eventHandler      EventHandler
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

// WithOutputSchema requests strict JSON-schema output for the final response.
func WithOutputSchema(schema json.RawMessage) Option {
	return func(config *config) { config.outputSchema = append(json.RawMessage(nil), schema...) }
}

func WithTools(tools ...Tool) Option {
	return func(config *config) { config.tools = append(config.tools, tools...) }
}

func WithMaxSteps(maxSteps int) Option {
	return func(config *config) { config.maxSteps = maxSteps }
}

// WithRequestTimeout limits each provider call. Zero disables the limit.
func WithRequestTimeout(timeout time.Duration) Option {
	return func(config *config) {
		config.requestTimeout, config.requestTimeoutSet = timeout, true
	}
}

// WithMaxRetries retries transient provider failures. Zero disables retries.
func WithMaxRetries(retries int) Option {
	return func(config *config) {
		config.maxRetries, config.maxRetriesSet = retries, true
	}
}

// WithToolTimeout limits each tool call. Zero disables the limit.
func WithToolTimeout(timeout time.Duration) Option {
	return func(config *config) {
		config.toolTimeout, config.toolTimeoutSet = timeout, true
	}
}

func WithEventHandler(handler EventHandler) Option {
	return func(config *config) { config.eventHandler = handler }
}

// NewOpenAICompatibleAgent builds an agent backed by an OpenAI-compatible API.
func NewOpenAICompatibleAgent(options ...Option) *Agent {
	config := applyOptions(options)
	if !config.requestTimeoutSet {
		config.requestTimeout = DefaultRequestTimeout
	}
	if !config.maxRetriesSet {
		config.maxRetries = DefaultMaxRetries
	}
	if !config.toolTimeoutSet {
		config.toolTimeout = DefaultToolTimeout
	}
	return &Agent{
		Provider:        NewOpenAICompatibleProvider(options...),
		SystemPrompt:    config.systemPrompt,
		ReasoningEffort: config.reasoningEffort,
		OutputSchema:    config.outputSchema,
		Tools:           config.tools,
		MaxSteps:        config.maxSteps,
		RequestTimeout:  config.requestTimeout,
		MaxRetries:      config.maxRetries,
		ToolTimeout:     config.toolTimeout,
		EventHandler:    config.eventHandler,
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
