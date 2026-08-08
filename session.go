package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

// Session retains a conversation's messages between runs. Runs are serialized.
type Session struct {
	agent    Agent
	mu       sync.Mutex
	messages []Message
}

// NewSession starts a conversation that retains messages between runs.
func (a Agent) NewSession() *Session {
	return &Session{agent: a}
}

// Messages returns the conversation history.
func (s *Session) Messages() []Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return copyMessages(s.messages)
}

// Run adds input to the session and serializes concurrent calls.
func (s *Session) Run(ctx context.Context, input string) (Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.run(ctx, input)
	if result.Messages != nil {
		s.messages = copyMessages(result.Messages)
	}
	return result, err
}

func (s *Session) run(ctx context.Context, input string) (Result, error) {
	a := s.agent
	if a.Provider == nil {
		err := errors.New("agentloop: provider is required")
		s.emit(Event{Type: EventError, Err: err})
		return Result{}, err
	}
	model := a.Model
	if model == "" {
		if configured, ok := a.Provider.(interface{ DefaultModel() string }); ok {
			model = configured.DefaultModel()
		}
	}
	if len(a.OutputSchema) > 0 && !json.Valid(a.OutputSchema) {
		err := errors.New("agentloop: output schema must be valid JSON")
		s.emit(Event{Type: EventError, Err: err})
		return Result{}, err
	}
	if model == "" {
		err := errors.New("agentloop: model is required")
		s.emit(Event{Type: EventError, Err: err})
		return Result{}, err
	}
	maxSteps := a.MaxSteps
	if maxSteps < 0 {
		err := errors.New("agentloop: max steps cannot be negative")
		s.emit(Event{Type: EventError, Err: err})
		return Result{}, err
	}
	if maxSteps == 0 {
		maxSteps = 20
	}

	tools := make(map[string]Tool, len(a.Tools))
	definitions := make([]ToolDefinition, 0, len(a.Tools))
	for _, tool := range a.Tools {
		definition := tool.Definition()
		if definition.Name == "" {
			err := errors.New("agentloop: tool name is required")
			s.emit(Event{Type: EventError, Err: err})
			return Result{}, err
		}
		if _, exists := tools[definition.Name]; exists {
			err := fmt.Errorf("agentloop: duplicate tool %q", definition.Name)
			s.emit(Event{Type: EventError, Err: err})
			return Result{}, err
		}
		tools[definition.Name] = tool
		definitions = append(definitions, definition)
	}

	messages := append(copyMessages(s.messages), Message{Role: "user", Content: input})
	result := Result{Messages: messages}
	for step := 1; step <= maxSteps; step++ {
		s.emit(Event{Type: EventModelRequest, Step: step})
		response, err := a.Provider.Complete(ctx, Request{Model: model, SystemPrompt: a.SystemPrompt, ReasoningEffort: a.ReasoningEffort, OutputSchema: a.OutputSchema, Messages: messages, Tools: definitions})
		if err != nil {
			s.emit(Event{Type: EventError, Step: step, Err: err})
			return result, fmt.Errorf("agentloop: complete: %w", err)
		}
		result.Steps = step
		result.Usage.InputTokens += response.Usage.InputTokens
		result.Usage.OutputTokens += response.Usage.OutputTokens
		messages = append(messages, Message{Role: "assistant", Content: response.Content, ToolCalls: response.ToolCalls})
		result.Messages = messages
		s.emit(Event{Type: EventModelResponse, Step: step, Content: response.Content})
		if len(response.ToolCalls) == 0 {
			result.Output, result.Messages = response.Content, messages
			s.emit(Event{Type: EventCompleted, Step: step, Content: response.Content})
			return result, nil
		}
		for _, call := range response.ToolCalls {
			s.emit(Event{Type: EventToolCall, Step: step, ToolCall: call})
			tool, ok := tools[call.Name]
			if !ok {
				s.emit(Event{Type: EventError, Step: step, ToolCall: call, Err: fmt.Errorf("unknown tool %q", call.Name)})
				return result, fmt.Errorf("agentloop: unknown tool %q", call.Name)
			}
			output, err := tool.Execute(ctx, call.Arguments)
			if err != nil {
				s.emit(Event{Type: EventError, Step: step, ToolCall: call, Err: err})
				return result, fmt.Errorf("agentloop: tool %q: %w", call.Name, err)
			}
			encoded, err := json.Marshal(output)
			if err != nil {
				s.emit(Event{Type: EventError, Step: step, ToolCall: call, Err: err})
				return result, fmt.Errorf("agentloop: encode tool %q output: %w", call.Name, err)
			}
			messages = append(messages, Message{Role: "tool", Content: string(encoded), ToolCallID: call.ID})
			result.Messages = messages
			s.emit(Event{Type: EventToolResult, Step: step, ToolCall: call, Content: string(encoded)})
		}
	}
	err := fmt.Errorf("agentloop: reached max steps (%d)", maxSteps)
	s.emit(Event{Type: EventError, Step: result.Steps, Err: err})
	return result, err
}

func (s *Session) emit(event Event) {
	if s.agent.EventHandler != nil {
		s.agent.EventHandler(event)
	}
}

func copyMessages(messages []Message) []Message {
	return append([]Message(nil), messages...)
}
