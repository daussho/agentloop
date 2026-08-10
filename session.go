package agentloop

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"time"
)

// Session retains a conversation's messages between runs. Runs are serialized.
type Session struct {
	agent     Agent
	mu        sync.Mutex
	sessionID string
	messages  []Message
}

// NewSession starts a conversation that retains messages between runs.
func (a Agent) NewSession() *Session {
	return &Session{agent: a, sessionID: newSessionID()}
}

// ResumeSession resumes a conversation with a caller-persisted session ID and
// history. The messages are copied; the caller may reuse or mutate the input.
func (a Agent) ResumeSession(sessionID string, messages []Message) (*Session, error) {
	if sessionID == "" {
		return nil, errors.New("agentloop: session ID is required")
	}
	if len(sessionID) > 256 {
		return nil, errors.New("agentloop: session ID cannot exceed 256 characters")
	}
	return &Session{agent: a, sessionID: sessionID, messages: copyMessages(messages)}, nil
}

func newSessionID() string {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	return hex.EncodeToString(id[:])
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
	if a.RequestTimeout < 0 {
		err := errors.New("agentloop: request timeout cannot be negative")
		s.emit(Event{Type: EventError, Err: err})
		return Result{}, err
	}
	if a.ToolTimeout < 0 {
		err := errors.New("agentloop: tool timeout cannot be negative")
		s.emit(Event{Type: EventError, Err: err})
		return Result{}, err
	}
	if a.MaxRetries < 0 {
		err := errors.New("agentloop: max retries cannot be negative")
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
		response, err := s.complete(ctx, Request{Model: model, SystemPrompt: a.SystemPrompt, ReasoningEffort: a.ReasoningEffort, OutputSchema: a.OutputSchema, Messages: messages, Tools: definitions, SessionID: s.sessionID})
		if err != nil {
			s.emit(Event{Type: EventError, Step: step, Err: err})
			return result, fmt.Errorf("agentloop: complete: %w", err)
		}
		result.Steps = step
		result.Usage.InputTokens += response.Usage.InputTokens
		result.Usage.OutputTokens += response.Usage.OutputTokens
		result.Usage.CachedTokens += response.Usage.CachedTokens
		result.Usage.CacheCreationTokens += response.Usage.CacheCreationTokens
		result.Usage.Cost += response.Usage.Cost
		messages = append(messages, Message{Role: "assistant", Content: response.Content, ToolCalls: response.ToolCalls})
		result.Messages = messages
		s.emit(Event{Type: EventModelResponse, Step: step, Content: response.Content, Usage: response.Usage})
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
			toolContext := ctx
			cancel := func() {}
			if a.ToolTimeout > 0 {
				toolContext, cancel = context.WithTimeout(ctx, a.ToolTimeout)
			}
			output, err := tool.Execute(toolContext, call.Arguments)
			if err == nil && toolContext.Err() != nil {
				err = toolContext.Err()
			}
			cancel()
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

func (s *Session) complete(ctx context.Context, request Request) (Response, error) {
	for attempt := 0; ; attempt++ {
		requestContext := ctx
		cancel := func() {}
		if s.agent.RequestTimeout > 0 {
			requestContext, cancel = context.WithTimeout(ctx, s.agent.RequestTimeout)
		}
		response, err := s.agent.Provider.Complete(requestContext, request)
		cancel()
		if err == nil || attempt == s.agent.MaxRetries || !isRetryable(err) || ctx.Err() != nil {
			return response, err
		}
		if err := waitRetry(ctx, attempt); err != nil {
			return Response{}, err
		}
	}
}

func isRetryable(err error) bool {
	var retryable interface{ Retryable() bool }
	return errors.As(err, &retryable) && retryable.Retryable()
}

func waitRetry(ctx context.Context, attempt int) error {
	delay := 100 * time.Millisecond * time.Duration(1<<attempt)
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (s *Session) emit(event Event) {
	if s.agent.EventHandler != nil {
		s.agent.EventHandler(event)
	}
}

func copyMessages(messages []Message) []Message {
	copied := make([]Message, len(messages))
	for i, message := range messages {
		if message.ToolCalls != nil {
			message.ToolCalls = append([]ToolCall(nil), message.ToolCalls...)
			for j, call := range message.ToolCalls {
				call.Arguments = append(json.RawMessage(nil), call.Arguments...)
				message.ToolCalls[j] = call
			}
		}
		copied[i] = message
	}
	return copied
}
