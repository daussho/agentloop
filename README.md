# agentloop

A small Go library for running autonomous, tool-using LLM loops.

```go
agent := agentloop.NewOpenAICompatibleAgent(
    agentloop.WithBaseURL("https://api.openai.com/v1"),
    agentloop.WithKey(os.Getenv("OPENAI_API_KEY")),
    agentloop.WithModel("gpt-5"),
    agentloop.WithSystemPrompt("Solve the task using available tools."),
    agentloop.WithReasoningEffort(agentloop.ReasoningEffortMedium),
    agentloop.WithOutputSchema(json.RawMessage(`{
        "type": "object",
        "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
        "required": ["title", "body"],
        "additionalProperties": false
    }`)),
    agentloop.WithTools(myTool),
    agentloop.WithMaxSteps(20),
)

result, err := agent.Run(ctx, "Complete the task")
```

## Configuration

| Option | Required | Description |
| --- | --- | --- |
| `WithBaseURL(url)` | Yes | OpenAI-compatible API base URL, such as `https://api.openai.com/v1`. |
| `WithKey(key)` | Usually | API key sent as a Bearer token. |
| `WithModel(model)` | Yes | Model used for completions. |
| `WithSystemPrompt(prompt)` | No | System instruction sent before conversation messages. |
| `WithReasoningEffort(effort)` | No | Provider reasoning setting: `ReasoningEffortLow`, `ReasoningEffortMedium`, or `ReasoningEffortHigh`. |
| `WithOutputSchema(schema)` | No | JSON Schema enforced by providers that support OpenAI structured outputs. |
| `WithTools(tools...)` | No | Tools the model may call. |
| `WithMaxSteps(n)` | No | Maximum model/tool-loop iterations; defaults to `20` and cannot be negative. |
| `WithRequestTimeout(timeout)` | No | Timeout for each provider call; defaults to 2 minutes. Set zero to disable it. |
| `WithMaxRetries(n)` | No | Retries transient provider failures (`429`, `5xx`, and network errors); defaults to 2. Set zero to disable retries. |
| `WithToolTimeout(timeout)` | No | Timeout for each tool call; defaults to 30 seconds. Set zero to disable it. |
| `WithEventHandler(handler)` | No | Synchronous callback for model, tool, error, and completion events. |

When tools are used, the output schema is sent on every model request. Tool arguments follow each tool's own schema; the structured output is the final assistant response in `result.Output`.

Retries use a 100ms exponential backoff and never retry tools. A timeout cancels the context passed to the provider or tool; implementations must honor that context.

Use an event handler to observe a running session:

```go
agentloop.WithEventHandler(func(event agentloop.Event) {
    log.Printf("step=%d type=%s", event.Step, event.Type)
})
```

For a multi-turn conversation, use a session. Concurrent calls to `Session.Run` wait and execute in order.

```go
session := agent.NewSession()
_, _ = session.Run(ctx, "Research the topic")
result, err := session.Run(ctx, "Now summarize it")
messages := session.Messages()
```

`Run` stops only when the provider returns a response without tool calls, an operation fails, or `MaxSteps` is reached. MCP, planning, streaming, and retries are intentionally deferred.
