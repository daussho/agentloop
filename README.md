# agentloop

A small Go library for running autonomous, tool-using LLM loops.

```go
agent := agentloop.NewOpenAICompatibleAgent(
    agentloop.WithBaseURL("https://api.openai.com/v1"),
    agentloop.WithKey(os.Getenv("OPENAI_API_KEY")),
    agentloop.WithModel("gpt-5"),
    agentloop.WithSystemPrompt("Solve the task using available tools."),
    agentloop.WithReasoningEffort(agentloop.ReasoningEffortMedium),
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
| `WithTools(tools...)` | No | Tools the model may call. |
| `WithMaxSteps(n)` | No | Maximum model/tool-loop iterations; defaults to `20` and cannot be negative. |

For a multi-turn conversation, use a session. Concurrent calls to `Session.Run` wait and execute in order.

```go
session := agent.NewSession()
_, _ = session.Run(ctx, "Research the topic")
result, err := session.Run(ctx, "Now summarize it")
messages := session.Messages()
```

`Run` stops only when the provider returns a response without tool calls, an operation fails, or `MaxSteps` is reached. MCP, planning, streaming, and retries are intentionally deferred.
