# agentloop

A small Go library for running autonomous, tool-using LLM loops.

```go
agent := agentloop.NewOpenAICompatibleAgent(
    agentloop.WithBaseURL("https://api.openai.com/v1"),
    agentloop.WithKey(os.Getenv("OPENAI_API_KEY")),
    agentloop.WithModel("gpt-5"),
    agentloop.WithSystemPrompt("Solve the task using available tools."),
    agentloop.WithReasoningEffort("medium"),
    agentloop.WithTools(myTool),
    agentloop.WithMaxSteps(20),
)

result, err := agent.Run(ctx, "Complete the task")
```

`Run` stops only when the provider returns a response without tool calls, an operation fails, or `MaxSteps` is reached. MCP, planning, streaming, and retries are intentionally deferred.
