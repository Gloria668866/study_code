from python_a2a import A2AClient

llm_client = A2AClient(f"http://localhost:6666")
print(llm_client.agent_card)
result = llm_client.ask("What is the capital of France?")
print(f"LLM Response: {result}")