from lyzr import Studio

# Initialize the SDK
studio = Studio(api_key="your-api-key")

# Create an agent
agent = studio.create_agent(
    name="My Assistant",
    provider="gpt-4o",
    role="Helpful assistant",
    goal="Help users with their questions",
    instructions="Be concise and accurate"
)

# Run the agent
response = agent.run("What is machine learning?")
print(response.response)