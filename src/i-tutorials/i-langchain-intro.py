"""                     Import libraries.                       """
# Import necessary libraries.
import os
import requests
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from IPython.display import display

"""
Resources:
    - https://docs.langchain.com/oss/python/langchain/overview
    - https://www.youtube.com/watch?v=vzJOAnwIokM
    - https://www.youtube.com/watch?v=J7j5tCB_y4w

"""


"""                     Setup.                       """
# Load environment variables from .env file.
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create the model_llm instance with NVIDIA model
model_llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

"""
ChatNVIDIA.get_available_models()
# model_llm.get_available_models()
"""


"""                     Agents.                       """
# Create a simple agent.
agent = create_agent(
    model=model_llm,
    tools=[],
    system_prompt="You are a helpful assistant."
)
display(agent)

# Define a simple tool/function to get the weather for a city.
@tool('get_weather', description="Get the current weather for a given city.", return_direct=False)
def get_Weather(city: str) -> str:
    """Get the current weather for a given city."""
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

# Create an agent with the weather tool.
agent_with_tool = create_agent(
    model=model_llm,
    tools=[get_Weather],
    system_prompt="You are a helpful weather assistant, who always cracks jokes and is humrous while remaining helpful."
)
display(agent_with_tool)

# Run the agent with a query.
response = agent_with_tool.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the current weather in New York?"
        }
    ]
})

print(response)
print(f"\n{'=' * 100}\n")
print(response['messages'][-1].content)

# Alternate way to run the agent with a query.
response = agent_with_tool.invoke(
    {
        "messages": "What is the current weather in New York?"
    }
)

print(f"\n{'=' * 100}\n")
print(response['messages'][-1].content)


"""                     Model Integrations.                       """
# Initialize different chat models from various providers.
model_nvidia = init_chat_model(model="nvidia:meta/llama-3.1-405b-instruct")
model_google = init_chat_model(model="google_genai:gemini-2.5-flash-lite")
model_groq = init_chat_model(model="groq:llama-3.3-70b-versatile")

# Invoke the models with a simple query.
query = "What is the capital of France?"

response_nvidia = model_nvidia.invoke(query)
response_google = model_google.invoke(query)
response_groq = model_groq.invoke(query)

# Print the responses.
print(f"\n{'=' * 100}\n")
print(f"NVIDIA response: {response_nvidia.content}\n")
print(f"Google response: {response_google.content}\n")
print(f"GROQ response: {response_groq.content}")

# Directly initialize and invoke the models.
model_nvidia = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
model_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
model_groq = ChatGroq (model="llama-3.3-70b-versatile")

# Invoke the models with a simple query.
query = "What is the capital of Australia?"

response_nvidia = model_nvidia.invoke(query)
response_google = model_google.invoke(query)
response_groq = model_groq.invoke(query)

# Print the responses.
print(f"\n{'=' * 100}\n")
print(f"NVIDIA response: {response_nvidia.content}\n")
print(f"Google response: {response_google.content}\n")
print(f"GROQ response: {response_groq.content}")


