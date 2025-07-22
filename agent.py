import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatOpenRouter
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.schema.messages import HumanMessage

# Load API keys
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Step 1: Setup Tavily Tool
tavily_tool = TavilySearchResults(api_key=tavily_api_key)
tools = [Tool.from_function(func=tavily_tool.run, name="tavily_search", description="Search the web for up-to-date information")]

# Step 2: Setup OpenRouter/Gemma LLM
llm = ChatOpenRouter(
    model="google/gemma-3n-e2b-it:free",
    openrouter_api_key=openrouter_api_key,
    headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Gemma Search Summarizer"
    }
)

# Step 3: Initialize the Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 4: Run the Agent with a prompt
query = input("What would you like to know? ")
response = agent.run(query)
print("\n🔎 Summary:\n", response)
