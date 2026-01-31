import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.tools import tool

from langgraph.prebuilt import create_react_agent

# Load keys
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# LLM (Gemma via OpenRouter)
llm = ChatOpenAI(
    model="google/gemma-3n-e2b-it:free",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Tavily instance
tavily = TavilySearch(api_key=tavily_api_key)

# Tool definition (new style)
@tool
def tavily_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return tavily.run(query)

tools = [tavily_search]

# ✅ Create ReAct agent using LangGraph
agent = create_react_agent(llm, tools)

# Run loop
while True:
    query = input("\nWhat would you like to know? (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    response = agent.invoke({"messages": [("user", query)]})

    print("\n🔎 Final Answer:\n", response["messages"][-1].content)
