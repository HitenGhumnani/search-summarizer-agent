import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Load API keys
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")


# Step 1: Setup LLM (Gemma via OpenRouter)
llm = ChatOpenAI(
    model="google/gemma-3n-e2b-it:free",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)


# Step 2: Setup Tavily Tool
tavily = TavilySearch(api_key=tavily_api_key)

search_tool = Tool(
    name="tavily_search",
    func=tavily.run,
    description="Search the web for up-to-date information",
)

tools = [search_tool]


# Step 3: Create ReAct Agent (NEW WAY)-
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Step 4: Run loop
while True:
    query = input("\nWhat would you like to know? (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    response = agent_executor.invoke({"input": query})
    print("\n🔎 Final Answer:\n", response["output"])
