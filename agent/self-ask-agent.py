from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.llms import Fireworks
from langchain_community.tools.tavily_search import TavilyAnswer
import os
from dotenv import load_dotenv

load_dotenv()

# For using langsmith to debug purpose
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

# For using Tavily
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')

tools = [TavilyAnswer(max_results=1, name="Intermediate Answer")]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/self-ask-with-search")

# Choose the LLM that will drive the agent
llm = Fireworks()

# Construct the Self Ask With Search Agent
agent = create_self_ask_with_search_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "who is the boyfriend of the singer who sang a song name Shake it off?"})