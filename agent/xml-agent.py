from langchain import hub
from langchain.agents import AgentExecutor, create_xml_agent
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv


load_dotenv()

# For using langsmith to debug purpose
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

# For using Tavily
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/xml-agent-convo")

# Choose the LLM that will drive the agent
llm = ChatAnthropic(model="claude-2.1")

# Construct the XML agent
agent = create_xml_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is Tavily?"})