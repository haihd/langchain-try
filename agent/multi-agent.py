import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

load_dotenv(verbose=True, override=True)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')


# model = ChatAnthropic(temperature=0, model_name="claude-2.1")
model = ChatOpenAI(temperature=0)

graph = MessageGraph()

graph.add_node("oracle", model)
graph.add_edge("oracle", END)

graph.set_entry_point("oracle")

runnable = graph.compile()

runnable.invoke(HumanMessage("What is 1 + 1?"))