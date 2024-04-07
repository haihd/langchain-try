import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

# For using langsmith to debug purpose
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

promt = ChatPromptTemplate.from_template("tell me a joke about {smt}")
model = ChatAnthropic(temperature=0, model_name="claude-2.1")

chain = promt | model | StrOutputParser()
res = chain.invoke({ "smt": "dogs"})

print(res)