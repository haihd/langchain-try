import os
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# For using langsmith to debug purpose
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

promt = ChatPromptTemplate.from_template("tell me a joke about {smt}")
model = ChatOllama(model="llama2", base_url="http://localhost:11434")

chain = promt | model | StrOutputParser()
res = chain.invoke({ "smt": "cats"})

print(res)