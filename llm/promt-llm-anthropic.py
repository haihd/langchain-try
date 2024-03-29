import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

promt = ChatPromptTemplate.from_template("tell me a 3 jokes about {smt}")
model = ChatAnthropic(temperature=0, model_name="claude-2.1")

chain = promt | model
res = chain.invoke({ "smt": "dogs"})

print(res.content)