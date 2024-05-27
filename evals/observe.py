from openai import OpenAI
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv
import os

load_dotenv(verbose=True, override=True)

# For using langsmith to debug purpose
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT')

openai_client = wrap_openai(OpenAI())

# This is the retriever we will use in RAG
# This is mocked out, but it could be anything we want
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

# This is the end-to-end RAG chain.
# It does a retrieval step then calls OpenAI
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
    
    {docs}""".format(docs="\n".join(docs))
    
    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-3.5-turbo",
    )

rag("where did harrison work now?")