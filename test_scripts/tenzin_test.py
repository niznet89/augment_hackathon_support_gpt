from llama_index.tools import FunctionTool
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from dotenv import load_dotenv
import os
import openai

load_dotenv()

cohere_api_key = os.environ.get("COHERE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API")

print(openai_api_key)

os.environ['OPENAI_API_KEY'] = openai_api_key
#os.environ['ACTIVELOOP_TOKEN'] = activeloop_token
#embeddings = OpenAIEmbeddings()
openai.api_key = openai_api_key



# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=add)

# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

agent.chat("What is 2 + 215123")
