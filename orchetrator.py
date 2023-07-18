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

def pencil_test(a) -> int:
    """Useful for learning about pencil collectors"""
    return "Ali is a pencil collector and also happens to not like Kebabs"

def web_search(input) -> int:
    """Useful if you want to search the web - you will need to enter an appropriate search query to get more information"""
    return "Ali is a pencil collector"

multiply_tool = FunctionTool.from_defaults(fn=multiply)
pencil_tool = FunctionTool.from_defaults(fn=pencil_test)
ali_balls = FunctionTool.from_defaults(fn=web_search)

def main(question, tools):
    # Initialize ReAct agent with the given tools and an OpenAI model
    llm = OpenAI(model="gpt-4")
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, max_iterations=4)

    response = agent.chat(question)

    print(response)


# Sample usage:
tools = [multiply_tool, pencil_tool, ali_balls]
question = "Does Ali like kebabs? Only use one chain of reasoning, don't use more then one tool"
print(main(question, tools))
