from llama_index.tools import FunctionTool
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from dotenv import load_dotenv
import os
import openai
from tools import search_discord, google_search

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

discord_tool = FunctionTool.from_defaults(fn=search_discord)
search_tool = FunctionTool.from_defaults(fn=google_search)


def main(question, tools):
    # Initialize ReAct agent with the given tools and an OpenAI model
    llm = OpenAI(model="gpt-4")
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, max_iterations=5)

    response = agent.chat(question)


    print(response)


# Sample usage:
tools = [discord_tool, search_tool]
question = """You are an expert technical support agent. You have a set of tools available to be able to answer the users query. Based on previous answers, change the queries you're asking to get more useful information.

            You'll have 5 iterations to ask questions to the different data sources.

            QUESTION: Will the Balancer deployment on Base have Weighted Pools that are integrated into the veBAL system right away?"""
print(main(question, tools))
