from llama_index.tools import FunctionTool
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from dotenv import load_dotenv
import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
import requests
import cohere

load_dotenv()
openai_embeddings = OpenAIEmbeddings()

cohere_api_key = os.environ.get("COHERE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
project_docs_link = os.environ.get("PROJECT_DOCS_LINK")
activeloop_token = os.environ.get("ACTIVELOOP_TOKEN")

print(openai_api_key)
# embeddings = OpenAIEmbeddings()
openai.api_key = openai_api_key
co = cohere.Client(cohere_api_key)

#### DeepLake####
# This function retrieves the DeepLake datasets


def retrieve_datasets_deeplake(list_of_dl_datasets):
    print("retrieve_datasets_deeplake hit")
    dl_dict = {}  # Define an empty dictionary to store the retrieved datasets
    for dataset in list_of_dl_datasets:  # Loop through the list of dataset names provided
        try:
            # Create a DeepLake object for each dataset and add it to the dictionary
            db = DeepLake(
                dataset_path=f"hub://tali/{dataset}", read_only=True, embedding_function=openai_embeddings)
            # Map dataset name to the corresponding DeepLake object
            dl_dict[dataset] = db
        except AssertionError:  # Catch any AssertionError that may arise during the DeepLake object creation
            print(AssertionError)  # Print the AssertionError
    return dl_dict  # Return the dictionary of DeepLake objects


db = retrieve_datasets_deeplake("test_balancer_docs")

##############




#### Cohere####
response = co.rerank(
    model='rerank-english-v2.0',
    query=query,
    documents=evaluated_docs,
    top_n=5,
)
##############



# Define Tools
def ali_color() -> int:
    """Useful to understand what Ali's favorite color is"""
    print("ali_color hit")
    return "Terquoise"


def ali_sport() -> int:
    """Useful to understand what Ali's favorite sport is"""
    print("ali_sport hit")
    return "Cycling"


def ali_food() -> int:
    """Useful to understand what Ali's favorite food is"""
    print("ali_food hit")
    return "Kabobs"


def get_joke():
    """Useful for getting jokes"""
    response = requests.get('https://v2.jokeapi.dev/joke/Any')
    if response.status_code == 200:
        data = response.json()
        if 'joke' in data:  # for single part jokes
            return data['joke']
        elif 'setup' in data and 'delivery' in data:  # for two-part jokes
            return f"{data['setup']} - {data['delivery']}"
        else:
            return 'No joke found'
    else:
        return 'Something went wrong'


ali_color = FunctionTool.from_defaults(fn=ali_color)
ali_sport = FunctionTool.from_defaults(fn=ali_sport)
ali_food = FunctionTool.from_defaults(fn=ali_food)
get_joke = FunctionTool.from_defaults(fn=get_joke)

# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# initialize ReAct agent
agent = ReActAgent.from_tools(
    [ali_color, ali_sport, ali_food, get_joke], llm=llm, verbose=True)


####Evaluate if the question is answered by the inital docs search#####

# query = "what is balancer?"

# prompt = f"""You are Tali, a developer support bot. Your role is to assist with project development and problem-solving. You do this by synthesizing context sources to answer a user query. You MUST follow these principles:

#     1. Only provide accurate and helpful information.
#     2. When synthesizing data from different context sources, integrate the information into a coherent response. Avoid mentioning the specific origin of each piece of information, even if the sources are varied (e.g., don't specify if information came from Google, project documentation, etc.).
#     3. Be succinct and direct in answering questions. Aim for brevity and clarity.
#     4. If unsure about the answer, admit uncertainty instead of providing potentially inaccurate information. Always base your answers on the provided Context sources.
#     5. Include code examples in your answer when it's relevant. Use markdown to format code.
#     6. Never provide code in non-markdown format.
#     7. If the context sources is indicated as '[]', which means no sources are available, say: "I don't know, please refer to {project_docs_link}".
#     8. Don't confuse the user with unnecessary content. Ensure your responses are useful and directly relevant.
#     9. If there are links, include them in your response. When citing sources, number them and place them at the end of your response.

#     The user's query is: {query}
#     Context sources, which include documentation, are: {docs}

#     There may be a chat history with previous questions and answers. Use this history if it's relevant to the question."""

# MAX_RETRIES = 3
# SLEEP_TIME = 1  # in seconds
# print("prompt bool", prompt)
# System prompt included at base. Array of messages from Supabase looped through and passed to OpenAI chat engine

# for _ in range(MAX_RETRIES):
#     try:
#         completion = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo-16k",
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0
#         )
#         # return the cleaned text from the model
#         print('completion', completion['choices'][0]['message']['content'])
#         completion = completion['choices'][0]['message']['content']
#     except Exception as e:
#         print(f"Error calling OpenAI API: {e}")


####Evaluate if the question is answered by the inital docs search#####


#########


###Agent tool if question not answered###
agent.chat("""Tell me a joke.
           
           You can only use tools to answer the question. You can only use one tool. Do not answer with anything outside of information from the tools.""")
print("agent.chat_history", agent.chat_history)
