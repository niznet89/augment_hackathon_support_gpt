from flask import Flask, request
from llama_index.tools import FunctionTool
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
import requests
import cohere
from llama_index import VectorStoreIndex, ServiceContext, LLMPredictor
from llama_index.readers.deeplake import DeepLakeReader
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
import random
import openai
from llama_index.indices.postprocessor.cohere_rerank import CohereRerank
import json
from tools import search_discord, google_search, ticket_escalation, ticket_solved
load_dotenv()
openai_embeddings = OpenAIEmbeddings()

cohere_api_key = os.environ.get("COHERE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
project_docs_link = os.environ.get("PROJECT_DOCS_LINK")
activeloop_token = os.environ.get("ACTIVELOOP_TOKEN")

print(openai_api_key)
openai.api_key = openai_api_key
co = cohere.Client(cohere_api_key)
app = Flask(__name__)
url = "https://hooks.zapier.com/hooks/catch/14962368/3mlonub/"

#### DeepLake####
query_vector = [random.random() for _ in range(1536)]
reader = DeepLakeReader()
documents = reader.load_data(
    query_vector=query_vector,
    dataset_path="hub://tali/ocean_protocol_docs",
    limit=30,
)
documents = documents

def email_received(query):
    global documents

    # Assuming documents_list is your list of Document objects
    documents = [{"text": doc.text} for doc in documents]

    response = co.rerank(
        model='rerank-english-v2.0',
        query=query,
        documents=documents,
        top_n=3,
    )

    print("response", response)

    documents_content = [result.document['text']
                         for result in response.results]

    print('documents_content', documents_content)

    # Define Tools
    google = FunctionTool.from_defaults(fn=google_search)
    ticket = FunctionTool.from_defaults(fn=ticket_escalation)

    # initialize llm
    llm = OpenAI(model="gpt-3.5-turbo-0613")

    # initialize ReAct agent
    agent = ReActAgent.from_tools(
        [google, ticket], llm=llm, verbose=True, max_iterations=2)

    #### Evaluate if the question is answered by the inital docs search#####

    prompt = f"""You are Tali, a developer support bot. Your role is to assist with project development and problem-solving. You do this by synthesizing context sources to answer a user query. You MUST follow these principles:

        1. Only provide accurate and helpful information.
        2. When synthesizing data from different context sources, integrate the information into a coherent response. Avoid mentioning the specific origin of each piece of information, even if the sources are varied (e.g., don't specify if information came from Google, project documentation, etc.).
        3. Be succinct and direct in answering questions. Aim for brevity and clarity.
        4. If unsure about the answer, admit uncertainty instead of providing potentially inaccurate information. Always base your answers on the provided Context sources.
        5. Include code examples in your answer when it's relevant. Use markdown to format code.
        6. Never provide code in non-markdown format.
        7. If the context sources is indicated as '[]', which means no sources are available, say: "I don't know, please refer to {project_docs_link}".
        8. Don't confuse the user with unnecessary content. Ensure your responses are useful and directly relevant.
        9. If there are links, include them in your response. When citing sources, number them and place them at the end of your response.

        The user's query is: {query}
        Context sources, which include documentation, are: {documents_content}

        Remember you can only rely on information given to you based on the context sources."""

    MAX_RETRIES = 3
    SLEEP_TIME = 1 
    print("prompt bool", prompt)

    for _ in range(MAX_RETRIES):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            # return the cleaned text from the model
            print('completion', completion['choices'][0]['message']['content'])
            initalAnswer = completion['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")

    #### Evaluate if the question is answered by the inital docs search#####
    prompt2 = f"""
        You are Tali, a developer support bot. Your role is to assist with project development and problem-solving. A user has asked a query, and an answer has been created based on synthesising multiple data sources.

        Your job is to evaluate if the provided answer satisfies the question just based on the context provided. 
        
        You MUST follow these principles:
        
        1) If the query is not being answered by the context respond with "No" 
        2) If the query is being answered sufficiently based on the context respond with "Yes"
        3) If you are not at least 80% sure that the answer is correct respond with "No"
        4) If you are at least 80% sure that the answer is correct respond with "Yes"
        5) If the answer says "I don't know", respond with "No"
        
        query: {query}

        answer: {initalAnswer}

        context: {documents_content}

        Remember, you can only use the context provided to answer the question. You can only reply with a "Yes" or "No"."""

    MAX_RETRIES = 3
    SLEEP_TIME = 1  
    print("prompt bool", prompt2)

    for _ in range(MAX_RETRIES):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt2}
                ],
                temperature=0
            )
            print('completion', completion['choices'][0]['message']['content'])
            evaluationResults = completion['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")

    print("evaluationResults", evaluationResults)

    ### Agent tool if question not answered###

    if evaluationResults == "Yes":
        print("Evaluation Results: ", evaluationResults)

        ticket_solved('alikaghatx@gmail.com', evaluationResults)

    else:
        agent.chat(f"""Use the following tools to answer this question:

            Query:{query}

            You can only use tools to answer the question. You can only use one tool. Do not answer with anything outside of information from the tools.

            You'll have 2 iterations to ask questions of the different tools. If you're on the 2nd iteration and you don't have an answer USE the Ticket Escalation tool.""")
        print("Agent Chat History: ", agent.chat_history)


def get_input():
    while True:
        try:
            query = input('Please enter an your support question: ')
            if not query:  
                raise ValueError('No input provided')
            else:
                email_received(query)
                break  
        except ValueError as e:
            print(e)


get_input()
