# Autonomous Dev Support Agents 

This repository contains code from our hack at the [Augment hackathon](https://www.augmenthack.xyz/) to build an autonomous agent that can answer developer support queries.

Nothing is more frustrating then being stuck in a problem and having to wait hours to get a response. RAG pipelines involve finding the most relevant documents to a query, providing checks then synthesizing an answer. 

When a human finds a solution it's often a dynamic process that involves piecing together different types of info to find a conclusive answer. We're looking to replicate that.

## Overview

The goal of this project was to leverage [Llama Index](https:/ /github.com/llama-index/llama-index) Agent capabilities to take in a query, reason through what steps the Agent should take (like a support agent would) and fill in the gaps of their information to coherently answer. 

The goal is to mimic how a human might approach the problem by:
1) Seeing if they can answer based on their own knowledge.
2) If they can't, formulate a web search to get more information on what they don't know.
3) Search within the community to see if similar questions have been answered.
4) If, through all of the above methods they couldn't answer escalate the question.

We used Zendesk email support a demo / test case, where the bot would either answer the question or escalate to a human and update the status of the ticket accordingly. 


## Architecture overview
 

It uses the  along with OpenAI and [Cohere](https://cohere.com/) APIs to build an orchestrator that can:

- Search through documentation and Q&A sites
- Query the web
- Use large language models to synthesize coherent responses
- Escalate to human support if it cannot answer

## Code Structure

The main components are:

- orchestrator.py - Orchestrates the overall workflow and tools using a ReAct agent
- tools.py - Contains reusable tools like search functions
- scrape_to_deeplake.py - Ingests docs/data into a DeepLake vector store
- test_scripts/* - Scripts to test individual components
- retrieve_doc_nodes.py - Functions to ingest web pages into documents
- discord_reader.py - Loads Discord data

The .env file contains API keys and configuration.

## Usage

To use this system:

Update .env with your own API keys
Run scrape_to_deeplake.py to index your documentation/data
Pass a question to orchestrator.py to get a response
The orchestrator will leverage the various tools and data sources to synthesize an answer to the question.

## Customization

- Add new tools in tools.py
- Configure additional data sources in scrape_to_deeplake.py
- Tweak the agent configuration in orchestrator.py
