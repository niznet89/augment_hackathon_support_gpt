# Autonomous Agents for Support
This repository contains code to support an AI assistant bot for answering developer questions. It was built during the [Augment](https://www.augmenthack.xyz/) hackathon.

Overview
The goal of this project is to create an AI-powered assistant that can understand a developer's question and synthesize responses by retrieving and combining information from various sources.

It uses the [Llama Index](https:/ /github.com/llama-index/llama-index) along with OpenAI and [Cohere](https://cohere.com/) APIs to build an orchestrator that can:

- Search through documentation and Q&A sites
- Query the web
- Use large language models to synthesize coherent responses
- Escalate to human support if it cannot answer

## Code Structure

The main components are:

orchestrator.py - Orchestrates the overall workflow and tools using a ReAct agent
tools.py - Contains reusable tools like search functions
scrape_to_deeplake.py - Ingests docs/data into a DeepLake vector store
test_scripts/* - Scripts to test individual components
retrieve_doc_nodes.py - Functions to ingest web pages into documents
discord_reader.py - Loads Discord data

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
