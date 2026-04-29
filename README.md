# TravelAgent

Travel Agent Chatbot

Travel Agent Chatbot using (LangChain / LangGraph)

Objective: Build a simple chatbot using LangChain that acts as a Travel Agent.

Requirements:

The chatbot must answer ONLY travel-related questions (flights, hotels, destinations, trips, travel tips).
For any non-travel-related question, respond exactly with: "I can't help with it.'
Functional Requirements:

Responses must be streamed (token-by-token output).
Maintain all conversation messages in a Python List.
After every 5 conversations, summarize the chat history.
Replace older messages with the summary to reduce context size.
Technical Guidelines:

Use LangChain prompt templates.
Implement memory manually using a List.
Use any LLM supported in the bootcamp.
Deliverables:

One Python file or Jupyter Notebook.
Working demo of the chatbot
Clear comments explaining streaming, memory, and summarization logic.
