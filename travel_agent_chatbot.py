import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Fetch the API key and Endpoint from the environment variables
api_key = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("OPENAI_ENDPOINT") 

# Validate API key before proceeding
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is missing.Please setup in .env file.")

# Initialize the LLM with streaming enabled
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    base_url=endpoint if endpoint else None,
    temperature=0,
    streaming=True
)

# Defining System prompt
SYSTEM_PROMPT = """You are an expert Travel Agent Chatbot. 
Your only job is to answer only travel-related questions (flights, hotels, destinations, trips, travel tips).

Critical Rule: If users asks ANY non-travel question e.g., coding, math, general history, cooking you MUST respond EXACTLY with the following sentence:
"I can't help with it."
"""

# LangChain prompt templates to structure users' input
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# Create the execution chain
chain = prompt_template | llm

print("Welcome to the Travel Agent Chatbot!\n")

# Maintain conversation messages in a Python List
history_list = []
conversation_turns = 0

# Starting the continuous chat loop
while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['quit', 'exit']:
        print("Safe travels! Goodbye.")
        break
        
    if not user_input:
        continue
        
    print("Agent: ", end="", flush=True)
    
    # Responses streaming token-by-token output
    full_response = ""
    for chunk in chain.stream({"history": history_list, "user_input": user_input}):
        content = chunk.content
        print(content, end="", flush=True)
        full_response += content
        
    # Append current interaction to the manual memory list
    history_list.append(HumanMessage(content=user_input))
    history_list.append(AIMessage(content=full_response))
    conversation_turns += 1
    
    # After every 5 conversations, summarize the chat history
    if conversation_turns >= 5:
        print("\n [System: Reached limit of 5 conversations. Summarizing conversation history to reduce context input size.]")
        
        # Replacing the old summarize_conversation function
        summary_prompt = "Provide highly concise summary of following travel conversation. Include  user's destination, preferences, and key suggestion provided:\n\n"
        
        for msg in history_list:
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            summary_prompt += f"{role}: {msg.content}\n"
                
        summary_request = [HumanMessage(content=summary_prompt)]
        summary_response = llm.invoke(summary_request)
        summary_text = summary_response.content
        
        # Replace older messages with the summary to act as the new memory base
        history_list = [
            AIMessage(content=f"Summary of previous conversations: {summary_text}")
        ]
        
        # Reset the turn counter
        conversation_turns = 0
        
        print("[System: Context compressed successfully.]\n")
