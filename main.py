from langchain_core.messages import HumanMessage # pyright: ignore[reportMissingImports]
from langchain_groq import ChatGroq # pyright: ignore[reportMissingImports]
from langchain.tools import tool # pyright: ignore[reportMissingImports]
from langgraph.prebuilt import create_react_agent # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import os

load_dotenv()

def main():
    # Use Groq's free LLaMA 3 model
    model = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",  # You can also try "mixtral-8x7b-32768"
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    tools = []  # No tools for now
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me anything.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        print("\nAssistant: ", end="", flush=True)
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    if hasattr(message, "content"):
                        print(message.content, end="", flush=True)
        print()

if __name__ == "__main__":
    main()
