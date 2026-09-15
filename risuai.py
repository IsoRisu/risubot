# main.py
# Simple text-in / text-out wrapper around freeflow-llm.
# Test this standalone first, then import `ask()` from your Discord bot.

import os

# Import your key from botkey.py and expose it to freeflow-llm
from botkey import GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

from freeflow_llm import FreeFlowClient

def ask(prompt: str) -> str:
    """
    Send a single prompt to the free LLM chain and return the reply as a string.
    This is the function you'll reuse inside your Discord bot.
    """
    with FreeFlowClient() as client:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model="gemini-3.6-flash"  # <-- add this line
        )
        return response.content

def main():
    """Interactive input/output loop for testing."""
    print("FreeFlow LLM test. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        try:
            reply = ask(user_input)
            print(f"Bot: {reply}\n")
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()