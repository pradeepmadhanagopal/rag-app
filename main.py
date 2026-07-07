import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

def ask(question: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text

if __name__ == "__main__":
    print(ask("In one sentence, what is retrieval-augmented generation?"))
