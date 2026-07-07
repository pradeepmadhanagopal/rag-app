from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import anthropic 

from loader import chunk_text, load_pdf

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

print("loading and chunking documets...")
text = load_pdf("docs/4a_Convolution.pdf")
chunks = chunk_text(text)

print(f"embedding {len(chunks)} chunks...")
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)

client = anthropic.Anthropic()

def retrieve(question: str, k: int = 3, threshold: float =0.5) -> list[str]:
    query_embedding = model.encode(question)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    ranked = sorted(zip(scores.tolist(), chunks), key=lambda x: x[0], reverse=True)

    selected = [(s,c) for s,c in ranked[:k] if s > threshold]

    print("\n Retrieved Chunks:")
    for s,c in selected:
        print(f" score={s:.3f} preview: {c[:60]}!r")

    return [c for _,c in selected]

def answer(question: str) -> str:
    context = "\n\n---\n\n".join(retrieve(question))

    prompt = f"""Answer the question using ONLY the context below from lecture notes.
If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    while True:
        q = input("\nAsk about the lecture (or 'quit'): ")
        if q.lower() == "quit":
            break
        print("\n" + answer(q)) 