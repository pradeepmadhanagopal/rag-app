from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import anthropic
import chromadb

from loader import load_pdf, chunk_text

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
client = anthropic.Anthropic()


db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("lecture_notes", metadata={"hnsw:space": "cosine"})


if collection.count() == 0:
    print("Empty collection — indexing documents...")
    text = load_pdf("docs/4a_Convolution.pdf")
    chunks = chunk_text(text)
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks)
    collection.add(
        ids=[f"conv-{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=[{"source": "4a_Convolution.pdf", "position": i} for i in range(len(chunks))],
    )
else:
    print(f"Loaded existing index: {collection.count()} chunks")

def retrieve_with_scores(question: str, k: int = 3, threshold: float = 0.65) -> list[tuple[str, float]]:
    q_emb = model.encode(question)
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    return [(doc, 1 - dist / 2) for doc, dist in zip(docs, distances) if (1 - dist / 2) >= threshold]    



def retrieve(question: str, k: int = 3, threshold: float = 0.65) -> list[str]:
   return [doc for doc, _ in retrieve_with_scores(question, k, threshold)]



def answer(question: str) -> str:
    retrieved = retrieve(question)
    if not retrieved:
        return "I couldn't find anything relevant to that in the loaded documents."

    context = "\n\n---\n\n".join(retrieved)

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