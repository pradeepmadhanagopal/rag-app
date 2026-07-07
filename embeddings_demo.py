from sentence_transformers import SentenceTransformer, util

# Small, fast, well-regarded model — downloads ~90MB on first run
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do neural networks learn?",
    "Backpropagation adjusts weights using gradients of the loss.",
    "The best pizza in Newcastle is at a place on Darby Street.",
    "Gradient descent minimises the loss function step by step.",
]

embeddings = model.encode(sentences)

print(f"Each sentence became a vector of {len(embeddings[0])} numbers\n")

query = sentences[0]
print(f"Query: {query}\n")

for sentence, emb in zip(sentences[1:], embeddings[1:]):
    score = util.cos_sim(embeddings[0], emb).item()
    print(f"  {score:.3f}  {sentence}")
