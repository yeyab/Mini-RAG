import chromadb, ollama

EMBED = "qwen3-embedding:4b"
LLM = "qwen2.5-coder:7b"
CONSIGNE = "Instruct: Given a question, retrieve passages that answer it\nQuery: "

def embed(textes):
    return ollama.embed(model=EMBED, input=textes)["embeddings"]

client = chromadb.PersistentClient(path="./ma_base")
col = client.get_or_create_collection("cours_ml", metadata={"hnsw:space": "cosine"})

docs = [
    "L'overfitting se corrige avec plus de données variées et moins d'époques.",
    "Teachable Machine exporte en TensorFlow.js, TensorFlow Lite et Keras.",
    "Un batch est un groupe d'échantillons traités avant chaque correction des poids.",
    "Le biais d'un neurone décale le seuil à partir duquel il s'active.",
]
if col.count() == 0:
    col.add(ids=[f"d{i}" for i in range(len(docs))], documents=docs, embeddings=embed(docs))
    print("Base indexée :", col.count(), "documents")

question = input("Ta question : ")
res = col.query(query_embeddings=embed([CONSIGNE + question]), n_results=2)

print("\nPassages trouvés :")
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"  - ({1 - dist:.3f}) {doc}")

contexte = "\n".join(res["documents"][0])
rep = ollama.chat(model=LLM, messages=[
    {"role": "system", "content": "Réponds en français, uniquement à partir du contexte fourni."},
    {"role": "user", "content": f"Contexte :\n{contexte}\n\nQuestion : {question}"},
])
print("\nRéponse :", rep["message"]["content"])