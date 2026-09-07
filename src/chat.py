import ollama
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag-demo"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3.5:4b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

client = QdrantClient(url=QDRANT_URL)

reranker = CrossEncoder(RERANKER_MODEL)


def retrieve(query, top_k=5):
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query
    )

    query_embedding = response["embedding"]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    )

    return results.points


def rerank(query, results, top_k=2):
    pairs = [
        [query, result.payload["text"]]
        for result in results
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


def generate_answer(query):
    retrieved = retrieve(query, top_k=5)

    ranked = rerank(
        query,
        retrieved,
        top_k=2
    )

    context = "\n\n".join(
        result.payload["text"]
        for result, score in ranked
    )

    prompt = f"""
Answer the question using only the context below.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"], retrieved, ranked


if __name__ == "__main__":
    question = input("Enter your question: ")

    answer, retrieved, ranked = generate_answer(question)

    print("\n" + "=" * 60)
    print("QDRANT RETRIEVAL")
    print("=" * 60)

    for i, result in enumerate(retrieved, start=1):
        print(
            f"{i}. {result.payload['source']} "
            f"(score: {result.score:.4f})"
        )

    print("\n" + "=" * 60)
    print("AFTER RE-RANKING")
    print("=" * 60)

    for i, (result, score) in enumerate(ranked, start=1):
        print(
            f"{i}. {result.payload['source']} "
            f"(rerank score: {score:.4f})"
        )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)