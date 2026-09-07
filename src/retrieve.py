import ollama
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag-demo"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

client = QdrantClient(url=QDRANT_URL)

def search(query, top_k=3):
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

if __name__ == "__main__":
    query = input("Enter your question: ")

    results = search(query)

    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 60)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score : {result.score:.4f}")
        print(f"Source: {result.payload['source']}")
        print("\nText:")
        print(result.payload["text"])
        print("-" * 60)