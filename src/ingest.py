from pathlib import Path
import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance




QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag-demo"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

DOCUMENTS_DIR = Path("data/documents")




client = QdrantClient(url=QDRANT_URL)




def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks




def get_embedding(text):

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )

    return response["embedding"]




def ingest_documents():

    documents = list(DOCUMENTS_DIR.glob("*.txt"))

    if not documents:
        print("No documents found.")
        return

    print(f"Found {len(documents)} documents.")

    
    print("Checking embedding model...")

    test_embedding = get_embedding("test")

    vector_size = len(test_embedding)

    print(f"Embedding dimension: {vector_size}")

    
    if client.collection_exists(COLLECTION_NAME):

        print("Deleting existing collection...")

        client.delete_collection(COLLECTION_NAME)

    print("Creating Qdrant collection...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    points = []

    for document in documents:

        print(f"\nProcessing: {document.name}")

        text = document.read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(text)

        print(f"Created {len(chunks)} chunks.")

        for index, chunk in enumerate(chunks):

            print(
                f"Creating embedding "
                f"{index + 1}/{len(chunks)}..."
            )

            embedding = get_embedding(chunk)

            point = PointStruct(
                id=str(uuid.uuid4()),

                vector=embedding,

                payload={
                    "text": chunk,
                    "source": document.name,
                    "chunk_id": index
                }
            )

            points.append(point)

    print("\nUploading vectors to Qdrant...")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(points)}")
    print(f"Collection: {COLLECTION_NAME}")




if __name__ == "__main__":
    ingest_documents()