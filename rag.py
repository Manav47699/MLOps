import os
import chromadb

# chromadb client and persistent collection
CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "fellowship_docs"

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def simple_chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    """Split text into smaller overlapping character chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    return [c for c in chunks if len(c) > 0]


def load_documents_from_folder(folder_path: str = "./docs"):
    """Read all .txt files from the docs folder."""
    documents = []

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist!")
        return documents

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({"filename": filename, "content": content})
                    print(f"Loaded file: {filename}")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    return documents


def ingest_documents(folder_path: str = "./docs"):
    """Load text files, chunk them, and store into chromadb."""
    print("Starting document ingestion...")
    documents = load_documents_from_folder(folder_path)

    if not documents:
        print("No documents found to ingest.")
        return 0

    chunk_ids = []
    chunk_texts = []
    chunk_metadatas = []

    total_chunks = 0

    for doc in documents:
        filename = doc["filename"]
        content = doc["content"]

        chunks = simple_chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)
            chunk_metadatas.append({"source": filename, "chunk_index": i})
            total_chunks += 1

    try:
        collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )
        print(f"Successfully ingested {total_chunks} chunks into ChromaDB!")
        return total_chunks
    except Exception as e:
        print(f"Failed to upsert chunks into ChromaDB: {e}")
        return 0


def query_rag(query_text: str, n_results: int = 3):
    """Query chromadb for nearest document chunks."""
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        retrieved_docs = []
        if results and "documents" in results and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if "metadatas" in results else []

            for i in range(len(docs)):
                meta = metadatas[i] if i < len(metadatas) else {}
                retrieved_docs.append({
                    "text": docs[i],
                    "metadata": meta
                })

        return retrieved_docs
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []


# simple script run test
if __name__ == "__main__":
    print("Testing RAG module locally...")
    ingest_documents("./docs")
    test_query = "What does week 15 focus on?"
    print(f"\nQuerying: '{test_query}'")
    matches = query_rag(test_query, n_results=2)
    for idx, match in enumerate(matches):
        print(f"\n--- Match {idx+1} (Source: {match['metadata'].get('source')}) ---")
        print(match["text"])
