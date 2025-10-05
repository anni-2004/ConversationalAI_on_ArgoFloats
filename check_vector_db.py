import chromadb
import os
import sys
from config import VECTOR_DB_CONFIG

print(f"Current working directory: {os.getcwd()}")
# Ensure the script can locate the database
print(f"Connecting to ChromaDB at: {os.path.abspath(VECTOR_DB_CONFIG['path'])}")

try:
    # 1. Connect to the persistent client
    client = chromadb.PersistentClient(path=VECTOR_DB_CONFIG["path"])
    collection_name = VECTOR_DB_CONFIG["collection_name"]
    
    # 2. Get the collection. If it doesn't exist, this will raise an error.
    collection = client.get_collection(name=collection_name)
    
    # 3. Retrieve all documents, their metadata, and embeddings.
    #    The 'get' method is the correct way to do this.
    documents = collection.get(
        include=['metadatas', 'documents', 'embeddings']
    )

    if not documents['documents']:
        print("Error: Vector store is empty. No documents found.")
        sys.exit()
    
    # 4. Process and print the retrieved data
    print(f"Found {len(documents['documents'])} documents:")
    print("-----------------------------------")
    
    for i, doc in enumerate(documents['documents']):
        metadata = documents['metadatas'][i]
        embedding = documents['embeddings'][i]
        
        # Format the output clearly
        print(f"Document {i+1}:")
        print(f"  Float ID: {metadata.get('float_id', 'N/A')}")
        print(f"  Summary: {doc}")
        print(f"  Embedding (first 50 values): {embedding[:50]}...")
        print(f"  Embedding Length: {len(embedding)}\n")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your data ingestion script (nc_to_postgreys.py) has been run successfully.")