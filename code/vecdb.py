import chromadb
from chromadb import PersistentClient

def _save_embeddings_to_db(vector_db_save_path, vector_db_name, input_embeddings, metadata, ids):
    client = PersistentClient(path=vector_db_save_path)
    if vector_db_name in client.list_collections():
        collection = client.get_collection(name=vector_db_name)
    else:
        collection = client.create_collection(name=vector_db_name)
    max_batch_size = 41660
    if(len(input_embeddings)>max_batch_size):
        for i in range(0, len(input_embeddings), max_batch_size):
            batch_embeddings = input_embeddings[i:i + max_batch_size]
            batch_metadata = metadata[i:i + max_batch_size]
            batch_ids = ids[i:i + max_batch_size]

            collection.add(embeddings=batch_embeddings, metadatas=batch_metadata, ids=[str(batch_id) for batch_id in batch_ids])
    else:
        collection.add(embeddings=input_embeddings, metadatas=metadata, ids=[str(id) for id in ids])