import pinecone
import numpy as np
import torch
from apikey import (apikey, pinecone_env, pinecone_key)

def connect_to_pinecone(api_key, environment):
    #Global variables for PINECONE
    environment = pinecone_env
    api_key = pinecone_key
    pinecone.init(api_key=api_key, environment=environment)

def disconnect_from_pinecone():
    # Pinecone doesn't require a disconnect function in the same way as Milvus
    pass



#Check connection
def check_connection():
    try:
        indexes = pinecone.list_indexes()
        print("Connection Successful. Found indexes:", indexes)
        return True
    except Exception as e:
        print("Connection Failed. Error:", e)
        return False



#Create and connect to Pinecone Index
def create_or_connect_index(index_name, dimension):
    environment = pinecone_env
    api_key = pinecone_key
    pinecone.init(api_key=api_key, environment=environment) #important?
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension, metric="cosine")  # or "euclidean"
    #import ipdb; ipdb.set_trace()
    return pinecone.Index(index_name)


#convert string to numeric embeddings
def convert_string_embeddings_to_numeric_embeddings(string_embeddings):
    numeric_embeddings = []
    for string_embedding in string_embeddings:
        numeric_embeddings.append(np.float64(string_embedding))
    return numeric_embeddings


 

#Inserting embeddings- Some thing else in the data cleaning is creating issue here
def insert_embeddings(index, ids, embeddings):
    vectors = []
    for id, embedding in zip(ids, embeddings):
        str_id = str(id)
        if isinstance(embedding, torch.Tensor):
            embedding_list = embedding.cpu().detach().numpy().tolist()
        elif isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        elif isinstance(embedding, list):
            embedding_list = embedding
        else:
            raise TypeError(f"Embedding must be a list, NumPy array, or PyTorch tensor, got {type(embedding)}")

        vector_data = {
            'id': str_id,
            'values': embedding_list
        }

        vectors.append(vector_data)
    #import ipdb; ipdb.set_trace()
    index.upsert(vectors=vectors)



#Searching
def retrieve_documents(index, query_embedding, top_k=10):
    # Convert query_embedding to list if it's an ndarray
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    try:
        response = index.query(queries=[query_embedding], top_k=top_k)
        # Extracting IDs from the response
        retrieved_ids = [match['id'] for match in response['results'][0]['matches']]
        return retrieved_ids
    except Exception as e:
        # Handle or log the exception
        print(f"Error in retrieving documents: {e}")
        return []


