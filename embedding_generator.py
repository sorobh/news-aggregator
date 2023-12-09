# Import necessary libraries
# For example, using Sentence Transformers for generating embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    print(f"Input text length: {len(text)}")
    embedding = model.encode(text, convert_to_tensor=True)
    print(f"Generated embedding size: {embedding.size()}")
    #import ipdb; ipdb.set_trace()
    return embedding.cpu().numpy()  # Assuming you're using PyTorch
