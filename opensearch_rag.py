"""
OpenSearch-based Retrieval Augmented Generation (RAG) implementation.
Uses sentence embeddings to find relevant context from past support tickets.
"""

from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Configuration constants
INDEX_NAME = "ticket-history"
EMBEDDING_DIM = 384  # Dimension for MiniLM embeddings

# Initialize sentence transformer model for generating embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def connect_opensearch():
    """
    Establishes connection to OpenSearch instance.
    Returns:
        OpenSearch client instance
    """
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],  # Change if running remotely
        http_auth=("admin", "Cradle2Grave@"),         # Use credentials if configured
        use_ssl=False,
        verify_certs=False
    )

def embed(text):
    """
    Generates embedding vector for input text using the sentence transformer model.
    
    Args:
        text (str): Input text to embed
        
    Returns:
        list: Embedding vector as a list of floats
    """
    return model.encode(text).tolist()

def create_index(client):
    """
    Creates OpenSearch index with k-NN vector search capabilities if it doesn't exist.
    
    Args:
        client: OpenSearch client instance
    """
    if client.indices.exists(INDEX_NAME):
        return
        
    mapping = {
        "settings": {
            "index": {
                "knn": True  # enables k-NN search
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",  # correct type for OpenSearch
                    "dimension": EMBEDDING_DIM
                }
            }
        }
    }
    client.indices.create(index=INDEX_NAME, body=mapping)

def ingest_tickets(file_path="ticket_history.json"):
    """
    Ingests support tickets from JSON file into OpenSearch index.
    Generates embeddings for each ticket and stores them with the original text.
    
    Args:
        file_path (str): Path to JSON file containing ticket history
    """
    client = connect_opensearch()
    create_index(client)
    with open(file_path) as f:
        data = json.load(f)

    # Prepare bulk ingestion actions
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": ticket["id"],
            "_source": {
                "text": ticket["text"],
                "embedding": embed(ticket["text"])
            }
        }
        for ticket in data
    ]
    helpers.bulk(client, actions)

def retrieve_context(message, top_k=3):
    """
    Retrieves most relevant context from past tickets using vector similarity search.
    
    Args:
        message (str): Query message to find relevant context for
        top_k (int): Number of most relevant tickets to retrieve
        
    Returns:
        list: List of relevant ticket texts
    """
    client = connect_opensearch()
    query_vector = embed(message)

    # Construct k-NN search query
    query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }
    response = client.search(index=INDEX_NAME, body=query)
    return [hit["_source"]["text"] for hit in response["hits"]["hits"]]
