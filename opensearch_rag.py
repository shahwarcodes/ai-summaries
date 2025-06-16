from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np
import json

INDEX_NAME = "ticket-history"
EMBEDDING_DIM = 384  # for MiniLM

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def connect_opensearch():
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],  # Change if running remotely
        http_auth=("admin", "Cradle2Grave@"),                 # Use credentials if configured
        use_ssl=False,
        verify_certs=False
    )

def embed(text):
    return model.encode(text).tolist()

def create_index(client):
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
    client = connect_opensearch()
    create_index(client)
    with open(file_path) as f:
        data = json.load(f)

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
    client = connect_opensearch()
    query_vector = embed(message)

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
