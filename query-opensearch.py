from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
import torch
from pprint import pprint

INDEX_NAME = "recipes"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = OpenSearch(
    hosts=["https://admin:admin@localhost:9200/"],
    http_compress=True,
    use_ssl=True,
    verify_certs=False,  # DONT USE IN PRODUCTION
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

text = input("What you're looking for? ")
with torch.no_grad():
    mean_pooled = model.encode(text)

query = {
    "size": 10,
    "query": {"knn": {"embedding": {"vector": mean_pooled, "k": 2}}},
    "_source": False,
    "fields": ["id", "name", "description"],
}

response = client.search(body=query, index=INDEX_NAME)  # the same as before
pprint(response["hits"]["hits"])
