from opensearchpy import OpenSearch, helpers

INDEX_NAME = "recipes"

client = OpenSearch(
    hosts=["https://admin:admin@localhost:9200/"],
    http_compress=True,
    use_ssl=True,  # DONT USE IN PRODUCTION
    verify_certs=False,  # DONT USE IN PRODUCTION
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# Create indicies
settings = {
    "settings": {
        "index": {
            "knn": True,
        }
    },
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "id": {"type": "integer"},
            "description": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,
            },
        }
    },
}

res = client.indices.create(index=INDEX_NAME, body=settings, ignore=[400])
print(res)

