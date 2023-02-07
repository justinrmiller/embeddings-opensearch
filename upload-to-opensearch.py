import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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

FILENAME = "RAW_recipes.zip"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device="mps"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Upload dataset to Openserach
chunksize = 300
lines = 267782  # wc -l RAW_recipes.csv
reader = pd.read_csv(
    FILENAME, chunksize=chunksize, usecols=["name", "id", "description"]
)

with tqdm(total=lines) as pbar:
    for i, chunk in enumerate(reader):
        print(i)
        # Remove NaN
        chunk.fillna("", inplace=True)
        docs = chunk.to_dict(orient="records")

        # Embed description
        with torch.no_grad():
            mean_pooled = model.encode([doc["description"] for doc in docs])
        for doc, vec in zip(docs, mean_pooled):
            doc["embedding"] = vec

        # Upload documents
        helpers.bulk(client, docs, index=INDEX_NAME, raise_on_error=True, refresh=True)

        # Clear CUDA cache
        del mean_pooled 
        torch.cuda.empty_cache()

        pbar.update(chunksize)
