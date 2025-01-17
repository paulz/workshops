import json
import wandb

from utils import (
    chunk_markdown,
    chunk_source_code,
    chunk_notebook,
)

import pathlib
from tqdm import tqdm

wandb_api = wandb.Api()
docs_root = wandb_api.artifact(
    "parambharat/rag-workshop/documentation:latest", type="dataset"
).download(root="data/docs")
docs_file = f"{docs_root}/wandb_docs.jsonl"
with open(docs_file, "r") as f:
    docs = f.readlines()
docs = [json.loads(doc) for doc in docs]


def load_dataset(docs_root):
    files = pathlib.Path(docs_root).rglob("*.jsonl")
    docs = []
    for file in files:
        for line in file.read_text().splitlines():
            docs.append(json.loads(line))
    return docs


ds = load_dataset(docs_root)


def chunk_dataset(ds, chunk_size=500):
    all_chunks = []
    for doc in tqdm(ds):
        if doc["metadata"]["file_type"] == "python":
            chunks = chunk_source_code(doc["content"], chunk_size=chunk_size)
        elif doc["metadata"]["file_type"] == "notebook":
            print("chunking notebook")
            chunks = chunk_notebook(doc["content"], chunk_size=chunk_size)
        else:
            chunks = chunk_markdown(doc["content"], chunk_size=chunk_size)
        all_chunks.extend(chunks)
    return all_chunks


chunked_ds = chunk_dataset(ds[880:890])
