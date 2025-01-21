import asyncio
import os
from functools import partial
from threading import Lock
from typing import Any

import Stemmer
import pinecone
from pinecone.grpc import PineconeGRPC as Pinecone

import bm25s
import litellm
import numpy as np
import Stemmer
import weave
from litellm.caching.caching import Cache
from pydantic import PrivateAttr
from rerankers import Reranker as AnsReranker
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

litellm.cache = Cache(disk_cache_dir="data/cache")


class TfidfSearchEngine:
    """
    A retriever model that uses TF-IDF for indexing and searching documents.

    Attributes:
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer.
        index (list): The indexed data.
        data (list): The data to be indexed.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._index = None
        self._data = None

    async def fit(self, data):
        """
        Indexes the provided data using TF-IDF.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        # store the original data as is so we can use it for retrieval later
        self._data = data
        # just extract the text from the documents
        docs = [doc["text"] for doc in self._data]
        # our instance is simply an instance of TfidfVectorizer from scikit-learn
        # we use the fit_transform method to vectorize the documents
        self._index = self._vectorizer.fit_transform(docs)
        return self

    async def search(
        self, query: str, top_k: int = 5, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Searches the indexed data for the given query using cosine similarity.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        assert self._index is not None, "Index is not set"
        assert self._data is not None, "Data is not set"
        # vectorize the query
        query_vec = self._vectorizer.transform([query])
        # compute the cosine distance between the query vector and the indexed vectors
        cosine_distances = cdist(
            query_vec.todense(), self._index.todense(), metric="cosine"
        )[0]
        # get the top-k indices of the indexed vectors that are most similar to the query vector
        top_k_indices = cosine_distances.argsort()[:top_k]
        # create the output list of dictionaries
        output = []
        # iterate over the top-k indices and append the corresponding document to the output list
        for idx in top_k_indices:
            output.append(
                {"score": round(float(1 - cosine_distances[idx]), 4), **self._data[idx]}
            )
        # return the output list of dictionaries
        return output


class BM25SearchEngine:
    """
    A retriever model that uses BM25 for indexing and searching documents.

    Attributes:
        index (bm25s.BM25): The BM25 index.
        _stemmer (Stemmer.Stemmer): The stemmer.
        data (list): The data to be indexed.
    """

    def __init__(self):
        self._index = bm25s.BM25()
        self._stemmer = Stemmer.Stemmer("english")
        self._data = None

    async def fit(self, data):
        """
        Indexes the provided data using BM25.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        # store the original data as is so we can use it for retrieval later
        self._data = data
        # just extract the text from the documents
        corpus = [doc["text"] for doc in data]
        # our instance is simply an instance of bm25s.BM25 from the bm25s library
        # we use the tokenize method to tokenize the documents
        # we use the index method to index the documents
        corpus_tokens = bm25s.tokenize(corpus, show_progress=False, stopwords="english")
        # index the documents and store the corpus tokens in the index
        self._index.index(corpus_tokens, show_progress=False)
        return self

    async def search(self, query, top_k=5, **kwargs):
        """
        Searches the indexed data for the given query using BM25.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        assert self._index is not None, "Index is not set"
        assert self._data is not None, "Data is not set"
        # tokenize the query
        query_tokens = bm25s.tokenize(query, show_progress=False, stopwords="english")
        # get the top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self._index.retrieve(
            query_tokens, corpus=self._data, k=top_k, show_progress=False
        )

        output = []
        for idx in range(results.shape[1]):
            output.append(
                {
                    "score": round(float(scores[0, idx]), 4),
                    **results[0, idx],
                }
            )
        return output


async def batch_embed(pc, docs, input_type="search_document", batch_size=50):
    all_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        embeddings = await pc.inference.embed(
            model="multilingual-e5-large",
            inputs=batch,
            parameters={"input_type": input_type, "truncate": "END"}
        )
        all_embeddings.append(embeddings)
    return np.array(all_embeddings)


class DenseSearchEngine:
    """
    A retriever model that uses dense embeddings for indexing and searching documents.

    Attributes:
        vectorizer (Callable): The function used to generate embeddings.
        index (np.ndarray): The indexed embeddings.
        data (list): The data to be indexed.
    """

    def __init__(self, model="multilingual-e5-large"):
        self._pc = Pinecone()
        self._model = model
        self._index = None
        self._data = None

    async def fit(self, data):
        """
        Indexes the provided data using dense embeddings.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        self._data = data
        docs = [doc["text"] for doc in data]
        embeddings = await batch_embed(
            self._pc, docs, input_type="search_document"
        )
        self._index = np.array(embeddings)
        return self

    async def search(self, query, top_k=5, **kwargs):
        """
        Searches the indexed data for the given query using cosine similarity.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        query_embedding = await batch_embed(
            self._pc,
            [query],
            input_type="search_query",
        )
        cosine_distances = cdist(query_embedding, self._index, metric="cosine")[0]
        top_k_indices = cosine_distances.argsort()[:top_k]
        output = []
        for idx in top_k_indices:
            output.append(
                {
                    "score": round(float(1 - cosine_distances[idx]), 4),
                    **self._data[idx],
                }
            )
        return output


def make_batches_for_db(
    vectorizer, data, batch_size=50, is_cohere_model=False, dimensions=512
):
    for i in range(0, len(data), batch_size):
        batch_docs = data[i : i + batch_size]
        batch_texts = [doc["text"] for doc in batch_docs]
        if is_cohere_model:
            embeddings = asyncio.run(
                batch_embed(vectorizer, batch_texts, input_type="search_document")
            )
        else:
            embeddings = asyncio.run(
                batch_embed(vectorizer, batch_texts, dimensions=dimensions)
            )
        yield [
            {"vector": embedding, **doc}
            for embedding, doc in zip(embeddings, batch_docs)
        ]


class VectorStoreSearchEngine:
    def __init__(
        self,
        index_name="docs",
        embedding_model="multilingual-e5-large",
        environment="gcp-starter"
    ):
        self._pc = Pinecone()
        self._model = embedding_model
        self._index_name = index_name
        self._environment = environment
        self._index = None

    async def fit(self, data):
        pinecone.init(environment=self._environment)
        
        if self._index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self._index_name,
                dimension=384,
                metric="cosine"
            )
        
        self._index = pinecone.Index(self._index_name)
        
        batch_size = 50
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            embeddings = await batch_embed(self._pc, texts, input_type="search_document")
            
            vectors = [
                (str(i + idx), embedding.tolist(), {k: v for k, v in doc.items() if k != "vector"})
                for idx, (embedding, doc) in enumerate(zip(embeddings, batch))
            ]
            self._index.upsert(vectors=vectors)
            
        return self

    async def load(self):
        pinecone.init(environment=self._environment)
        self._index = pinecone.Index(self._index_name)
        return self

    async def search(self, query, top_k=5, filters=None):
        query_embedding = await batch_embed(
            self._pc,
            [query],
            input_type="search_query",
        )
        
        query_results = self._index.query(
            vector=query_embedding[0].tolist(),
            top_k=top_k,
            filter=filters,
            include_metadata=True
        )
        
        results = []
        for match in query_results.matches:
            result = match.metadata
            result["score"] = round(float(match.score), 4)
            results.append(result)
            
        return results


class Retriever(weave.Model):
    search_engine: Any

    @weave.op
    async def invoke(self, query: str, top_k: int, **kwargs) -> list[dict[str, Any]]:
        return await self.search_engine.search(query, top_k, **kwargs)


class Reranker(weave.Model):
    model: str = "bge-reranker-v2-m3"

    @weave.op
    async def invoke(self, query: str, documents: list[dict[str, Any]], top_n: int = 5):
        pc = Pinecone()
        texts = [doc["text"] for doc in documents]
        
        reranked = await pc.inference.rerank(
            model=self.model,
            query=query,
            documents=[{"text": text} for text in texts],
            top_n=top_n,
            return_documents=True,
            parameters={"truncate": "END"}
        )
        
        output_docs = []
        for result in reranked:
            doc = documents[result.index]
            doc["score"] = round(float(result.score), 4)
            output_docs.append(doc)
        return output_docs


class RetrieverWithReranker(Retriever):
    """
    A retriever model that uses dense embeddings for retrieval and a reranker for re-ranking the results.

    Attributes:
        retriever (DenseRetriever): The dense retriever model.
        reranker (Reranker): The reranker model.
    """

    search_engine: Any
    reranker: Any = Reranker()

    @weave.op
    async def invoke(self, query: str, top_k: int = None, top_n: int = None, **kwargs):
        """
        Predicts the top-n results for the given query after re-ranking.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top results to retrieve before re-ranking. Default is None.
            top_n (int, optional): The number of top results to return after re-ranking. Default is None.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-n results.
        """
        if top_k and not top_n:
            top_n = top_k
            top_k = top_k * 2
        elif top_n and not top_k:
            top_k = top_n * 2
        elif top_k == top_n:
            top_k = top_k * 2
        else:
            top_k = 10
            top_n = 5
        retrievals = await self.search_engine.search(query, top_k, **kwargs)
        reranked = await self.reranker.invoke(query, retrievals, top_n)
        return reranked


class HybridRetrieverWithReranker(weave.Model):
    sparse_search_engine: Any
    dense_search_engine: Any
    reranker: Any = Reranker()

    @weave.op
    async def invoke(self, query: str, top_k: int = None, top_n: int = None, **kwargs):
        if top_k and not top_n:
            top_n = top_k
            top_k = top_k * 2
        elif top_n and not top_k:
            top_k = top_n * 2
        elif top_k == top_n:
            top_k = top_k * 2
        else:
            top_k = 10
            top_n = 5

        sparse_retrievals = await self.sparse_search_engine.search(
            query, top_k, **kwargs
        )
        dense_retrievals = await self.dense_search_engine.search(query, top_k, **kwargs)
        retrievals = sparse_retrievals + dense_retrievals
        deduped_retrievals = {}
        for doc in retrievals:
            deduped_retrievals[doc["chunk_id"]] = doc
        deduped_retrievals = list(deduped_retrievals.values())
        reranked = await self.reranker.invoke(query, deduped_retrievals, top_n)
        return reranked
