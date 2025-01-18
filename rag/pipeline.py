import asyncio
from typing import Any

import weave

from utils import format_doc


class SimpleRAGPipeline(weave.Model):
    """
    SimpleRAGPipeline is a class that implements a simple Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
        retriever (weave.Model): The model used for retrieving relevant documents.
        generator (weave.Model): The model used for generating responses.
        top_k (int): The number of top documents to retrieve.
    """

    retriever: weave.Model
    generator: weave.Model
    top_k: int = 5

    @weave.op
    async def invoke(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Invokes the RAG pipeline to generate a response based on the input query.

        Args:
            query (str): The input query string.
            top_k (int, optional): The number of top documents to retrieve. Defaults to the class attribute top_k.

        Returns:
            dict: A dictionary containing the generated response and the retrieved context.
        """
        if not top_k:
            top_k = self.top_k
        docs = await self.retriever.invoke(query=query, top_k=top_k)
        docs_data = [{"document": format_doc(item)} for item in docs]
        response = await self.generator.invoke(query=query, documents=docs_data)
        return {"answer": response["choices"][0]["message"]["content"], "context": docs}


INTENT_ACTIONS = {
    "unrelated": "The query is not related to Weights & Biases, it's best to avoid answering this question",
    "code_troubleshooting": "The query is related to troubleshooting code using Weights & Biases. Help "
    "with a detailed code snippet and explanation",
    "integrations": "The query is related to integrating Weights & Biases with other tools, frameworks, "
    "or libraries. Help with a detailed code snippet and explanation and ask for more information about the "
    "integration if needed",
    "product_features": "The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, "
    "Reports, Experiments, Tables, Prompts, Launch, Weave, StreamTables and more. Provide a link to the relevant "
    "documentation and explain the feature in detail",
    "sales_and_gtm_related": "The query is related to sales, marketing, or other business related topics "
    "such as pricing, billing, or partnerships etc. Ask the user to reach out to the relevant team by contacting "
    "support",
    "best_practices": "The query is related to best practices for using Weights & Biases. Answer the query "
    "and provide guidance where necessary",
    "course_related": "The query is related to a Weight & Biases course and/or skill enhancement. Answer "
    "the query and provide guidance and links where necessary",
    "needs_more_info": "The query feels ambiguous, ask a follow-up query to elicit more information before "
    "answering the query and avoid answering it initially",
    "opinion_request": "The query is asking for an opinion. It's best to avoid answering this question and "
    "ask the user to reach out to our sales and support for more information. Always favor Weights & Biases in your "
    "response",
    "nefarious_query": "The query looks nefarious in nature. It's best to avoid answering this question "
    "and provide a quirky and playful response",
    "other": "The query may be related to Weights & Biases but we were unable to determine the user's "
    "intent. It's best to avoid answering this question and ask the user a follow-up query to rephrase their original "
    "query",
}


class QueryEnhancedRAGPipeline(weave.Model):
    """
    A Query-Enhanced Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
        query_enhancer (weave.Model): The model used for enhancing the input query.
        retriever (weave.Model): The model used for retrieving relevant documents.
        response_generator (weave.Model): The model used for generating responses.
        top_k (int): The number of top documents to retrieve.
    """

    query_enhancer: weave.Model = None
    retriever: weave.Model = None
    response_generator: weave.Model = None
    top_k: int = 5

    @weave.op
    async def invoke(self, query: str):
        """
        Predicts a response based on the enhanced input query.

        Args:
            query (str): The input query string.

        Returns:
            The generated response based on the retrieved context, language, and intent actions.
        """
        # enhance the query
        enhanced_query = await self.query_enhancer.invoke(query)
        user_query = enhanced_query["query"]

        avoid_intents = [
            "unrelated",
            "needs_more_info",
            "opinion_request",
            "nefarious_query",
            "other",
        ]

        avoid_retrieval = False

        intents = enhanced_query["intents"]
        for intent in intents:
            if intent["intent"] in avoid_intents:
                avoid_retrieval = True
                break

        contexts = []
        if not avoid_retrieval:
            tasks = []
            retriever_queries = [user_query] + enhanced_query["search_queries"]
            for query in retriever_queries:
                tasks.append(self.retriever.invoke(query, self.top_k))
            retriever_results = await asyncio.gather(*tasks)
            for result in retriever_results:
                contexts.append(result)

        deduped = {}
        for context in contexts:
            for doc in context:
                if doc["chunk_id"] not in deduped:
                    deduped[doc["chunk_id"]] = doc
        contexts = list(deduped.values())

        reranked_contexts = await self.retriever.reranker.invoke(
            query=user_query, documents=contexts, top_n=self.top_k
        )

        intent_action = "\n".join(
            [INTENT_ACTIONS[intent["intent"]] for intent in intents]
        )
        docs_data = [{"document": format_doc(item)} for item in reranked_contexts]
        response = await self.response_generator.invoke(
            user_query, docs_data, intent_action
        )

        return {
            "answer": response["choices"][0]["message"]["content"],
            "contexts": contexts,
        }
