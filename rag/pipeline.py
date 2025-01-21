import asyncio
from typing import Any

import weave

from utils import format_doc


class RAGPipeline(weave.Model):
    """
    A Retrieval-Augmented Generation (RAG) pipeline.

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


class SimpleRAGPipeline(RAGPipeline):
    pass


INTENT_ACTIONS = {
    "financial_performance": "The query is related to financial performance such as revenue, profit, margins, or overall financial health. Provide detailed analysis based on the available financial reports.",
    "operational_metrics": "The query is about specific business metrics, KPIs, or operational performance. Analyze the relevant metrics from the financial reports.",
    "market_analysis": "The query is related to market share, competition, or industry trends. Provide insights based on the company's disclosures and market information in the reports.",
    "risk_assessment": "The query is about potential risks, legal issues, or uncertainties facing the company. Analyze the risk factors and management's discussion in the reports.",
    "strategic_initiatives": "The query is about company strategy, new products/services, or future plans. Provide information based on management's strategic discussions in the reports.",
    "accounting_practices": "The query is about specific accounting methods, policies, or financial reporting practices. Explain the relevant accounting principles and their application.",
    "management_insights": "The query is related to management commentary, guidance, or leadership decisions. Analyze the management's discussion and analysis sections of the reports.",
    "capital_structure": "The query is about debt, equity, capital allocation, or financing activities. Provide analysis based on the balance sheet and cash flow statements.",
    "segment_analysis": "The query is about performance or metrics of specific business segments or divisions. Analyze the segment reporting in the financial statements.",
    "comparative_analysis": "The query is comparing current results to past periods or to other companies. Provide a comparative analysis using the available financial data.",
    "unrelated": "The query is not related to financial analysis or SEC filings. It's best to avoid answering this question and ask for a finance-related query.",
    "needs_more_info": "The query is ambiguous or lacks context. Ask a follow-up question to elicit more specific information about the financial analysis needed.",
    "opinion_request": "The query is asking for a subjective opinion rather than factual analysis. Clarify that as an AI, you provide objective analysis based on financial reports, not personal opinions.",
    "nefarious_query": "The query appears to have potentially unethical or malicious intent. Avoid answering and suggest focusing on legitimate financial analysis questions.",
    "other": "The query may be related to financial analysis, but its intent is unclear. Ask the user to rephrase their question, focusing specifically on aspects of financial reports or SEC filings."
}


class QueryEnhancedRAGPipeline(RAGPipeline):
    """
    A Query-Enhanced Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
        query_enhancer (weave.Model): The model used for enhancing the input query.
        retriever (weave.Model): The model used for retrieving relevant documents.
        response_generator (weave.Model): The model used for generating responses.
        top_k (int): The number of top documents to retrieve.
    """

    query_enhancer: weave.Model = None

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
        try:
            reranked_contexts = await self.retriever.reranker.invoke(
                query=user_query, documents=contexts, top_n=self.top_k
            )
        except Exception:
            reranked_contexts = contexts[: self.top_k]

        intent_action = "\n".join(
            [INTENT_ACTIONS[intent["intent"]] for intent in intents]
        )
        docs_data = [{"document": format_doc(item)} for item in reranked_contexts]
        response = await self.generator.invoke(user_query, docs_data, intent_action)

        return {
            "answer": response["choices"][0]["message"]["content"],
            "contexts": contexts,
        }
