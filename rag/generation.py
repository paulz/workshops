from typing import Any

import weave
from litellm import acompletion


class SimpleResponseGenerator(weave.Model):
    model: str = "command-r"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = (
        "You are an AI assistant specializing in financial analysis and SEC filings interpretation. "
        "Answer questions about company financial reports, earnings calls, and regulatory filings "
        "based on the provided documents. Provide accurate, factual information and cite specific "
        "sources when referencing financial data. Note: not all documents may be relevant to the query."
    )

    @weave.op
    async def invoke(
        self, query: str, documents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            documents=documents,
        )
        return response.model_dump(mode="json")


class QueryEnhancedResponseGenerator(weave.Model):
    model: str = "command-r"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = open("prompts/qe_system_prompt.txt", "r").read()

    @weave.op
    async def invoke(
        self, query: str, documents: list[dict[str, Any]], intents: str, language: str = "English"
    ) -> dict[str, Any]:
        response = await acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt.format(intents=intents, language=language),
                },
                {"role": "user", "content": query},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            documents=documents,
        )
        return response.model_dump(mode="json")
