from typing import Any

import weave
from litellm import acompletion


class SimpleResponseGenerator(weave.Model):
    model: str = "command-r"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = (
        "Answer the following question about Weights & Biases based on the provided product documentation. Note: not all documents may be relevant to the query."
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
        self, query: str, documents: list[dict[str, Any]], intents: str
    ) -> dict[str, Any]:
        response = await acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt.format(intents=intents),
                },
                {"role": "user", "content": query},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            documents=documents,
        )
        return response.model_dump(mode="json")
