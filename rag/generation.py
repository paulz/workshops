from typing import Any

import weave
from litellm import acompletion


class SimpleResponseGenerator(weave.Model):
    """
    SimpleResponseGenerator is a class that generates responses based on a query and provided documents.

    Attributes:
        model (str): The model to use for generating responses.
        temperature (float): The temperature setting for the model.
        max_tokens (int): The maximum number of tokens for the response.
        system_prompt (str): The system prompt to use for the model.
    """

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
        """
        Invokes the model to generate a response based on the query and documents.

        Args:
            query (str): The user query to process.
            documents (list[dict[str, Any]]): The list of documents to use for generating the response.

        Returns:
            dict: The response from the model in JSON format.
        """
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
    """
    QueryEnhancedResponseGenerator is a class that generates responses based on a query, provided documents, and specified intents.

    Attributes:
        model (str): The model to use for generating responses.
        temperature (float): The temperature setting for the model.
        max_tokens (int): The maximum number of tokens for the response.
        system_prompt (str): The system prompt to use for the model.
    """

    model: str = "command-r"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = open("prompts/qe_system_prompt.txt", "r").read()

    @weave.op
    async def invoke(
        self, query: str, documents: list[dict[str, Any]], intents: str
    ) -> dict[str, Any]:
        """
        Invokes the model to generate a response based on the query, documents, and intents.

        Args:
            query (str): The user query to process.
            documents (list[dict[str, Any]]): The list of documents to use for generating the response.
            intents (str): The intents to format into the system prompt.

        Returns:
            dict: The response from the model in JSON format.
        """
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
