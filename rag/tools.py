import json
import os

from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient


class SearchInternet(BaseModel):
    """Tool to search the internet for information"""

    search_query: str = Field(description="The query to search the internet for")

    async def run(self) -> list[dict[str, str]]:
        tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        context = await tavily_client.get_search_context(query=self.search_query)
        return [json.loads(item) for item in json.loads(json.loads(context))]
