import json
from typing import Any

import weave
from litellm import acompletion
from litellm.types.utils import ModelResponse
from tools import TOOL_SCHEMAS, FUNCTION_MAP


class WandbAgent(weave.Model):
    system_prompt: str = open("prompts/agent.txt", "r").read()
    model: str = "command-r-08-2024"
    temperature: float = 0.1
    tools: list[dict[str, Any]] = TOOL_SCHEMAS

    @weave.op
    async def call_llm(self, messages=None, tools=None) -> ModelResponse:
        response = await acompletion(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=tools,
        )
        return response

    @weave.op
    async def run_tool_calls(self, messages=None, tool_calls=None):
        for idx, tc in enumerate(tool_calls):
            tool_result = await FUNCTION_MAP[tc.function.name](
                **json.loads(tc.function.arguments)
            ).run()
            tool_result = json.dumps(tool_result)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
            )
            return messages

    @weave.op
    async def run_agent(self, query, messages=None):
        if messages is None:
            messages = []

        if "system" not in {m.get("role") for m in messages}:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": query})

        response = await self.call_llm(messages=messages, tools=self.tools)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append({"role": "assistant", "tool_calls": tool_calls})
            messages = await self.run_tool_calls(messages, tool_calls)

            response = await self.call_llm(messages=messages, tools=self.tools)
            response_message = response.choices[0].message

        messages.append({"role": "assistant", "content": response_message.content})

        return messages

    @weave.op
    async def invoke(self, query: str) -> dict[str, str]:
        results = await self.run_agent(query)
        return {"answer": results[-1]["content"]}
