from importlib.util import find_spec
from typing import Any, Optional

import weave
from pydantic import Field
from transformers import pipeline
from weave import Scorer


def set_device(device: str = "auto") -> "device":
    """Set the device to use for the model.

    Args:
        device: The device to use for the model.

    Returns:
        The device to use for the model.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    if not cuda_available and "cuda" in device:
        # could be `cuda:0`, `cuda:1`, etc.
        raise ValueError("CUDA is not available")
    if device == "auto":
        if cuda_available:
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


class HuggingFacePipelineScorer(Scorer):
    """
    Base class for using Hugging Face pipelines for moderation scoring.

    This class simplifies the use of Hugging Face pipelines by handling the initialization and providing a common interface for scoring.

    Args:
        task (str): The pipeline task type (e.g., `"text-classification"`).
        model_name_or_path (str): The name or path of the model to use.
        device (str): The device to use for inference. Defaults to `"cpu"`.
        pipeline_kwargs (dict[str, Any]): Additional keyword arguments for the pipeline. Defaults to `{}`.

    Returns:
        list[dict[str, Any]]: The pipeline's output after processing the input text.

    """

    task: str = Field(
        description="The task to use for the pipeline, for example 'text-classification'"
    )
    model_name_or_path: str = Field(default="", description="The path to the model")
    device: str = Field(default="auto", description="The device to use for the model")
    pipeline_kwargs: dict[str, Any] = Field(default_factory=dict)
    pipeline: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        self.device = set_device(self.device)
        try:
            if find_spec("transformers") is None:
                print(
                    "The `transformers` package is required to use PipelineScorer, please run `pip install transformers`"
                )
        except ImportError:
            print(
                "The `transformers` package is required to use PipelineScorer, please run `pip install transformers`"
            )
        if self.pipeline is None:
            self.set_pipeline()

    def load_pipeline(self) -> None:
        raise NotImplementedError(
            "Subclasses must implement the `load_pipeline` method."
        )

    @weave.op
    def score(self, *, output: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ClassificationResponseScorer(HuggingFacePipelineScorer):
    task: str = "nli-scorer"
    model_name_or_path: str = "param-bharat/ModernBERT-base-nli-scorer"
    model_max_length: int = 2048
    base_url: Optional[str] = None
    pipeline_kwargs: dict[str, Any]

    def load_pipeline(self) -> None:
        self.pipeline = pipeline(
            self.task,
            model=self.model_name_or_path,
            device=self.device,
            trust_remote_code=True,
            batch_size=4,
        )

    def set_pipeline(self) -> None:
        self.load_pipeline()

    @weave.op
    def score(
        self,
        input: str | None = None,
        output: str | None = None,
        context: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        pipeline_inputs = {
            "Prompt": input,
            "Completion": output,
            "Context": context,
            "ChatHistory": chat_history,
        }
        pipeline_outputs = self.pipeline(inputs=pipeline_inputs, **self.pipeline_kwargs)
        return pipeline_outputs


class ResponseCorrectnessScorer(ClassificationResponseScorer):
    pipeline_kwargs: dict[str, Any] = {
        "task_type": "Quality/Response/Correctness",
        "threshold": 0.8,
    }

    @weave.op
    def score(
        self,
        input: str | None = None,
        output: str | None = None,
        context: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        pipeline_outputs = super().score(
            input=input, output=output, context=context, chat_history=chat_history
        )
        result = {
            "correct": pipeline_outputs["label"] == 1,
            "extras": pipeline_outputs,
        }
        return result


class ResponseHelpfulnessScorer(ClassificationResponseScorer):
    pipeline_kwargs: dict[str, Any] = {
        "task_type": "Quality/Response/Helpfulness",
        "threshold": 0.8,
    }

    @weave.op
    def score(
        self,
        input: str | None = None,
        output: str | None = None,
        context: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        pipeline_outputs = super().score(
            input=input, output=output, context=context, chat_history=chat_history
        )
        result = {
            "helpful": pipeline_outputs["label"] == 1,
            "extras": pipeline_outputs,
        }
        return result


class ResponseRelevanceScorer(ClassificationResponseScorer):
    pipeline_kwargs: dict[str, Any] = {
        "task_type": "Quality/Response/Relevance",
        "threshold": 0.8,
    }

    @weave.op
    def score(
        self,
        input: str | None = None,
        output: str | None = None,
        context: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        pipeline_outputs = super().score(
            input=input, output=output, context=context, chat_history=chat_history
        )
        result = {
            "relevant": pipeline_outputs["label"] == 1,
            "extras": pipeline_outputs,
        }
        return result


class DocumentRelevanceScorer(ClassificationResponseScorer):
    pipeline_kwargs: dict[str, Any] = {
        "task_type": "Quality/Context/Document Relevance",
        "threshold": 0.8,
    }

    @weave.op
    def score(
        self,
        input: str | None = None,
        output: list[str] | None = None,
        context: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        relevance_outputs = []
        relevance_scores = []
        for response in output:
            pipeline_outputs = super().score(
                input=input, output=response, context=context, chat_history=chat_history
            )
            relevance_outputs.append(pipeline_outputs["label"])
            relevance_scores.append(
                pipeline_outputs["score"]
                if pipeline_outputs["label"] == 1
                else 1 - pipeline_outputs["score"]
            )
        result = {
            "relevance": round(sum(relevance_outputs) / len(relevance_outputs), 4),
            "relevance_score": round(sum(relevance_scores) / len(relevance_scores), 4),
        }

        return result
