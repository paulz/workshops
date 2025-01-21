# From Simple to Agentic RAG Workshop

This workshop demonstrates how to build and evolve a Retrieval-Augmented Generation (RAG) system from a simple implementation to a more sophisticated agentic system. The workshop uses the Weights & Biases documentation as a case study.

## Workshop Overview

The notebook covers:

1. Building a basic RAG pipeline
2. Evaluating RAG components
3. Iteratively improving the pipeline
4. Implementing an agentic RAG system

## Key Components Covered

### Simple RAG
- Basic document chunking
- TF-IDF based retrieval
- Simple response generation

### Improved Retrieval
- BM25 retrieval
- Dense retrieval using embeddings
- Hybrid retrieval combining multiple approaches
- Reranking for better precision

### Advanced Techniques
- Structured chunking for better context preservation
- Vector store integration for scalability
- Query enhancement with intent classification
- Tool-augmented RAG with internet search capabilities

## Setup Requirements

1. OpenAI API key
2. Tavily API key (sign up at [Tavily](https://app.tavily.com/home))
3. Python environment with required packages (installed via `uv`)
4. W&B API key (sign up at [W&B](https://app.wandb.ai/login?signup=true))
## Installation

```bash
!git clone https://github.com/parambharar/workshops.git
!cd workshops/rag
!pip install uv
!uv sync
```

## Environment Setup

Create an `.env` file with:
```
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
WANDB_PROJECT="rag-workshop-iiit-blr"
```

## Key Features

- Comprehensive evaluation metrics for retrieval and response generation
- Integration with W&B Weave for experiment tracking
- Structured approach to improving RAG components
- Implementation of advanced features like query enhancement and tool use

## Resources Used

- W&B Documentation as source data
- Cohere for embeddings and reranking
- Tavily for internet search
- W&B Weave for experiment tracking

## Workshop Flow

1. Start with a simple RAG implementation
2. Add evaluation metrics
3. Improve retrieval with different algorithms
4. Implement structured chunking
5. Add query enhancement
6. Build an agentic system with tool use

This workshop is designed to provide a hands-on understanding of RAG systems and their evolution from simple to more sophisticated implementations.