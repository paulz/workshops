import json
import pathlib
import weave
import requests

def transform_finance_dataset(input_data):
    """
    Transform financial dataset to match expected evaluation format.
    
    Expected columns:
    - question
    - answer
    - source_docs
    - question_type
    - source_chunk_type
    - contexts: List[{
        "content": str,
        "source": str,
        "score": float,
        "relevance": int,
        "chunk_index": int
    }]
    """
    transformed = {
        "question": input_data["question"],
        "answer": input_data["answer"],
        "source_docs": input_data["doc_name"],  # Using primary document name
        "question_type": input_data.get("question_type", "Information Extraction"),
        "source_chunk_type": "Financial Statement",
        "contexts": []
    }
    
    # Transform evidence into contexts with required structure
    for idx, evidence in enumerate(input_data["evidence"]):
        context = {
            "content": evidence["evidence_text"],
            "source": evidence["doc_name"],
            "score": 1.0,  # Default score for ground truth evidence
            "relevance": 2,  # Mark as highly relevant since it's ground truth
            # "chunk_index": idx
        }
        transformed["contexts"].append(context)
        
    return transformed

@weave.op()
def create_evaluation_dataset(input_file: str) -> list:
    """Create evaluation dataset with specified column structure."""
    eval_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                transformed = transform_finance_dataset(sample)
                eval_data.append(transformed)
    
    return eval_data

# Example usage:
if __name__ == "__main__":
    # Download the dataset from GitHub
    weave.init("rag-workshop-pc-nyc-financebench")
    url = "https://raw.githubusercontent.com/patronus-ai/financebench/refs/heads/main/data/financebench_open_source.jsonl"
    response = requests.get(url)
    
    # Save the downloaded file locally
    data_dir = pathlib.Path("./data/eval")
    data_dir.mkdir(parents=True, exist_ok=True)
    input_file = str(data_dir / "financebench_open_source.jsonl")
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    # Process the downloaded file
    eval_dataset = create_evaluation_dataset(input_file)
    
    # Save as Weave dataset
    weave_dataset = weave.Dataset(name="financebench_eval_dataset", rows=eval_dataset)
    weave.publish(weave_dataset)