"""
Code adapted from the Rank-R1 repository: https://github.com/ielab/llm-rankers/tree/main/Rank-R1
"""

from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

bright_queries = load_dataset("xlangai/BRIGHT", 'examples')
splits = list(bright_queries.keys())

output_directory = Path(__file__).resolve().parent / "pyserini_qrels"
output_directory.mkdir(parents=True, exist_ok=True)
for split in splits:
    queries = bright_queries[split]
    print(split)
    with (output_directory / f"{split}.tsv").open('w', encoding='utf-8') as f:
        for query in tqdm(queries):
            query_id = query['id']
            positive_doc_ids = query['gold_ids']
            for doc_id in set(positive_doc_ids):
                doc_id = doc_id.replace(' ', '_')
                f.write(f'{query_id}\t0\t{doc_id}\t1\n')
