import sys
import subprocess
from pathlib import Path

from retrieval.index_paths import THE_TOPICS

def write_scores_to_file(all_qids, all_docids, scores, output_filename):
    output_filename = Path(output_filename)
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, 'w')  as f:
        rerank_scores = [{'qid': qid, 'docid': docid, 'score': score} for qid, docid, score in zip(all_qids, all_docids, scores)]
        reranked_scores_sorted = sorted(
                rerank_scores,
                key=lambda x: (x['qid'], -x['score'])
            )
        
        rank = 0
        prev_qid = None
        for document in reranked_scores_sorted:
            qid = document["qid"]
            if qid != prev_qid:
                rank = 1
                prev_qid = qid
            else:
                rank += 1
            f.write(f'{qid} Q0 {document["docid"]} {rank} {document["score"]} rank\n')

def evaluate(corpus_name, output_filename, qrels_path=None):
    if corpus_name in ['dl21', 'dl22', 'dl23']:
        qrels_name = f'{THE_TOPICS[corpus_name]}-passage'
    elif corpus_name in ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "pony", "leetcode", "aops", "theoremqa_theorems", "theoremqa_questions"]:
        if qrels_path is None:
            raise ValueError(f"qrels_path is required for BRIGHT corpus {corpus_name}")
        qrels_name = qrels_path
    else:
        qrels_name = THE_TOPICS[corpus_name]

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.eval.trec_eval",
            "-c",
            "-m",
            "ndcg_cut.10",
            str(qrels_name),
            str(output_filename),
        ],
        check=True,
    )
