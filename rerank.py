import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher

from evaluation.helpers import evaluate, write_scores_to_file
from rerankers.prompts import get_prompt
from rerankers.rank1 import Rank1
from rerankers.rank1_no_reason import Rank1NoReason
from rerankers.rank1_ours import Rank1OurImplementation
from rerankers.rank1_self_consistency import Rank1SelfConsistency
from rerankers.rank0 import Rank0
from retrieval.bm25 import BM25
from retrieval.index_paths import THE_SPARSE_INDEX, load_queries_qids

BRIGHT_CORPUS = {'biology', 
                'earth_science', 
                'economics', 
                'psychology',
                'robotics',
                'stackoverflow',
                'sustainable_living',
                'pony',
                'leetcode',
                'aops',
                'theoremqa_theorems',
                'theoremqa_questions'}

MODEL_NAMES = (
    "rank0",
    "rank1",
    "rank1_ours",
    "rank1_no_reason",
    "rank1_self_consistency",
)

LORA_MODEL_NAMES = {
    "rank0",
    "rank1_ours",
    "rank1_self_consistency",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Rerankers')
    parser.add_argument('--model_path', required=True, help='base model path')
    parser.add_argument('--model_name', required=True, choices=MODEL_NAMES, help='reranker variant')
    parser.add_argument('--lora_module', default=None, help='lora path')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--corpus_name', required=True, help='dataset path')
    parser.add_argument('--bright_run_file', required=False, help='first stage retrieval run file for BRIGHT dataset')
    parser.add_argument('--qrels_path', required=False, default=None, help='Custom Qrels file for non-Pyserini corpora')
    parser.add_argument('--k', type=int, default=100, help='search results for final retrieval')
    parser.add_argument('--output_filename', required=True)
    args = parser.parse_args()
    if args.model_name in LORA_MODEL_NAMES and args.lora_module is None:
        parser.error(f"--lora_module is required for {args.model_name}")
    if args.corpus_name in BRIGHT_CORPUS:
        if args.bright_run_file is None:
            parser.error("--bright_run_file is required for BRIGHT corpora")
        if args.qrels_path is None:
            parser.error("--qrels_path is required for BRIGHT corpora")

    output_parent = Path(args.output_filename).expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if args.corpus_name not in BRIGHT_CORPUS:
        qids, queries = load_queries_qids(args.corpus_name)
        searcher = LuceneSearcher.from_prebuilt_index(THE_SPARSE_INDEX[args.corpus_name])
        if searcher == None:
            searcher = LuceneSearcher(THE_SPARSE_INDEX[args.corpus_name])
        bm25 = BM25(searcher=searcher, task=args.corpus_name)
        bm25_outputs = bm25.run_search(qids, queries, k=args.k, return_passage_texts=True)
        bm25_output_filename = os.path.join(os.path.dirname(args.output_filename), f'bm25_{args.corpus_name}') 
    else:
        bright_corpus = load_dataset("xlangai/BRIGHT", 'documents')
        # Use the original query for reranking, following the Rank1 setup.
        bright_queries = load_dataset("xlangai/BRIGHT", 'examples')
        corpus = bright_corpus[args.corpus_name]
        queries = bright_queries[args.corpus_name]

        run_file = Path(args.bright_run_file)
        bm25_outputs = {}
        if run_file.suffix.lower() == '.json':
            with run_file.open('r', encoding='utf-8') as file:
                bm25_run = json.load(file)
            bm25_run = {
                        query_id: dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:args.k])
                        for query_id, doc_scores in bm25_run.items()
                        }

        else:
            bm25_run = {}
            with run_file.open('r', encoding='utf-8') as f:
                for line in f:
                    query_id, _, doc_id, _, score, _ = line.strip().split()
                    bm25_run.setdefault(query_id, {})[doc_id] = float(score)

            bm25_run = {
                query_id: dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:args.k])
                for query_id, doc_scores in bm25_run.items()
            }

        bm25_outputs['qids'] = [i for i in bm25_run for _ in bm25_run[i]]
        id_to_query = {ex['id']: ex['query'] for ex in queries}
        bm25_outputs['queries'] = [id_to_query[qid] for qid in bm25_outputs['qids']]
        bm25_outputs['docids'] = [docid.replace(' ', '_') for i in bm25_run for docid in bm25_run[i]]
        id_to_passage = {ex['id'].replace(' ', '_'): ex['content'] for ex in corpus}
        bm25_outputs['passage_texts'] = [id_to_passage[docid] for docid in bm25_outputs['docids']]    
        bm25_outputs['bm25_scores'] = [bm25_run[i][docid] for i in bm25_run for docid in bm25_run[i]]
        if 'bm25' in str(run_file).lower():
            bm25_output_filename = os.path.join(os.path.dirname(args.output_filename), f'bm25_{args.corpus_name}') 
        elif 'reasonir' in str(run_file).lower():
            bm25_output_filename = os.path.join(os.path.dirname(args.output_filename), f'reasonir_{args.corpus_name}') 
        else:
            bm25_output_filename = os.path.join(os.path.dirname(args.output_filename), f'first_stage_{args.corpus_name}')
    write_scores_to_file(all_qids=bm25_outputs['qids'],
                            all_docids=bm25_outputs['docids'], 
                            scores=bm25_outputs['bm25_scores'], 
                            output_filename=bm25_output_filename)
    print("Quickly evaluating BM25....", flush=True)
    evaluate(args.corpus_name, bm25_output_filename, qrels_path=args.qrels_path)
    prompt = get_prompt(args.corpus_name)
    context_size=32768
    num_gpus=args.num_gpus
    model_name=args.model_name
    if model_name == 'rank0':
        reranker = Rank0(
            base_model_name_or_path=args.model_path,
            lora_module=args.lora_module,
            context_size=context_size,
            num_gpus=num_gpus,
            dataset_prompt=prompt,
        )
    elif model_name == 'rank1':
        reranker = Rank1(
            model_name_or_path=args.model_path,
            context_size=context_size,
            num_gpus=num_gpus,
            dataset_prompt=prompt,
        )
    elif model_name == 'rank1_ours':
        reranker = Rank1OurImplementation(
            base_model_name_or_path=args.model_path,
            lora_module=args.lora_module,
            context_size=context_size,
            num_gpus=num_gpus,
            dataset_prompt=prompt,
        )
    elif model_name == 'rank1_no_reason':
        reranker = Rank1NoReason(
            base_model_name_or_path=args.model_path,
            lora_module=args.lora_module,
            context_size=context_size,
            num_gpus=num_gpus,
            dataset_prompt=prompt,
        )
    elif model_name == 'rank1_self_consistency':
        reranker = Rank1SelfConsistency(
            base_model_name_or_path=args.model_path,
            lora_module=args.lora_module,
            context_size=context_size,
            num_gpus=num_gpus,
            dataset_prompt=prompt,
        )
    print("Now reranking!", flush=True)
    reranker_output_filename = args.output_filename
    reranker_scores = reranker.predict(queries=bm25_outputs['queries'], passages=bm25_outputs['passage_texts'])

    write_scores_to_file(bm25_outputs['qids'], bm25_outputs['docids'], reranker_scores, reranker_output_filename)
    evaluate(args.corpus_name, reranker_output_filename, qrels_path=args.qrels_path)
