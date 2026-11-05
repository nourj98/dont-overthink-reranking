# Don’t “Overthink” Pointwise Reranking: Is Reasoning Truly Necessary?

Official implementation of **[Don’t “Overthink” Pointwise Reranking: Is Reasoning Truly Necessary?](https://aclanthology.org/2026.findings-emnlp.2158/)** by Nour Jedidi, Yung-Sung Chuang, James R. Glass, and Jimmy Lin, published in **Findings of EMNLP 2026**.

## Overview

With the success of reasoning models across complex natural language tasks, researchers in the information retrieval (IR) community have begun exploring how similar reasoning capabilities can be integrated into passage rerankers built on large language models (LLMs). These methods typically employ an LLM to produce an explicit, step-by-step reasoning process before arriving at a final relevance prediction.

But does reasoning actually improve pointwise reranking? We investigate this question through controlled comparisons of Rank1, a pointwise reasoning reranker, with two non-reasoning counterparts. Across in-domain and out-of-domain datasets, we find that Rank1 generally underperforms Rank0, an otherwise equivalent non-reasoning pointwise reranker trained using the same backbone LLM and underlying training data as Rank1. To isolate the contribution of Rank1’s inference-time reasoning, we then replace its generated reasoning process with a fixed dummy sequence. Surprisingly, this variant, Rank1-NoReason, can also outperform Rank1. Analysis of Rank1’s score distributions provides a likely explanation: once Rank1 deems a passage relevant, it assigns nearly identical relevance scores, limiting its ability to distinguish degrees of relevance and ultimately harming its ranking accuracy.

## Setup

The original experiments used Python 3.10.16. Use a CUDA-capable GPU and Java 21 for Pyserini. Run the commands below from the repository root.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Required Python packages and pinned versions are listed in [`requirements.txt`](requirements.txt).

## Model names in the paper and code

Use the following values for `--model_name`:

| Paper model / variant | CLI mode | Model input |
| --- | --- | --- |
| Rank0 | `rank0` | Qwen2.5 base model + Rank0 LoRA adapter |
| Rank1 (our implementation) | `rank1_ours` | Qwen2.5 base model + our Rank1 LoRA adapter |
| Rank1 (released model) | `rank1` | A full model such as `jhu-clsp/rank1-7b` |
| Rank1-NoReason | `rank1_no_reason` | The same base model + Rank1 adapter, or a full Rank1 model |
| Rank1 + Self-Consistency | `rank1_self_consistency` | Qwen2.5 base model + our Rank1 LoRA adapter |

LoRA adapters contain only the fine-tuned adapter weights. Set `--model_path` to the corresponding base model and `--lora_module` to the adapter directory. A full released Rank1 model does not need a separate adapter.

## Step 0: Train Rank0 and Rank1

Skip this step if you already have trained adapters or want to evaluate a released Rank1 model.

Download `train.json` from the [Rank1 training dataset](https://huggingface.co/datasets/jhu-clsp/rank1-training-data). Both models use the same examples; Rank0 removes the reasoning from the training targets.

### Rank0

```bash
python -m training.train_reranker_sft \
  --base_model Qwen/Qwen2.5-7B \
  --sft_data_path /path/to/rank1-training-data/train.json \
  --output_model_path /path/to/models \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --learning_rate 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --epochs 1 \
  --remove_reasoning
```

This saves the adapter to `/path/to/models/Qwen2.5-7B_rank0`.

### Rank1 (our implementation)

Use the same settings and retain the reasoning by omitting `--remove_reasoning`:

```bash
python -m training.train_reranker_sft \
  --base_model Qwen/Qwen2.5-7B \
  --sft_data_path /path/to/rank1-training-data/train.json \
  --output_model_path /path/to/models \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --learning_rate 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --epochs 1
```

This saves the adapter to `/path/to/models/Qwen2.5-7B_rank1_ours`. These settings give an effective batch size of 128 for a single training process.

Use the corresponding Qwen2.5 model identifier to train another model size.

## Step 1: Prepare evaluation data

### TREC Deep Learning

For `dl19`, `dl20`, `dl21`, `dl22`, and `dl23`, the code loads topics, relevance judgments, and prebuilt BM25 indexes through Pyserini. No separate first-stage run file is required. Continue to Step 2.

### BRIGHT relevance judgments

The repository includes TREC-format qrels in `bright/pyserini_qrels/`. To regenerate them:

```bash
python bright/generate_pyserini_qrels.py
```

The script writes the files to `bright/pyserini_qrels/`. Reranking loads the documents and original queries from [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT).

### BRIGHT first-stage retrieval

To match the main paper experiments, use **BM25 + GPT-4 CoT** first-stage runs and rerank their top 100 passages. The reranker itself receives the **original query**, rather than the GPT-4 reasoning used for retrieval.

Download compatible runs from [Rank-R1’s BRIGHT BM25 + GPT-4 runs](https://github.com/ielab/llm-rankers/tree/main/Rank-R1/bright/bm25_gpt4_run), or generate them following the [BRIGHT evaluation instructions](https://github.com/xlang-ai/BRIGHT). Pass the selected dataset’s `trec.txt` to `--bright_run_file`.

Other first-stage runs can also be evaluated, but they change the candidate set and therefore the comparison with the paper. The loader accepts TREC runs or JSON mappings from query IDs to document-score mappings.

## Step 2: Rerank and evaluate

`rerank.py` retrieves or loads candidates, reranks the top `--k` passages per query, writes a TREC run, and evaluates **nDCG@10** for both the first-stage run and the reranked run. The examples below use `--k 100`.

### Rank0 on TREC DL19

```bash
python rerank.py \
  --model_path Qwen/Qwen2.5-7B \
  --model_name rank0 \
  --lora_module /path/to/models/Qwen2.5-7B_rank0 \
  --corpus_name dl19 \
  --k 100 \
  --output_filename results/dl19/rank0_7b.trec
```

### Rank1 (our implementation) on TREC DL19

```bash
python rerank.py \
  --model_path Qwen/Qwen2.5-7B \
  --model_name rank1_ours \
  --lora_module /path/to/models/Qwen2.5-7B_rank1_ours \
  --corpus_name dl19 \
  --k 100 \
  --output_filename results/dl19/rank1_ours_7b.trec
```

### Rank1-NoReason on TREC DL19

Reuse the **same Rank1 adapter** to isolate the effect of removing inference-time reasoning:

```bash
python rerank.py \
  --model_path Qwen/Qwen2.5-7B \
  --model_name rank1_no_reason \
  --lora_module /path/to/models/Qwen2.5-7B_rank1_ours \
  --corpus_name dl19 \
  --k 100 \
  --output_filename results/dl19/rank1_ours_noreason_7b.trec
```

### Released Rank1 on TREC DL19

```bash
python rerank.py \
  --model_path jhu-clsp/rank1-7b \
  --model_name rank1 \
  --corpus_name dl19 \
  --k 100 \
  --output_filename results/dl19/rank1_released_7b.trec
```

For its no-reasoning counterpart, use `--model_name rank1_no_reason` with the same full model and a distinct output filename. Omit `--lora_module` for both runs.

### BRIGHT

For BRIGHT, add **both** `--bright_run_file` and `--qrels_path`. For example, to evaluate Rank0 on TheoremQA Questions:

```bash
python rerank.py \
  --model_path Qwen/Qwen2.5-7B \
  --model_name rank0 \
  --lora_module /path/to/models/Qwen2.5-7B_rank0 \
  --corpus_name theoremqa_questions \
  --bright_run_file /path/to/theoremqa_questions_bm25_long_False/trec.txt \
  --qrels_path bright/pyserini_qrels/theoremqa_questions.tsv \
  --k 100 \
  --output_filename results/theoremqa_questions/rank0_7b.trec
```

To compare Rank1 and Rank1-NoReason, use `rank1_ours` and `rank1_no_reason`, respectively, with the Rank1 adapter. Keep the candidate run, qrels, and `--k` fixed across models, and give each reranked run a distinct output filename.

Supported BRIGHT datasets are `biology`, `earth_science`, `economics`, `psychology`, `robotics`, `stackoverflow`, `sustainable_living`, `leetcode`, `pony`, `aops`, `theoremqa_questions`, and `theoremqa_theorems`. When changing the dataset, update the corpus name, first-stage run, qrels, and output path together.

## Additional variants

### Self-consistency

Use `--model_name rank1_self_consistency` with the base model and our Rank1 adapter. This implementation averages relevance scores over eight sampled reasoning outputs. Add the same corpus and retrieval arguments as above.

### Inverse training

To train the label-first variant:

```bash
python -m training.train_rank1_inverse \
  --base_model Qwen/Qwen2.5-7B \
  --sft_data_path /path/to/rank1-training-data/train.json \
  --output_model_path /path/to/models \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --learning_rate 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --epochs 1
```

At inference, use `--model_name rank0` with the resulting `Qwen2.5-7B_rank1_inverse` adapter so that scoring uses the relevance label directly.

## Repository structure

```text
rerank.py          Inference and evaluation entry point
rerankers/         Reranker implementations and prompts
retrieval/         BM25 retrieval and index configuration
evaluation/        TREC run and evaluation utilities
training/          Training programs
bright/            BRIGHT qrels and qrels-generation utility
```

The implementation in `rerankers/rank1.py` is adapted from the [original Rank1 repository](https://github.com/orionw/rank1).

## Citation

```bibtex
@inproceedings{jedidi2026don,
  title={Don’t “Overthink” Pointwise Reranking: Is Reasoning Truly Necessary?},
  author={Jedidi, Nour and Chuang, Yung-Sung and Glass, James and Lin, Jimmy},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages={46159--46172},
  year={2026}
}
```
