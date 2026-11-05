"""Adapted from the original Rank1 implementation:
https://github.com/orionw/rank1/blob/main/rank1.py
"""

import os
import math
from typing import List, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


class Rank1:
    name: str = "rank1"

    def __init__(
        self,
        model_name_or_path: str,
        context_size: int = 16000,
        max_output_tokens: int = 8192,
        fp_options: str = "float16",
        num_gpus: int = 1,
        dataset_prompt: str = None,
    ):
        """
        rank1 is a reasoning reranker model (using test-time compute) which generates a reasoning chain before deciding true or false

        Args:
            model_name_or_path: Path to the model or name of the model on HuggingFace Hub
            context_size: Maximum context length for the model (default: 16000)
            max_output_tokens: Maximum number of tokens to generate (default: 8192)
            fp_options: Floating point precision to use, e.g. 'float16' (default: 'float16')
            num_gpus: Number of GPUs to use for tensor parallelism (default: 1)
        """        
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.num_gpus = num_gpus
        self.model_name_or_path = model_name_or_path
        self.dataset_prompt = dataset_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.true_token = self.tokenizer(" true", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer(" false", add_special_tokens=False).input_ids[0]

        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            dtype=fp_options,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_output_tokens,
            logprobs=20,
            stop=["</think> true", "</think> false"],
            skip_special_tokens=False
        )

    def _fix_incomplete_responses(
        self, 
        original_prompts: List[str], 
        generated_texts: List[str]
    ) -> Tuple[List[str], List[int], List[float]]:
        """Close incomplete reasoning and force a true/false relevance prediction."""
        cleaned_texts = []
        for text in generated_texts:
            text = text.rstrip()
            if not text.endswith(('.', '!', '?')):
                last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last_punct != -1:
                    text = text[:last_punct + 1]
            cleaned_texts.append(text.strip())
        
        forced_prompts = [
            f"{original_prompt}\n{cleaned_text}\n</think>" 
            for original_prompt, cleaned_text in zip(original_prompts, cleaned_texts)
        ]

        new_sampling_args = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
            skip_special_tokens=False
        )
        outputs = self._generate(forced_prompts, new_sampling_args)

        all_final_texts = []
        all_token_counts = []
        all_scores = []    
        for i in range(len(outputs)):
            try:
                text = outputs[i].outputs[0].text
                final_logits = outputs[i].outputs[0].logprobs[-1]
                assert self.false_token in final_logits and self.true_token in final_logits, f"final logits are missing true or false: {final_logits}"
            except Exception as e:
                print(f"Error: {e} on fixing error, setting at 0.5 score: {outputs[i].outputs}")
                all_scores.append(0.5)
                all_token_counts.append(len(outputs[i].outputs[0].token_ids))
                all_final_texts.append(text)
                continue
                
            token_count = len(outputs[i].outputs[0].token_ids)
            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            
            all_final_texts.append(text)
            all_token_counts.append(token_count)
            all_scores.append(score)
        
        return all_final_texts, all_token_counts, all_scores

    def _generate(self, prompts, sampling_params):
        return self.model.generate(prompts, sampling_params)

    def _generate_model_outputs(self, prompts):
        return self._generate(prompts, self.sampling_params)

    def _process_with_vllm(self, prompts):
        outputs = self._generate_model_outputs(prompts)

        total_length = len(prompts)
        all_outputs = [None] * total_length
        all_output_token_counts = [None] * total_length
        all_scores = [None] * total_length
        
        incomplete_prompts = []
        incomplete_texts = []
        incomplete_indices = []
        
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            try:
                final_logits = output.outputs[0].logprobs[-1]
            except Exception as e:
                print(f"Error: {e} on getting final logits: {output.outputs[0]}")
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                incomplete_indices.append(i)
                continue

            if self.true_token not in final_logits or self.false_token not in final_logits:
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                incomplete_indices.append(i)
                continue
                
            token_count = len(output.outputs[0].token_ids)
            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            
            all_outputs[i] = text
            all_output_token_counts[i] = token_count
            all_scores[i] = score
        
        if incomplete_indices:
            fixed_texts, fixed_counts, fixed_scores = self._fix_incomplete_responses(
                incomplete_prompts, incomplete_texts
            )
            
            for orig_idx, (text, count, score) in zip(
                incomplete_indices, zip(fixed_texts, fixed_counts, fixed_scores)
            ):
                all_outputs[orig_idx] = text
                all_output_token_counts[orig_idx] = count
                all_scores[orig_idx] = score

        return all_outputs, all_output_token_counts, all_scores

    def return_prompt(self, query, doc_content, prompt) -> str:
        query = prompt.replace("FILL_QUERY_HERE", query) if prompt else query
        return "Determine if the following passage is relevant to the query. " \
                "Answer only with 'true' or 'false'.\n" \
                f"Query: {query}\n" \
                f"Passage: {doc_content}\n" \
                "<think>" 

    @torch.inference_mode()
    def predict(self, queries, passages):
        """Return relevance scores for query-passage pairs."""

        prompts = [
            self.return_prompt(query, passage, self.dataset_prompt)
            for query, passage in zip(queries, passages)
        ]

        _, _, scores = self._process_with_vllm(prompts)
        return scores
