import math

import numpy as np
import torch
from vllm import SamplingParams

from .rank1_ours import Rank1OurImplementation


class Rank1SelfConsistency(Rank1OurImplementation):
    """Rank1 with the paper's stochastic reasoning scores averaged per pair."""

    def __init__(self, *args, num_samples: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=self.max_output_tokens,
            n=num_samples,
            logprobs=20,
            stop=["</think> true", "</think> false"],
            skip_special_tokens=False,
        )

    def _process_with_vllm(self, prompts):
        outputs = self._generate_model_outputs(prompts)
        all_texts = []
        all_scores = []
        incomplete_prompts = []
        incomplete_texts = []
        incomplete_locations = []

        for prompt_index, output in enumerate(outputs):
            prompt_texts = []
            prompt_scores = []
            for sample in output.outputs:
                text = sample.text
                prompt_texts.append(text)
                try:
                    final_logits = sample.logprobs[-1]
                    true_logit = final_logits[self.true_token].logprob
                    false_logit = final_logits[self.false_token].logprob
                except (IndexError, KeyError, TypeError):
                    incomplete_prompts.append(prompts[prompt_index])
                    incomplete_texts.append(text)
                    incomplete_locations.append((prompt_index, len(prompt_scores)))
                    prompt_scores.append(None)
                    continue

                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)
                prompt_scores.append(true_score / (true_score + false_score))

            all_texts.append(prompt_texts)
            all_scores.append(prompt_scores)

        if incomplete_locations:
            _, _, fixed_scores = self._fix_incomplete_responses(
                incomplete_prompts,
                incomplete_texts,
            )
            for (prompt_index, sample_index), score in zip(
                incomplete_locations,
                fixed_scores,
            ):
                all_scores[prompt_index][sample_index] = score

        return all_texts, [float(np.mean(scores)) for scores in all_scores]

    @torch.inference_mode()
    def predict(self, queries, passages):
        prompts = [
            self.return_prompt(query, passage, self.dataset_prompt)
            for query, passage in zip(queries, passages)
        ]
        _, scores = self._process_with_vllm(prompts)
        return scores
