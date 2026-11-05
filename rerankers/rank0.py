import os
import math 

from transformers import AutoTokenizer 

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .rank1 import Rank1

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

class Rank0(Rank1):
    """Paper baseline that predicts relevance without generating reasoning."""
    def __init__(
        self,
        base_model_name_or_path: str,
        lora_module: str, 
        context_size: int = 16000,
        max_output_tokens: int = 8192,
        fp_options: str = "float16",
        num_gpus: int = 1,
        dataset_prompt: str = None,
    ):

        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.num_gpus = num_gpus
        self.dataset_prompt = dataset_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.true_token = self.tokenizer("true", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("false", add_special_tokens=False).input_ids[0]
        
        self.lora_module = lora_module
        self.model = LLM(
            model=base_model_name_or_path,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            dtype=fp_options,
            enable_lora=True,
            max_lora_rank=32,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
            skip_special_tokens=False
        )
    
    def _generate_model_outputs(self, prompts):
        return self.model.generate(prompts, self.sampling_params, lora_request=LoRARequest("PointwiseRerankerAdapter", 1, self.lora_module))

    def _process_with_vllm(self, prompts):
        outputs = self._generate_model_outputs(prompts)
        total_length = len(prompts)
        all_outputs = [None] * total_length
        all_output_token_counts = [None] * total_length
        all_scores = [None] * total_length
        
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            try:
                final_logits = output.outputs[0].logprobs[-1]
                assert self.false_token in final_logits and self.true_token in final_logits, f"final logits are missing true or false: {final_logits}"
            except Exception as e:
                print(f"Error: {e} on fixing error, setting at 0.5 score: {output.outputs[0].logprobs[-1]}")                
                all_outputs[i] = text
                all_output_token_counts[i] = len(output.outputs[0].token_ids)
                all_scores[i] = 0.5
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
        
        return all_outputs, all_output_token_counts, all_scores

    def return_prompt(self, query, doc_content, prompt) -> str:
        query = prompt.replace("FILL_QUERY_HERE", query) if prompt else query
        chat = [
            {'role': "system", 'content': "Determine if the following passage is relevant to the query. Answer only with 'true' or 'false'."},
            {'role': "user", 'content': f"Query: {query}\nPassage: {doc_content}\n"},
            ]
       
        prompt_text = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        return prompt_text
