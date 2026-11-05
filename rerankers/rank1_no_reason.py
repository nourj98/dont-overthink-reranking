import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .rank0 import Rank0


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

class Rank1NoReason(Rank0):
    """Rank1 with the paper's dummy reasoning process prefilled at inference."""

    def __init__(
        self,
        base_model_name_or_path: str,
        lora_module: str = None,
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
        self.lora_module = lora_module
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.true_token = self.tokenizer(" true", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer(" false", add_special_tokens=False).input_ids[0]
        
        self.model = LLM(
            model=base_model_name_or_path,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            dtype=fp_options,
            enable_lora=lora_module is not None,
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
        if self.lora_module is None:
            return self.model.generate(prompts, self.sampling_params)
        return self.model.generate(
            prompts,
            self.sampling_params,
            lora_request=LoRARequest("Rank1Adapter", 1, self.lora_module),
        )

    def return_prompt(self, query, doc_content, prompt) -> str:
        query = prompt.replace("FILL_QUERY_HERE", query) if prompt else query
        chat = [
            {
                "role": "system",
                "content": (
                    "Determine if the following passage is relevant to the query. "
                    "Answer only with 'true' or 'false'."
                ),
            },
            {"role": "user", "content": f"Query: {query}\nPassage: {doc_content}\n"},
            {
                "role": "assistant",
                "content": "<think>\nOkay, I have finished thinking.\n</think>",
            },
        ]
        return self.tokenizer.apply_chat_template(
            chat,
            continue_final_message=True,
            tokenize=False,
        )
