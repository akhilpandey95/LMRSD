# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/LMRSD/blob/main/LICENSE.

import os

# set polars threads and new eager engine
os.environ['POLARS_MAX_THREADS'] = "12"
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import time
import json
import pathlib
import argparse
import logging
import datasets
import polars as pl
from collections import defaultdict
from vllm import LLM, SamplingParams
from typing import Any, Dict, List, Optional

# enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# set polars threads and new eager engine
logger.info(f"Polars concurrency set to {pl.thread_pool_size()} threads.")

# directory constants
output_dir = pathlib.Path("./")

# helper function to resume text generation
def get_gen_state(path):
    done = defaultdict(set)

    # set key
    key = "paper_id"

    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done[rec["model_name"]].add(rec[key])
            except Exception:
                # corrupt line? ignore – safer to regenerate
                continue
    return done

# helper function to calculate perplexity
def calculate_perplexity(logprobs: List[float]) -> float:
    """
    Calculate perplexity from log probabilities

    Parameters
    ------------
    logprobs: List[float]
        List of log probabilities

    Returns
    ------------
    perplexity: float
        Perplexity value
    """
    # check if logprobs is empty
    if not logprobs:
        return float('inf')

    # calculate average log probability
    avg_log_prob = sum(logprobs) / len(logprobs)

    # calculate perplexity
    return math.exp(-avg_log_prob)

# helper function to initialize the vllm model
def initialize_vllm_model(model_path: str, tensor_parallel_size: int = 4) -> LLM:
    """
    Initialize vLLM model with proper configuration for multi-GPU setup

    Parameters
    ------------
    model_path: str
        Path to the model
    tensor_parallel_size: int
        Number of GPUs to use for tensor parallelism (default 4 for 4xH100)

    Returns
    ------------
    llm: LLM instance
    """
    # init Flash-Attention backend
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    logger.info(f"Initializing vLLM model from {model_path}")
    logger.info(f"Using tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Flash Attention backend enabled")

    # vllm config
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        gpu_memory_utilization=0.85,
        max_num_seqs=100,
        max_num_batched_tokens=131072,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        swap_space=0,
        seed=2025,
    )

    logger.info("vLLM model initialized successfully with optimized settings")

    # return model obj
    return llm

# helper function to generate content for a batch
def generate_batch_vllm(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[Optional[Dict[str, Any]]]:
    """
    Generate content using vLLM for a batch of prompts

    Parameters
    ------------
    llm: LLM
        vLLM model instance
    prompts: List[str]
        List of formatted prompts
    sampling_params: SamplingParams
        Sampling parameters for generation

    Returns
    ------------
    outputs: List of generation payloads or None if failed
    """
    try:
        # batch promots and generate
        outputs = llm.generate(prompts, sampling_params)

        # get generated text from outputs
        generated_batches: List[Optional[Dict[str, Any]]] = []
        for output in outputs:
            if not output.outputs:
                generated_batches.append(None)
                continue

            completion = output.outputs[0]
            token_stats: List[Dict[str, Any]] = []
            logprob_values: List[float] = []
            cot_output: Optional[str] = None

            if completion.logprobs:
                for idx, logprob_dict in enumerate(completion.logprobs):
                    if not logprob_dict:
                        continue

                    chosen_token = None
                    if hasattr(completion, "tokens") and completion.tokens and idx < len(completion.tokens):
                        chosen_token = completion.tokens[idx]

                    # pick chosen token (rank 0 if available else highest logprob)
                    selected_token_str = None
                    selected_entry = None
                    for token_str, entry in logprob_dict.items():
                        if getattr(entry, "rank", None) == 0:
                            selected_token_str = token_str
                            selected_entry = entry
                            break

                    if selected_entry is None:
                        selected_token_str, selected_entry = max(
                            logprob_dict.items(),
                            key=lambda item: item[1].logprob,
                        )

                    logprob_values.append(selected_entry.logprob)
                    token_stats.append(
                        {
                            "token": chosen_token if chosen_token is not None else selected_token_str,
                            "token_str": selected_token_str,
                            "logprob": selected_entry.logprob,
                            "rank": getattr(selected_entry, "rank", None),
                            "top_logprobs": [
                                {
                                    "token": tok,
                                    "logprob": candidate.logprob,
                                    "rank": getattr(candidate, "rank", None),
                                }
                                for tok, candidate in sorted(
                                    logprob_dict.items(),
                                    key=lambda item: getattr(item[1], "rank", float("inf")),
                                )
                            ],
                        }
                    )

            perplexity = calculate_perplexity(logprob_values) if logprob_values else None
            # capture chain-of-thought / reasoning content when available
            cot_output = getattr(completion, "reasoning_output", None)
            if cot_output is None:
                cot_output = getattr(completion, "reasoning_content", None)
            if cot_output is None:
                cot_output = getattr(completion, "reasoning_text", None)
            if isinstance(cot_output, list):
                cot_output = "\n".join(str(chunk) for chunk in cot_output if chunk)

            generated_batches.append(
                {
                    "output": completion.text,
                    "cot_output": cot_output,
                    "token_logprobs": token_stats,
                    "logprob_values": logprob_values,
                    "perplexity": perplexity,
                    "cumulative_logprob": getattr(completion, "cumulative_logprob", None),
                    "num_output_tokens": len(logprob_values),
                }
            )

        return generated_batches
    except Exception as e:
        logger.warning(f"Batch generation failed: {str(e)}")
        return [None] * len(prompts)

# helper function to load/initalize the prompt
def process_prompt(raw_text, task="post_pub"):
    """
    Given raw input text generate a prompt that will
    be supplied to a preference dataset loader.

    Parameters
    ------------
    arg1 | raw_text: dict
        Raw input text without prompt template
    arg2 | task: str[OPTIONAL]
        Task type either Review the "post_pub" outcomes of paper.

    Returns
    ------------
        Text
    """
    # init
    prompt = None
    add_generation_prompt = True
    sys_prompt, user_prompt, input_text = None, None, None
    #print(raw_text)

    # init system prompt available
    sys_prompt = """
    You are an expert peer reivew agent working on assessing
    evaluating and giving a review score for the ideas and
    content represented in any given scientific texts.
    """

    # review only the idea of the paper
    if task == "post_pub":
        # init user prompt for the task
        user_prompt = """**Task:** You are given Paper title, keywords and full text of a scientific
paper. Your goal is to accurately analyze the entire manuscript and play the
role of a peer-reviewer to evaluate the entire manuscript of the paper. You need
to give a numerical score outlining your full paper review and confidence in
your decision.

**Considerations:**
1. You have to give a review of the idea from the range 1 to 10.
1. You have to also give a review of the entire full text of the paper from the range 1 to 10.
2. Your rating of the paper's idea and full text must include a confidence on the range of 1 to 10.
3. You will be given papers across different scientific fields so be adaptable when reviewing the idea.
4. Carefully evaluate the full content of the paper and donot jump to quick conclusions.
5. Review content should be elaborate and detailed, covering all sections of the paper typically more than 1000 words.
6. DONOT hallucinate and produce new information.

**Scoring Calibration:**
- 1-3: Fundamentally flawed
- 4-5: Below average, limited contribution
- 6-7: Solid work, useful but not exceptional
- 8: Exceptional - top 10% material
- 9: Outstanding - top 5% material
- 10: Paradigm-shifting - top 1% material

**Critical Context:**
- Only ~5% of papers should be considered truly exceptional (score 8+)
- Most papers, even good ones, should score between 4-7
- Reserve scores of 9-10 for paradigm-shifting work only

**Idea Review and Full Text Review combined OUTPUT JSON Format:**
```json
{
  'idea_only_review_confidence': int,
  'idea_only_review_content': str,
  'idea_only_review_rating': int,
  'full_text_review_confidence': int,
  'full_text_review_content': str,
  'full_text_review_rating': int
}
```
        """

        # init the prompt
        input_text = """**Paper Title:**
```plaintext
TITLE
```

**Keywords:**
```plaintext
KEYWORDS
```

**Paper Full text:**
```plaintext
FULL_TEXT
```
        """

        # adjust and replace Paper title, abstract, keywords and full text based on task
        TITLE = raw_text["paper_title"].strip()
        FULL_TEXT = raw_text["paper_content"].strip()
        KEYWORDS = raw_text["paper_keywords"].strip()
        input_text = input_text.replace("TITLE", TITLE).replace("FULL_TEXT", FULL_TEXT).replace("KEYWORDS", KEYWORDS)

    # return sys + user prompt
    return sys_prompt, user_prompt + input_text

# CRITICAL CHANGE: Format prompt for vLLM (combine system and user messages)
def format_prompt_for_vllm(sys_prompt: str, user_prompt: str, model_type: str = "llama") -> str:
    """
    Format the prompt according to the model's expected format

    Parameters
    ------------
    sys_prompt: str
        System prompt
    user_prompt: str
        User prompt
    model_type: str
        Type of model for formatting (llama, qwen, etc.)

    Returns
    ------------
    formatted_prompt: str
    """
    # grab chat templates or get basic format
    if "llama" in model_type.lower() or "gemma" in model_type.lower():
        # Llama-3 style formatting
        formatted = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
        formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen" in model_type.lower() or "qwq" in model_type.lower():
        # Qwen style formatting
        formatted = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        formatted += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
    else:
        # Generic format
        formatted = f"System: {sys_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    return formatted

# main processing function for running the ablation
def lmrsd_abl1(llm: LLM, dataset: datasets.Dataset, model_name: str, model_path: str, opfile: str, bs: int):
    """
    Given a vLLM model, process dataset for ablation 1

    Parameters
    ------------
    llm: LLM
        vLLM model instance
    dataset: Dataset
        Dataset to process
    model_name: str
        Model name identifier
    model_path: str
        Full model path
    opfile: str
        Output file name for ZS eval
    bs: int
        Batch size for ZS eval

    Returns
    ------------
    None
    """

    # greedy decoding and reproducible generation
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32768,
        seed=2025,
        logprobs=10,
    )

    # op dir path
    output_file_path = os.path.join(output_dir, f"{opfile}.jsonl")
    progress = get_gen_state(output_file_path)

    # capture the key based on the dataset
    key = "paper_id"

    # start performance monitoring
    total_processed = 0
    start_time = time.time()

    # model type for prompt formatting
    model_type = "llama"  # default
    if "qwen" in model_name.lower() or "qwq" in model_name.lower():
        model_type = "qwen"
    elif "gemma" in model_name.lower():
        model_type = "gemma"

    processed = progress[model_path]
    total_token_count = 0
    total_logprob_sum = 0.0
    perplexity_values: List[float] = []

    # iterate over dataset for zs-eval in batches
    batch_size = bs
    with open(output_file_path, "a", encoding="utf-8") as f_out:
        for start in range(0, len(dataset), batch_size):

            # start and end
            end = min(start + batch_size, len(dataset))
            idx_range = range(start, end)

            # get the batch
            batch = [dataset[i] for i in idx_range if dataset[i][key] not in processed]
            # is the batch empty?
            if not batch:
                continue

            # pack prompts for vLLM batches
            prompts = []
            input_prompts = []
            for row in batch:
                sys_prompt, user_prompt = process_prompt(row)
                formatted_prompt = format_prompt_for_vllm(sys_prompt, user_prompt, model_type)
                prompts.append(formatted_prompt)
                input_prompts.append(user_prompt)

            # llm generate
            outputs = generate_batch_vllm(llm, prompts, sampling_params)

            # extract outputs
            for output, row, inp_prompt in zip(outputs, batch, input_prompts):
                # save the output to result
                result = {
                    "model_name": model_path,
                    key: row[key],
                    "input": inp_prompt,
                    "output": output["output"] if output else None,
                    "cot_output": output["cot_output"] if output else None,
                    "perplexity": output["perplexity"] if output else None,
                    "logprob_values": output["logprob_values"] if output else None,
                    "cumulative_logprob": output["cumulative_logprob"] if output else None,
                    "num_output_tokens": output["num_output_tokens"] if output else None,
                    "token_logprobs": output["token_logprobs"] if output else None,
                }

                # write all results for the batch
                f_out.write(json.dumps(result) + "\n")
                processed.add(row[key])
                total_processed += 1

                if output and output["logprob_values"]:
                    total_logprob_sum += sum(output["logprob_values"])
                    total_token_count += len(output["logprob_values"])
                    if output["perplexity"] is not None:
                        perplexity_values.append(output["perplexity"])

            # flush the output file
            f_out.flush()

            if start % 100 == 0:
                logger.info(f"Processed {start} / {len(dataset)} samples")

    # fin
    total_time = time.time() - start_time
    if total_token_count > 0:
        avg_logprob = total_logprob_sum / total_token_count
        dataset_perplexity = math.exp(-avg_logprob)
        logger.info(
            f"Ablation 1 token stats -> avg logprob: {avg_logprob:.4f}, "
            f"dataset perplexity estimate: {dataset_perplexity:.4f}"
        )
    if perplexity_values:
        mean_sample_ppl = sum(perplexity_values) / len(perplexity_values)
        logger.info(f"Mean sample perplexity across {len(perplexity_values)} generations: {mean_sample_ppl:.4f}")
    logger.info(f"Completed Ablation 1 evaluation for {model_name} successfully! Total time: {total_time/60:.2f} minutes")

# kaboom
if __name__ == "__main__":
    # cli args
    parser = argparse.ArgumentParser(description="Compute result extraction for a given model")
    parser.add_argument("--dataset", required=True, help="Parquet file with Openarxiv id/DOI info data and prompts.")
    parser.add_argument("--llm", choices=["gptoss-120b", "qwen3-32b"], required=True, help="Model being used for ZS inference.")
    parser.add_argument("--opfile", required=True, help="Output file name for ZS inference.")
    parser.add_argument("--bs", type=int, required=True, help="Batch size for ZS inference.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Number of GPUs to use for tensor parallelism (default 4)")
    args = parser.parse_args()

    # model path
    model_path = {
        "gptoss-120b": "/projects/p32534/code/hypeline/models/gpt-oss-120b",
        "qwen3-32b": "/kellogg/proj/dashun/LLM/HuggingFaceCache/Qwen3-32B"
    }

    # get model path
    model_path = model_path[args.llm]

    # load dataset on disk
    logger.info(f"Loading dataset from {args.dataset} ...")
    raw_datasets = datasets.load_dataset("parquet", data_files={"train": str(args.dataset)})
    dataset = raw_datasets["train"]
    logger.info(f"LMRSD.c dataset length: {len(dataset)}")

    # init vLLM model
    llm = initialize_vllm_model(model_path, tensor_parallel_size=args.tensor_parallel_size)

    # run the ablation 1 pipeline
    lmrsd_abl1(llm=llm, dataset=dataset, model_name=args.llm, model_path=model_path, opfile=args.opfile, bs=args.bs)