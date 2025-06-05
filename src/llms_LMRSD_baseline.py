# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/LMRSD/blob/main/LICENSE.

import os
import gc
import time
import json
import torch
import pathlib
import logging
import datasets
import numpy as np
import polars as pl
from tqdm import tqdm
from typing import Any, List
from outlines import models, generate
from pydantic import BaseModel, Field, constr, conlist
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# set polars threads and new eager engine
os.environ['POLARS_MAX_THREADS'] = "12"
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger.info(f"Polars concurrency set to {pl.thread_pool_size()} threads.")

# structured output schema for peer review scoring
class PeerReviewScoringTuple(BaseModel):
    description: str
    score: int

# structured output schema for paper idea
class PeerReviewIdea(BaseModel):
    idea_only_review_confidence: PeerReviewScoringTuple
    idea_only_review_content: str
    idea_only_review_rating: PeerReviewScoringTuple

# structured output schema for paper content
class PeerReviewPaperContent(BaseModel):
    review_confidence: PeerReviewScoringTuple
    review_content: str
    review_rating: PeerReviewScoringTuple

# structured output schema for peer review scoring
class PeerReview(BaseModel):
    peer_review_title: str
    peer_review_summary: str
    peer_review_paper_idea: PeerReviewIdea
    peer_review_paper_content: PeerReviewPaperContent

# helper function to load/initalize the prompt
def process_prompt(raw_text, tokenizer, device, task="idea", prompt_type="prompt"):
    """
    Given raw input text generate a prompt that will
    be supplied to a preference dataset loader.

    Parameters
    ------------
    arg1 | raw_text: str
        Raw input text without prompt template
    arg2 | tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast
        Tokenizer from the model
Qwen3-0.6B    arg3 | device: str
        Device name for the inputs and attention masks to sit on
    arg4 | task: str[OPTIONAL]
        Task type either Review the "idea" of paper or review the "content"
    arg5 | prompt_type: str[OPTIONAL]
        String flag to be applied at the top of messages to create "prompt"
        "chosen" or "rejected" chat responses for the preference dataset

    Returns
    ------------
        Text
    """
    # init
    prompt = None
    messages = []
    add_generation_prompt = True
    sys_prompt, user_prompt, input_text = None, None, None

    # init system prompt available
    sys_prompt = """
    You are an expert peer reivew agent working on assessing
    evaluating and giving a review score for the ideas and
    content represented in any given scientific texts.
    """

    # review only the idea of the paper
    if task == "idea":
        # init user prompt for the task
        user_prompt = """
        **Task:** You are given Paper title, abstract and keywords of a scientific 
        paper. Your goal is to accurately analyze the entire manuscript and play the
        role of a peer-reviewer to evaluate the ideas presented in the paper. You need
        to give a numerical score outlining your review and confidence in your decision.

        **Considerations:**
        1. You have to give a review of the idea from the range 1 to 10.
        2. Your rating of the paper's idea must include a confidence on the range of 1 to 10.
        3. You will be given papers across different scientific fields so be adaptable when reviewing the idea.
        4. DONOT hallucinate and produce new information.

        **Response Format:**
        ```json
        {
        'idea_only_review_confidence': int
        'idea_only_review_content': str
        'idea_only_review_rating': int
        }
        ```
        """

        # init the prompt
        input_text = """
        **Paper Title:**
        ```plaintext
        TITLE
        ```

        **Paper Abstract:**
        ```plaintext
        ABSTRACT
        ```

        **Keywords:**
        ```plaintext
        KEQwen3-0.6BYWORDS
        ```
        """
    else:
        # init user prompt for the task
        user_prompt = """
        **Task:** You are given Paper title, keywords and full text of a scientific
        paper. Your goal is to accurately analyze the entire manuscript and play the
        role of a peer-reviewer to evaluate the entire manuscript of the paper. You need
        to give a numerical score outlining your full paper review and confidence in 
        your decision.

        **Considerations:**
        1. You have to give a review of the idea from the range 1 to 10.
        2. Your rating of the paper's idea must include a confidence on the range of 1 to 10.
        3. You will be given papers across different scientific fields so be adaptable when reviewing the idea.
        4. Carefully evaluate the full content of the paper and donot jump to quick conclusions.
        5. DONOT hallucinate and produce new information.

        **Response Format:**
        ```json
        {
        'full_text_review_confidence': int
        'full_text_review_content': str
        'full_text_review_rating': int
        }
        ```
        """

        # init the prompt
        input_text = """
        **Paper Title:**
        ```plaintext
        TITLE
        ```

        **Paper Full text:**
        ```plaintext
        FULL_TEXT
        ```
        """

    # apply chat template on the chosen/rejected response
    if prompt_type == "chosen":
        # set the chosen response for the preferences
        messages.append([{"role": "assistant", "content": raw_text}])

        # apply prompt template
        add_generation_prompt = False

        # apply prompt and remove the system prompt
        prompt = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               use_system_prompt=add_generation_prompt,
                                               add_generation_prompt=add_generation_prompt)
    elif prompt_type == "rejected":
        # set the rejected response for the preferences
        messages.append([{"role": "assistant", "content": raw_text}])

        # apply prompt template
        add_generation_prompt = False

        # apply prompt and remove the system prompt
        prompt = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               use_system_prompt=add_generation_prompt,
                                               add_generation_prompt=add_generation_prompt)
    else:
        # adjust and replace Paper title, abstract, keywords and full text based on task
        if task == "idea":
            TITLE = raw_text["paper_title"].strip()
            ABSTRACT = raw_text["paper_abstract"].strip()
            KEYWORDS = raw_text["paper_keywords"].strip()
            input_text = input_text.replace("TITLE", TITLE).replace("ABSTRACT", ABSTRACT).replace("KEYWORDS", KEYWORDS)
        else:
            TITLE = raw_text["paper_title"].strip()
            FULL_TEXT = raw_text["paper_content"].strip()
            input_text = input_text.replace("TITLE", TITLE).replace("FULL_TEXT", FULL_TEXT)

        # set the prompt for the preferences
        messages.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt + input_text}
        ])

        # apply prompt template
        prompt = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               use_system_prompt=add_generation_prompt,
                                               add_generation_prompt=add_generation_prompt)

    # return the processed prompt
    return prompt

# helper function to load/initalize the model
def lmrsd_zero_shot_eval(task, dataset):
    """
    Given a model path, load tokenizer-model
    pair and return the objects tagged to the
    given device (cpu/cuda)

    Parameters
    ------------
    arg1 | model_name: str
        Use model catalog to load local model weights
    arg2 | device: str
        Hardware acceleration, defaults to "cpu" if any errors arise

    Returns
    ------------
        Tuple(AutoModel, AutoTokenizer) for local (model_client, model_name)
    """
    # model path
    base_model_path = "/kellogg/proj/dashun/LLM/HuggingFaceCache/"

    # set the model-id
    model_catalog = [
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Llama-3.3-70B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B"
    ]

    # op dir path
    output_dir = "/projects/p32534/code/LMRSD/lmrsd_zs_eval_outputs"
    output_file_path = os.path.join(output_dir, f"{task}.jsonl")

    # iterate and run
    for model_name in model_catalog:
        # set a model-id
        model_id = base_model_path + model_name

        # get device status
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              ("mps" if torch.backends.mps.is_available() else "cpu"))
        logger.info("----------------------------------")
        logger.info(
            f"Using {device} to run SciRIFF task: {task} on {model_name}")
        logger.info("----------------------------------")

        # get model-tokenizer pair
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.padding_side = "left"

        # load the model
        if "gemma" in model_name:
            attn_implementation = "eager"
        else:
            attn_implementation = "sdpa"

        # load model
        # enable bfloat16 for half-precision training
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     torch_dtype="bfloat16",
                                                     trust_remote_code=True,
                                                     low_cpu_mem_usage=True,
                                                     attn_implementation=attn_implementation,
                                                     device_map="auto")

        # is it a llama tokenizer ?
        if "llama" in model_name.lower():
            # pa/kellogg/proj/dashun/LLM/HuggingFaceCache/d token if needed
            tokenizer.add_special_tokens(
                {"pad_token": "<|finetune_right_pad_id|>"})
            logger.info(
                f"Setting <|finetune_right_pad_id|> token for {model_id}")
            model.resize_token_embeddings(len(tokenizer))

            # llama prompt template
            llama_template = r"""
            {% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
            """

            # set the chat template
            tokenizer.chat_template = llama_template

        # load time
        end = time.time()
        logger.info(f"Model-tokenizer Load Time:, {end - start} seconds")
        logger.info("----------------------------------")

        # tokenizer quirks
        logger.info(f"BOS token-id: {tokenizer.bos_token_id}")
        logger.info(f"EOS token-id: {tokenizer.eos_token_id}")
        logger.info(f"PAD token-id: {tokenizer.pad_token_id}")

        # set top_p and temperature to none for reproducible outputs
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

        # sequential iterate over dataset for zs-eval
        instances = []
        with open(output_file_path, "a", encoding="utf-8") as f_out:
            for i in tqdm(range(len(dataset))):
                paperid = dataset[i]["paper_id"]
                text = dataset[i]["input"]
                y_true = dataset[i]["output"]

                # get the tokenized representations of the input title
                input_encoded = tokenizer(text, padding=True, return_tensors="pt")
                input_encoded_ids = input_encoded["input_ids"].to(model.device)
                input_encoded_attn_mask = input_encoded["attention_mask"].to(model.device)
                input_shape = input_encoded["input_ids"].shape[1]

                # get the outputs
                outputs = model.generate(input_ids=input_encoded_ids,
                                         attention_mask=input_encoded_attn_mask,
                                         max_new_tokens=4096,
                                         do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=tokenizer.eos_token_id)

                # decode generated text
                output = tokenizer.decode(
                    outputs[0][input_shape:], skip_special_tokens=True)

                # save the output to file
                result = {
                    "model_name": model_name,
                    "input": text,
                    "y_true": y_true,
                    "output": output
                }
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()

                # randomly print an example to showcase the model outputs
                random_snap_points = [5, 10, 15, 20]
                # if i in random_snap_points:
                #    logger.info(f"Randomly printing an example: index={i+1}:")
                #    logger.info(f"Ground Truth: {y_true}")
                #    logger.info(f"Prediction: {output}")
                #logger.info("----------------------------------")

                # add for metric
                try:
                    pred_json = json.loads(output)
                except Exception as e:
                    pred_json = {}
                    logger.info(e)

                # add the instance
                instances.append(
                    {"pred": pred_json, "ref": json.loads(y_true)})

        # evaluate scores for the model
        scores = metric.evaluate(instances)
        logger.info(
            f"{model_name} ner_f1_typed on {task}: {scores['typed']['f1']}")
        logger.info(
            f"{model_name} ner_f1_untyped on {task}: {scores['untyped']['f1']}")

        logger.info(
            f"Completed running zs-evals for len(dataset) items on {model}")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # fin
    logger.info("Completed Zero-shot evaluation on task: {task} successfully!")


if __name__ == "__main__":
    # set seed
    set_seed(2025)

    # data dir
    data_dir = "/projects/p32534/code/LMRSD/data"
    logger.info(f"Using data from {data_dir}")

    # load dataset on disk
    logger.info("Loading datasets from parquet files...")
    raw_datasets = datasets.load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(data_dir, "lmrsd_idea_evaluation.parquet"),
        }
    )
    dataset = raw_datasets["train"]

    logger.info(f"LMRSD dataset length: {len(dataset)}")

    logger.info("Sample Input from LMRSD dataset:")
    logger.info(f"TITLE:\n{dataset[0]['paper_title']}")
    logger.info("----------------------------------")
    logger.info(f"ABSTRACT:\n{dataset[0]['paper_abstract']}")
    logger.info("----------------------------------")
    logger.info(f"KEYWORDS:\n{dataset[0]['paper_keywords']}")
    logger.info("----------------------------------")
    #logger.info(f"IDEA Evaluation\n: Rating: {dataset[0]['idea_only_review_rating_score']}, Confidence: {dataset[0]['idea_only_review_confidence_score']}")

    # run the eval on all models for task
    LMRSD_TASK = "idea"
    tokenizer = AutoTokenizer.from_pretrained("/kellogg/proj/dashun/LLM/HuggingFaceCache/Qwen3-0.6B", trust_remote_code=True)
    tokenizer.padding_side = "left"
    sample_prompt = process_prompt(dataset[0], tokenizer, "cpu", task="idea", prompt_type="prompt")
    logger.info(f"Sample prompt:\n{sample_prompt[0]}")

    #lmrsd_zero_shot_eval(task=LMRSD_TASK, dataset=dataset)






