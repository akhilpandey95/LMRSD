# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, you can obtain one at the repository root in LICENSE.

import os

# set polars threads and new eager engine
os.environ['POLARS_MAX_THREADS'] = "12"
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import asyncio
import pathlib
import argparse
import logging
import datasets
import polars as pl
from tqdm import tqdm
from openai import AsyncOpenAI as oai
from collections import defaultdict

# enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# set polars threads and new eager engine
logger.info(f"Polars concurrency set to {pl.thread_pool_size()} threads.")

# directory constants
data_dir = pathlib.Path("/projects/p32534/code/LMRSD/data")
output_dir = pathlib.Path("/projects/p32534/code/LMRSD/lmrsd_zs_eval_outputs")

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

# helper function with retry mechanism for API calls
async def generate_async(client, model, contents, retries=3, delay=5):
    """
    Generate content with OAI client mechanism and API structure asynchronously

    Parameters
    ------------
    client: oai.AsyncOpenAI
        Async OAI client
    model: str
        Model name to use
    contents: list
        Content list for generation
    retries: int
        Number of retry attempts on failure
    delay: int
        Initial delay in seconds for retries

    Returns
    ------------
    response: Generated response from API or None if failed
    """
    last_exception = None
    for attempt in range(retries):
        try:
            # simple async generate
            response = await client.chat.completions.create(
                model=model,
                messages=contents,
                max_tokens=32768,
                seed=2025,
                top_p=1.0,
                temperature=0.0,
                reasoning_effort="high",
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "separate_reasoning": True
                }
            )
            return response
        except Exception as e:
            last_exception = e
            logger.warning(f"API call failed on attempt {attempt + 1}/{retries}. Error: {str(e)}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= 2

    logger.error(f"API call failed after {retries} attempts. Final error: {str(last_exception)}")
    return None

# helper function for dynamic, token-aware batching
def dynamic_batch_generator(dataset, token_budget, max_output_tokens, input_token_key="tkn_count"):
    """
    Yields batches of samples from a dataset, ensuring the total
    token count (input + max_output) in each batch does not exceed
    the specified token_budget.

    Parameters
    ------------
    dataset: parquet
        The Hugging Face dataset, pre-sorted by token length.
    token_budget: int
        The maximum number of total tokens allowed in a batch
    max_output_tokens: int
        The max_tokens setting for generation
    input_token_key: str
        The column name for input token counts

    Returns
    ------------
        None
    """
    current_batch = []
    current_batch_token_cost = 0

    for sample in dataset:
        # worst-case memory cost for single sample
        sample_token_cost = sample[input_token_key] + max_output_tokens

        # unprocessable
        if sample_token_cost > token_budget:
            logger.warning(f"Skipping sample as its token cost ({sample_token_cost}) exceeds the total budget ({token_budget}).")
            continue

        # exceed budget ?, yield current batch
        if current_batch and (current_batch_token_cost + sample_token_cost > token_budget):
            yield current_batch

            # new batch
            current_batch = [sample]
            current_batch_token_cost = sample_token_cost
        else:
            # add sample to current batch
            current_batch.append(sample)
            current_batch_token_cost += sample_token_cost

    # yield the last batch if it's not empty
    if current_batch:
        yield current_batch

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

# helper function to process a batch of samples
async def process_batch(client, batch, model, semaphore, content_choice="post_pub"):
    """
    Process a batch of samples concurrently

    Parameters
    ------------
    client: oai.AsyncOpenAI
        Async OAI client
    batch: list
        List of dataset rows to process
    model: str
        Model name to use
    semaphore: asyncio.Semaphore
        Semaphore to control concurrent requests

    Returns
    ------------
    responses: List of responses from API
    """
    # process with semaphore to control concurrency
    async def process_single(row):
        async with semaphore:
            # get sys prompt, and input
            SYS_PROMPT, INP_PROMPT = process_prompt(row, content_choice)

            # contents
            contents = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": INP_PROMPT},
            ]
            # await generations
            return await generate_async(client, model, contents)

    # gather respones
    responses = await asyncio.gather(*[process_single(row) for row in batch])
    return responses

# helper function to load/initialize the model
async def lmrsd_zero_shot_eval(client, dataset, llm, opfile, content_choice, max_concurrent=32):
    """
    Given a TF model id, load paper
    object and return the ZS eval

    Parameters
    ------------
    client: oai.AsyncOpenAI
        Async OAI client
    dataset: Dataset
        Dataset to process
    llm: str
        Model name to use
    opfile: str
        Output file name for ZS eval
    content_choice: str
        Content choice for ZS eval (idea or content)
    max_concurrent: int
        Maximum concurrent API requests (default 32)

    Returns
    ------------
    None
    """
    # init semaphore for concurrent request control
    semaphore = asyncio.Semaphore(max_concurrent)

    # set the model-id
    model_catalog = {
        "r1-llama": "/path/to/HuggingFaceCache/DeepSeek-R1-Distill-Llama-70B",
        "r1-qwen": "/path/to/HuggingFaceCache/DeepSeek-R1-Distill-Qwen-32B",
        "llama-33": "/path/to/HuggingFaceCache/Llama-3.3-70B-Instruct/",
        "llama-4-scout": "/projects/p32494/ai4sciscibench/models/Llama-4-Scout-17B-16E-Instruct",
        "gemma-3-27b": "/path/to/HuggingFaceCache/gemma-3-27b-it",
        "gptoss-20b": "/projects/p32534/code/hypeline/models/gpt-oss-20b",
        "gptoss-120b": "/projects/p32534/code/hypeline/models/gpt-oss-120b",
        "tulu3-70b": "/projects/p32534/code/hypeline/models/Llama-3.1-Tulu-3-70B",
        "nemotron-49b": "/projects/p32534/code/hypeline/models/Llama-3_3-Nemotron-Super-49B-v1_5",
        "qwq-32b": "/projects/p32534/code/hypeline/models/QwQ-32B",
        "qwen3-32b": "/path/to/HuggingFaceCache/Qwen3-32B",
        "qwen3-moe": "/projects/p32494/ai4sciscibench/models/Qwen3-Next-80B-A3B-Thinking",
        "magistral-small": "/path/to/HuggingFaceCache/Magistral-Small-2506",
        "or-nemotron-32b": "/projects/p32534/code/hypeline/models/OpenReasoning-Nemotron-32B",
        "k2-think": "/projects/p32534/code/hypeline/models/K2-Think"
    }

    # filter the model catalog
    model = model_catalog[llm]

    # op dir path
    output_file_path = os.path.join(output_dir, f"{opfile}.jsonl")
    progress = get_gen_state(output_file_path)

    # capture the key based on the dataset
    key = "paper_id"

    # track timing
    total_processed_in_run = 0
    start_time = time.time()

    # max tokens for dynamic batching
    MAX_BATCH_TOTAL_TOKENS = 600000
    MAX_OUTPUT_TOKENS = 32768

    # init dynamic batch generator
    batch_generator = dynamic_batch_generator(dataset, MAX_BATCH_TOTAL_TOKENS, MAX_OUTPUT_TOKENS, input_token_key="tkn_count")
    total_samples_in_dataset = len(dataset)

    with open(output_file_path, "a", encoding="utf-8") as f_out:
        # iterate
        for batch in tqdm(batch_generator, desc="Processing dynamic batches"):
            batch_start_time = time.time()

            # filter samples already processed
            processed_in_session = progress.get(model, set())
            filtered_batch = [row for row in batch if row[key] not in processed_in_session]

            # is the batch empty?
            if not filtered_batch:
                continue

            # generate for batch
            responses = await process_batch(client, filtered_batch, model, semaphore, content_choice)
            for response, row in zip(responses, filtered_batch):
                # init op
                cot_output, output = None, None

                # get inp
                _, INP_PROMPT = process_prompt(row, content_choice)

                try:
                    # check if response is not empty
                    if response is not None and response.choices:
                        try:
                            # add reasoning COT + output
                            cot_output = response.choices[0].message.reasoning_content
                            output = response.choices[0].message.content
                        except Exception as e:
                            # get the output
                            output = response.choices[0].message.content
                    else:
                        logger.info(f"Failed to generate content for {row[key]} from {llm}")
                except Exception as e:
                    logger.error(f"Failed to generate content for {row[key]} from {llm}: {str(e)}")
                    output = None

                # output available?
                if output:
                    if content_choice == "post_pub":
                        # save the output to result list
                        result = {
                            "model_name": model,
                            key: row[key],
                            "openalex_id": row["paperid"],
                            "mag_concept_id": row["fieldid"],
                            "adversarial": row["type"],
                            "y_true_ft_med": row["median_ft_score"],
                            "y_true_ft_med_cf": row["median_ft_score_cf"],
                            "y_true_ft_avg": row["avg_ft_score"],
                            "y_true_ft_avg_cf": row["avg_ft_score_cf"],
                            "hit_1pct": row["Hit_1pct"],
                            "hit_5pct": row["Hit_5pct"],
                            "hit_10pct": row["Hit_10pct"],
                            "citation_count": row["citation_count"],
                            "input": INP_PROMPT,
                            "output": output,
                            "cot_output": cot_output
                        }
                    else:
                        # save the output to result list
                        result = {
                            "model_name": model,
                            key: row[key],
                            "y_true_ft_med": row["median_ft_score"],
                            "y_true_ft_med_cf": row["median_ft_score_cf"],
                            "y_true_ft_avg": row["avg_ft_score"],
                            "y_true_ft_avg_cf": row["avg_ft_score_cf"],
                            "input": "",
                            "output": "",
                            "cot_output": ""
                        }

                    # write all results for the batch
                    f_out.write(json.dumps(result) + "\n")
                    processed_in_session.add(row[key])
                    total_processed_in_run += 1

            # flush the output file
            f_out.flush()

            # log time
            total_time = time.time() - start_time
            samples_per_second = total_processed_in_run / total_time if total_time > 0 else 0

            # log total samples processed
            total_done = len(processed_in_session)
            eta_seconds = (total_samples_in_dataset - total_done) / samples_per_second if samples_per_second > 0 else 0

            # log
            logger.info(f"Processed {total_done} / {total_samples_in_dataset} samples. " f"Speed: {samples_per_second:.2f} samples/s, ETA: {eta_seconds/60:.1f} min")

    # fin
    total_time = time.time() - start_time
    logger.info(f"Completed Zero-shot evaluation for {llm} successfully! Total time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    # cli args
    parser = argparse.ArgumentParser(description="Compute result extraction for a given model")
    parser.add_argument("--data-file", required=True, help="Parquet file with Openarxiv id/DOI info data and prompts.")
    parser.add_argument("--llm", required=True, help="Model being used for ZS inference.")
    parser.add_argument("--opfile", required=True, help="Output file name for ZS inference.")
    parser.add_argument("--content-choice", type=str, required=True, help="Content choice for ZS inference (idea or full_text)")
    parser.add_argument("--backend", type=int, choices=[0, 1], required=True, help="Backend for ZS inference (0=vllm, 1=sglang)")
    parser.add_argument("--max-concurrent", type=int, default=32, help="Maximum concurrent API requests (default 32)")
    args = parser.parse_args()

    # params for vllm
    openai_api_key = "EMPTY"

    # capture backend
    """
    TODO: gotta fix this later to have similar ports
    """
    if args.backend == 0:
        # vllm
        openai_api_base = "http://localhost:8000/v1"
    else:
        # sglang
        openai_api_base = "http://127.0.0.1:30000/v1"

    # set client
    oai_client = oai(api_key=openai_api_key, base_url=openai_api_base)

    # data dir
    logger.info(f"Using data from {data_dir}")

    # load dataset on disk
    logger.info("Loading LMRSD parquet...")
    raw_datasets = datasets.load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(data_dir, args.data_file),
        }
    )

    # grab the dataset
    dataset = raw_datasets["train"]

    # srot by token count
    logger.info("Sorting dataset by token length for optimized and stable batching.")
    dataset = dataset.sort("tkn_count")

    # run the eval on all models for task
    smpl_idx = 0
    smpl_prmpt = process_prompt(dataset[smpl_idx])
    logger.info(f"LMRSD EXP4 Post-publication outcomes dataset length: {len(dataset)}")
    logger.info("----------------------------------")
    logger.info(dataset[0].keys())
    logger.info("----------------------------------")
    #logger.info(f"Sample prompt:\n{smpl_prmpt}")

    # run the zs pipeline
    asyncio.run(lmrsd_zero_shot_eval(oai_client, dataset=dataset, llm=args.llm, opfile=args.opfile, content_choice=args.content_choice, max_concurrent=args.max_concurrent))
