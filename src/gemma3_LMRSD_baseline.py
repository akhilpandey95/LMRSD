# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/LMRSD/blob/main/LICENSE.

import os
import gc
import time
import json
import base64
import pathlib
import contextlib
import argparse
import logging
import datasets
import polars as pl
from tqdm import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
logger.info(f"Polars concurrency set to {pl.thread_pool_size()} threads.")

# directory constants
#data_dir = pathlib.Path("/home/ubuntu/data")
data_dir = pathlib.Path("/projects/p32534/code/LMRSD/data")
#output_dir = pathlib.Path("/home/ubuntu/lmrsd_zs_eval_outputs")
output_dir = pathlib.Path("/projects/p32534/code/LMRSD/lmrsd_zs_eval_outputs")

# rate limiting constants
RPM_LIMIT = 30
TPM_LIMIT = 15000
REQUEST_DELAY = 60 / RPM_LIMIT

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

# helper function to handle rate limiting
def rate_limit_sleep():
    time.sleep(REQUEST_DELAY)

# helper function to resume text generation
def get_gen_state(path):
    done = defaultdict(set)
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done[rec["model_name"]].add(rec["paperid"])
            except Exception:
                # corrupt line? ignore – safer to regenerate
                continue
    return done

# helper function to load/initalize the prompt
def process_prompt(raw_text, task="idea"):
    """
    Given raw input text generate a prompt that will
    be supplied to a preference dataset loader.

    Parameters
    ------------
    arg1 | raw_text: dict
        Raw input text without prompt template
    arg2 | task: str[OPTIONAL]
        Task type either Review the "idea" of paper or review the "content"

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
        KEYWORDS
        ```
        """

        # adjust and replace Paper title, abstract, keywords and full text based on task
        TITLE = raw_text["paper_title"].strip()
        ABSTRACT = raw_text["paper_abstract"].strip()
        KEYWORDS = raw_text["paper_keywords"].strip()
        input_text = input_text.replace("TITLE", TITLE).replace("ABSTRACT", ABSTRACT).replace("KEYWORDS", KEYWORDS)
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

        # adjust and replace Paper title, abstract, keywords and full text based on task
        TITLE = raw_text["paper_title"].strip()
        FULL_TEXT = raw_text["paper_content"].strip()
        input_text = input_text.replace("TITLE", TITLE).replace("FULL_TEXT", FULL_TEXT)

    # return sys + user prompt
    return sys_prompt, user_prompt + input_text

# helper function with retry mechanism for API calls
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(f"API call failed, retrying in {retry_state.next_action.sleep} seconds...")
)
def generate_with_retry(client, model, contents, config):
    """
    Generate content with retry mechanism for rate limiting and API errors
    
    Parameters
    ------------
    client: genai.Client
        Google GenAI client
    model: str
        Model name to use
    contents: list
        Content list for generation
    config: types.GenerateContentConfig
        Generation configuration
        
    Returns
    ------------
    response: Generated response from API
    """
    try:
        # simple generate
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        # return the response
        return response
    except Exception as e:
        logger.warning(f"API call failed: {str(e)}")
        raise

# helper function to load/initalize the model
def lmrsd_gemma_zero_shot_eval(task, dataset, llm, opfile):
    """
    Given a gemma3 model id, load paper
    object and return the ZS eval

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
    # set the model-id
    model_catalog = {
        "1b": "gemma-3-1b-it",
        "4b": "gemma-3-4b-it",
        "12b": "gemma-3-12b-it",
        "27b": "gemma-3-27b-it"
    }

    # filter the model catalog
    model = model_catalog[llm]

    # op dir path
    output_file_path = os.path.join(output_dir, f"{opfile}.jsonl")
    progress = get_gen_state(output_file_path)

    # sequential iterate over dataset for zs-eval
    with open(output_file_path, "a", encoding="utf-8") as f_out:
        for start in range(0, len(dataset)):
            # start and end
            end = min(start + 1, len(dataset))
            idx_range = range(start, end)

            # get the batch
            processed = progress.get(model, set())
            batch = [dataset[i] for i in idx_range if dataset[i]["paper_id"] not in processed]

            # is the batch empty ?
            if not batch:
                continue

            # generate
            for row in batch:
                # prepare system and user prompt
                sys_prompt, usr_prompt = process_prompt(row, task=task)

                # user input
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=sys_prompt + usr_prompt)]
                    ),
                ]

                # generate config
                generate_content_config = types.GenerateContentConfig(
                    seed=2025,
                    temperature=0,
                    top_p=0.95,
                    top_k=20,
                    max_output_tokens=4096,
                    response_mime_type="text/plain",
                )


                # capture response
                #response = client.models.generate_content(
                #    model=model,
                #    contents=contents,
                #    config=generate_content_config
                #)

                # init op
                output = None

                try:
                    # rate limit before API call
                    rate_limit_sleep()

                    # capture response with retry mechanism
                    response = generate_with_retry(
                        client=client,
                        model=model,
                        contents=contents,
                        config=generate_content_config
                    )

                    # check if response is not empty
                    if not response.text:
                        # log the error
                        logger.info("Failed to generate content for {row['paper_id']} from {llm}")
                    else:
                        # get the output
                        output = response.text
                except Exception as e:
                    logger.error(f"Failed to generate content for {row['paper_id']} from {llm}: {str(e)}")
                    output = None

                # save the output to file
                result = {
                    "model_name": model,
                    "paperid": row["paper_id"],
                    "y_true_med": row["median_idea_score"],
                    "y_true_med_cf": row["median_idea_score_cf"],
                    "y_true_avg": row["avg_idea_score"],
                    "y_true_avg_cf": row["avg_idea_score_cf"],
                    "input": sys_prompt + usr_prompt,
                    "output": output
                }

                # write teh output
                f_out.write(json.dumps(result) + "\n")
                processed.add(row["paper_id"])
                f_out.flush()

            # log
            if start % 10 == 0:
                logger.info(f"Processed {start} / {len(dataset)} samples.")

    # fin
    logger.info("Completed Zero-shot evaluation on task: {task} successfully!")

if __name__ == "__main__":
    # cli args
    parser = argparse.ArgumentParser(description="Compute lmrsd ZS eval for a given model")
    parser.add_argument("--data-file", required=True, help="Parquet file (LMRSD)")
    parser.add_argument("--llm", required=True, help="Model being used for ZS eval (LMRSD)")
    parser.add_argument("--opfile", required=True, help="Output file name for ZS eval (LMRSD)")
    args = parser.parse_args()

    # init google client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # data dir
    logger.info(f"Using data from {data_dir}")

    # load dataset on disk
    logger.info("Loading datasets from parquet files...")
    raw_datasets = datasets.load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(data_dir, args.data_file),
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
    logger.info(dataset[0].keys())
    logger.info(f"IDEA Evaluation\n: Rating: {dataset[0]['median_idea_score']}, Confidence: {dataset[0]['median_idea_score_cf']}")

    # run the eval on all models for task
    LMRSD_TASK = "idea"
    smpl_idx = 0
    sys, usr = process_prompt(dataset[smpl_idx], task=LMRSD_TASK)
    logger.info(f"Sample prompt:\n{sys+usr}")

    # run the zs pipeline
    lmrsd_gemma_zero_shot_eval(task=LMRSD_TASK, dataset=dataset, llm=args.llm, opfile=args.opfile)

