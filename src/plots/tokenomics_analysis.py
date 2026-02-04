# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, you can obtain one at the repository root in LICENSE.

from __future__ import annotations

import os
import json
import argparse
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# ---------------------------------------------------------------
# Model catalogue (same as your eval script)
# ---------------------------------------------------------------
MODEL_CATALOG: List[str] = [
    "gemma-3-1b-it",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Qwen3-0.6B",
    "OLMo-2-0425-1B-Instruct",
]

MODEL_CATALOG_FAMILY = {
    "gemma-3-1b-it": "Gemma-3 Family",
    "Llama-3.1-8B-Instruct": "Llama-3.1 Family",
    "Llama-3.2-1B-Instruct": "Llama-3.2 Family",
    "Qwen3-0.6B": "Qwen3 Family",
    "OLMo-2-0425-1B-Instruct": "OLMo-2 Family"
}

BASE_MODEL_PATH = "/home/ubuntu/models/"

# ---------------------------------------------------------------
# Prompt builder (stripped-down: uses only title/abstract/keywords)
# ---------------------------------------------------------------

def process_prompt(row: dict, task: str = "idea") -> str:
    """Return the plain prompt string for a single dataset row."""
    sys_prompt = """You are an expert peer reivew agent working on assessing
        evaluating and giving a review score for the ideas and
        content represented in any given scientific texts."""

    if task == "idea":
        user_template = """
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
            ```.\n\n
            **Paper Title:**\n{TITLE}\n\n
            **Paper Abstract:**\n{ABSTRACT}\n\n
            **Keywords:**\n{KEYWORDS}\n
        """

        user_block = user_template.replace("TITLE", row["paper_title"].strip()).replace("ABSTRACT", row["paper_abstract"].strip()).replace("KEYWORDS", row["paper_keywords"].strip())
    else:
        user_template = """
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
            ```\n\n
            **Paper Title:**\n{TITLE}\n\n
            **Paper Full text:**\n{FULL_TEXT}\n
        """

        user_block = user_template.replace("TITLE", row["paper_title"].strip()).replace("FULL_TEXT", row["paper_content"].strip())

    return f"<system>\n{sys_prompt}\n</system>\n<user>\n{user_block}\n</user>\n<assistant>"

# ---------------------------------------------------------------
# Token counting logic
# ---------------------------------------------------------------

def count_tokens_for_model(
    tokenizer: AutoTokenizer, prompts: List[str]
) -> List[int]:
    lens: List[int] = []
    for prompt in prompts:
        # fast tokenisation: no attention mask, no tensors
        ids = tokenizer(prompt).input_ids
        lens.append(len(ids))
    return lens

# ---------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute token statistics per model")
    parser.add_argument("--data-file", required=True, help="Parquet file (LMRSD)")
    parser.add_argument("--output-dir", required=True, help="Where to write CSV/plots")
    parser.add_argument("--task", default="idea", choices=["idea", "full"], help="Prompt task")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------- load dataset once --------------------
    ds = datasets.load_dataset("parquet", data_files={"data": args.data_file})["data"]
    prompts = [process_prompt(row, task=args.task) for row in tqdm(ds, desc="build-prompts")]

    # store prompt length dicts for CSV
    stats_rows = []

    # ------------------- loop over models ----------------------
    all_lengths = {}
    for model_name in MODEL_CATALOG:
        print(f"\n>>> {model_name} – loading tokenizer…", flush=True)
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH + model_name, trust_remote_code=True)

        # ensure pad token exists to avoid warnings
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        lengths = count_tokens_for_model(tok, prompts)
        arr = np.array(lengths)
        all_lengths[model_name] = arr

        # ---------- write per-model histogram ------------------
        plt.figure(figsize=(6, 4))
        plt.style.use('classic')
        plt.grid(visible=True)
        plt.xticks(fontsize=8)
        plt.xticks(fontsize=8)
        plt.hist(arr, bins=50, alpha=0.8, edgecolor="black")
        plt.title(f"LMRSD: Overview of token distribution across – {model_name}", fontsize=11)
        #plt.xlabel("# of tokens per idea review prompt", fontsize=10, color="red")
        plt.xlabel("# of tokens per full paper review prompt", fontsize=10, color="red")
        plt.ylabel("# of papers", fontsize=10, color="red")
        plt.tight_layout()
        #plt.savefig(outdir / f"{model_name}_hist.png", dpi=150)
        plt.savefig(outdir / f"{model_name}_hist_ft.png", dpi=150)
        plt.close()

        # ---------- accumulate summary stats ------------------
        stats_rows.append({
            "model": model_name,
            "min": int(arr.min()),
            "mean": arr.mean().round(2),
            "p50": int(np.percentile(arr, 50)),
            "p95": int(np.percentile(arr, 95)),
            "max": int(arr.max()),
        })

    # ------------------- overlay plot --------------------------
    # ------------------- overlay plot --------------------------
    plt.figure(figsize=(8, 5))
    plt.style.use('classic')
    plt.grid(visible=True)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    for model_name, arr in all_lengths.items():
        plt.hist(arr, bins=60, histtype="step",
                 linewidth=1.2, label=MODEL_CATALOG_FAMILY[model_name], alpha=0.7)
    plt.legend(fontsize=7, ncol=2)
    #plt.xlabel("# of tokens per idea review prompt", fontsize=10, color="red")
    plt.xlabel("# of tokens per full paper review prompt", fontsize=10, color="red")
    plt.ylabel("# of papers", fontsize=10, color="red")
    plt.title("LMRSD: Overview of token distribution across different model families", fontsize=11)
    plt.tight_layout()
    #plt.savefig(outdir / "overlay_hist.png", dpi=150)
    plt.savefig(outdir / "overlay_hist_ft.png", dpi=150)
    plt.close()

    # ------------------- write CSV -----------------------------
    #pd.DataFrame(stats_rows).to_csv(outdir / "token_stats.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(outdir / "token_stats_ft.csv", index=False)
    print(f"✓ Done. Plots and CSV saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
