### Research Agenda
- Exp 1: Assess and review the ideas presented in a scientific article just using the abstract.
- Exp 2: Assess and review the ideas and fully review the scientific article using the full-text.
- Exp 3: Create Post publications outcomes dataset and statistical comparisons with full-text reviews.
- Exp 4: Comparing the post-publication outcomes with open weight reasoning models with high-degree of effort.
- Exp 5: Comparing the post-publication outcomes with SOTA private models.
- Ablation 1: Effect of instructions in Dense over alignment of review scores.
- Ablation 2: Effect of reasoning strength in arguing merits of peer review for reasoning models.

### Repository structure
```shell
├── __init__.py
├── ablations
│   ├── gemma3_LMRSD_baseline.py
│   ├── icl.py
│   └── llms_LMRSD_baseline.py
├── exp1
│   └── lmrsd_exp1.py
├── plots
│   └── tokenomics_analysis.py
├── README.md
└── util
    ├── prompts.py
    └── schema.py
```

### Inference engine Initialization
**vLLM**
```shell
nohup /projects/p32534/code/LMRSD/qwen3_vllm.sh > /projects/p32534/code/LMRSD/qwen3_vllm_LMRSD.log 2>&1 &
nohup /projects/p32534/code/LMRSD/llama33.sh > /projects/p32534/code/LMRSD/llama33_LMRSD.log 2>&1 &
nohup /projects/p32534/code/LMRSD/tulu3.sh > /projects/p32534/code/LMRSD/tulu3_LMRSD.log 2>&1 &
nohup /projects/p32534/code/LMRSD/gpt20b.sh > /projects/p32534/code/LMRSD/gpt20b_LMRSD.log 2>&1 &
nohup /projects/p32534/code/LMRSD/gpt120b.sh > /projects/p32534/code/LMRSD/gpt120b_LMRSD.log 2>&1 &
nohup /projects/p32534/code/LMRSD/gemma3_27b_vllm.sh > /projects/p32534/code/LMRSD/gemma3_27b_vllm_LMRSD.log 2>&1 &
```

**sglang**
```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server --model /kellogg/proj/dashun/LLM/HuggingFaceCache/gemma-3-27b-it --tp 4 --mem-fraction-static 0.85 --context-length 36864 --max-running-requests 256 --max-total-tokens 131072 > sglang_gemma3_LMRSD.log 2>&1 &
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server --model /kellogg/proj/dashun/LLM/HuggingFaceCache/Llama-3.3-70B-Instruct/ --tp 4 --mem-fraction-static 0.95 --context-length 36864 --max-running-requests 256 --max-total-tokens 131072 > sglang_llama3_LMRSD.log 2>&1 &
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server --model /projects/p32534/code/hypeline/models/Llama-3.1-Tulu-3-70B --tp 4 --mem-fraction-static 0.95 --context-length 36864 --max-running-requests 256 --max-total-tokens 131072 > sglang_tulu3_LMRSD.log 2>&1 &
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server --model /projects/p32534/code/hypeline/models/gpt-oss-20b-bf16 --tp 4 --mem-fraction-static 0.85 > /projects/p32534/code/LMRSD/sglang_gpt20b_LMRSD.log 2>&1 &
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server --model /projects/p32534/code/hypeline/models/gpt-oss-20b --tp 4 --mem-fraction-static 0.85 > /projects/p32534/code/LMRSD/sglang_gpt20b_LMRSD.log 2>&1 &
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server --model /projects/p32534/code/hypeline/models/gpt-oss-120b-bf16 --tp 4 --mem-fraction-static 0.85 > /projects/p32534/code/LMRSD/sglang_gpt120b_LMRSD.log 2>&1 &
```

> NOTE: If you pre-calculate the overall token counts for the entire dataset
> it is easier to set the below parameters ensuring maximum throughput.
**sglang dense models (optimized)**
```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server \
  --model /projects/p32534/code/hypeline/models/Llama-3.1-Tulu-3-70B \
  --tp 4 \
  --mem-fraction-static 0.90 \
  --context-length 32768 \
  --max-running-requests 64 \
  --max-total-tokens 524288 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 65536 \
  --kv-cache-dtype fp8_e5m2 \
  --schedule-policy lpm \
  --enable-torch-compile \
  > sglang_tulu3_LMRSD.log 2>&1 &
```

```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server \
  --model /projects/p32534/code/hypeline/models/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --tp 4 \
  --mem-fraction-static 0.90 \
  --context-length 32768 \
  --max-running-requests 64 \
  --max-total-tokens 800000 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 131072 \
  --kv-cache-dtype fp8_e5m2 \
  --schedule-policy lpm \
  --enable-torch-compile \
  --trust-remote-code \
  > sglang_nemotron_LMRSD.log 2>&1 &
```

**Llama-4-Scout-17B-16E-Instruct**
```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server \
  --model /projects/p32494/ai4sciscibench/models/Llama-4-Scout-17B-16E-Instruct \
  --tp 4 \
  --mem-fraction-static 0.90 \
  --context-length 32768 \
  --max-running-requests 64 \
  --max-total-tokens 800000 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 131072 \
  --kv-cache-dtype fp8_e5m2 \
  --schedule-policy lpm \
  > sglang_llama4_scout_17B_16E_LMRSD.log 2>&1 &
```

**reasoning models on sglang (optimized):**

**R1-Distill-llama-70b**
```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server \
  --model /kellogg/proj/dashun/LLM/HuggingFaceCache/DeepSeek-R1-Distill-Llama-70B \
  --tp 4 \
  --mem-fraction-static 0.90 \
  --context-length 32768 \
  --max-running-requests 64 \
  --max-total-tokens 524288 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 65536 \
  --kv-cache-dtype fp8_e5m2 \
  --schedule-policy lpm \
  --enable-torch-compile \
  > sglang_r1_llama70b_LMRSD.log 2>&1 &
```

**R1-Distill-qwen-32b**
```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server \
  --model /kellogg/proj/dashun/LLM/HuggingFaceCache/DeepSeek-R1-Distill-Qwen-32B \
  --tp 4 \
  --mem-fraction-static 0.90 \
  --context-length 32768 \
  --max-running-requests 64 \
  --max-total-tokens 800000 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 131072 \
  --kv-cache-dtype fp8_e5m2 \
  --schedule-policy lpm \
  --enable-torch-compile \
  > sglang_r1_qwen32b_LMRSD.log 2>&1 &
```

**Qwen3-Next-80B-A3B-Thinking (optimized)**
```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server \
  --model /projects/p32494/ai4sciscibench/models/Qwen3-Next-80B-A3B-Thinking \
  --tp 4 \
  --mem-fraction-static 0.90 \
  --reasoning-parser deepseek-r1 \
  --context-length 128000 \
  --max-running-requests 64 \
  --max-total-tokens 800000 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 131072 \
  --kv-cache-dtype fp8_e5m2 \
  --schedule-policy lpm \
  --enable-torch-compile \
  > sglang_qwen3_next_80B_A3b_LMRSD.log 2>&1 &
```

**GPT-OSS 120b (optimized)**
```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server \
  --model /projects/p32534/code/hypeline/models/gpt-oss-120b \
  --tp 4 \
  --mem-fraction-static 0.85 \
  --reasoning-parser gpt-oss \
  --max-running-requests 32 \
  --max-total-tokens 800000 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 131072 \
  --schedule-policy lpm \
  > sglang_gpt120b_LMRSD.log 2>&1 &
```

**Choice of inference engine**
```plaintext
- Qwen3-32b (vLLM)
- Gemma3-27b (sglang)
- GPT-OSS-20b (vLLM/sglang)
- GPT-OSS-120b (vLLM/sglang)
```

### Run `LMRSD` experiment 1

**Exp 1: idea generation with just abstract | type="idea"**
```shell
python lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_idea --content-choice idea --backend 0 --max-concurrent 32
```

or run in the background

```shell
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_idea --content-choice idea --backend 0 --max-concurrent 32 > lmrsd_exp1_qwen3_32b_idea.log 2>&1 &
```

**To run the actual experiment flow, please run the following**
```shell
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_idea --content-choice idea --backend 0 --max-concurrent 32 > lmrsd_exp1_qwen3_32b_idea.log 2>&1 &
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm llama-33 --opfile lmrsd_exp1_llama_33_70b_idea --content-choice idea --backend 0 --max-concurrent 32 > lmrsd_exp1_llama_33_70b_idea.log 2>&1 &

nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm tulu3-70b --opfile lmrsd_exp1_tulu3_70b_idea --content-choice idea --backend 0 --max-concurrent 32 > lmrsd_exp1_tulu3_70b_idea.log 2>&1 &

nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm nemotron-49b --opfile lmrsd_exp1_nemotron_49b_idea --content-choice idea --backend 1 --max-concurrent 32 > lmrsd_exp1_nemotron_49b_idea.log 2>&1 &

nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm gptoss-20b --opfile lmrsd_exp1_gptoss_20b_idea --content-choice idea --backend 0 --max-concurrent 32 > lmrsd_exp1_gptoss_20b_idea.log 2>&1 &
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm gptoss-120b --opfile lmrsd_exp1_gptoss_120b_idea --content-choice idea --backend 1 --max-concurrent 32 > lmrsd_exp1_gptoss_120b_idea.log 2>&1 &
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_abs_evaluation.parquet --llm gemma-3-27b --opfile lmrsd_exp1_gemma3_27b_idea --content-choice idea --backend 1 --max-concurrent 32 > lmrsd_exp1_gemma3_27b_idea.log 2>&1 &
```

### Run `LMRSD` experiment 2

**Exp 2: idea review and paper review with full-text | type="full_text"**
```shell
/projects/p32534/mlx/bin/python lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_full_text --content-choice full_text --backend 0 --max-concurrent 32
```

or run in the background
```shell
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_full_text --content-choice full_text --backend 0 --max-concurrent 13 > lmrsd_exp1_qwen3_32b_full_text.log 2>&1 &
```

**To run the actual experiment flow for part A, please run the following**
```shell
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_full_text --content-choice full_text --backend 0 --max-concurrent 13 > lmrsd_exp1_qwen3_32b_full_text.log 2>&1 &
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm llama-33 --opfile lmrsd_exp1_llama_33_70b_full_text --content-choice full_text --backend 0 --max-concurrent 13 > lmrsd_exp1_llama_33_70b_full_text.log 2>&1 &
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm tulu3-70b --opfile lmrsd_exp1_tulu3_70b_full_text --content-choice full_text --backend 1 --max-concurrent 13 > lmrsd_exp1_tulu3_70b_full_text.log 2>&1 &
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm nemotron-49b --opfile lmrsd_exp1_nemotron_49b_full_text --content-choice full_text --backend 1 --max-concurrent 48 > lmrsd_exp1_nemotron_49b_full_text.log 2>&1 &
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm gemma-3-27b --opfile lmrsd_exp1_gemma3_27b_full_text --content-choice full_text --backend 1 --max-concurrent 13 > lmrsd_exp1_gemma3_27b_full_text.log 2>&1 &
```

**To run the experiment flow for part B**
```shell
nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm r1-llama --opfile lmrsd_exp1_r1_llama70b_full_text --content-choice full_text --backend 1 --max-concurrent 32 > lmrsd_exp1_r1_llama70b_full_text.log 2>&1 &

nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm r1-qwen --opfile lmrsd_exp1_r1_qwen32b_full_text --content-choice full_text --backend 1 --max-concurrent 54 > lmrsd_exp1_r1_qwen32b_full_text.log 2>&1 &

nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm qwen3-moe --opfile lmrsd_exp1_qwen3_80b_A3b_full_text --content-choice full_text --backend 1 --max-concurrent 32 > lmrsd_exp1_qwen3_80b_A3b_full_text.log 2>&1 &

nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm gptoss-20b --opfile lmrsd_exp1_gptoss_20b_full_text --content-choice full_text --backend 1 --max-concurrent 13 > lmrsd_exp1_gptoss_20b_full_text.log 2>&1 &

nohup /projects/p32534/mlx/bin/python -u lmrsd_exp1_exp2.py --data-file lmrsd_ft_evaluation_26k.parquet --llm gptoss-120b --opfile lmrsd_exp1_gptoss_120b_full_text --content-choice full_text --backend 1 --max-concurrent 13 > lmrsd_exp1_gptoss_120b_full_text.log 2>&1 &
```

> NOTE: When you run a vLLM/sgLang server for experiment 2 on TP:4xH100 or similar setups, memory leaks, fragmentation errors
> and multiprocessing errors can occur causing the inference engine to crash thereby increasing the time to complete the actual
> ZS inference script. It is advised to monitor the progress and periodically restart the inference engine to mitigate accumulation
> of the above mentioned issues.

### Run `LMRSD` experiment 3
```python
# three core groups
adversarial = pl.concat([
    # high citations, not hits
    df_lmrsd_post_pub.filter(
        (pl.col('citation_count') > df_lmrsd_post_pub['citation_count'].quantile(0.95)) & 
        (pl.col('Hit_1pct') == 0)
    ).unique('paper_id').with_columns(pl.lit('false_positive').alias('type')),
    
    # hit papers, low citations  
    df_lmrsd_post_pub.filter(
        (pl.col('Hit_1pct') == 0) & (pl.col('Hit_5pct') == 1) &
        (pl.col('citation_count') < df_lmrsd_post_pub['citation_count'].mean())
    ).unique('paper_id').with_columns(pl.lit('false_negative').alias('type')),
    
    # clear hits
    df_lmrsd_post_pub.filter(
        (pl.col('Hit_1pct') == 1) & 
        (pl.col('citation_count') > df_lmrsd_post_pub['citation_count'].quantile(0.95))
    ).unique('paper_id').with_columns(pl.lit('true_positive').alias('type'))
])
```

### Run `LMRSD` experiment 4
```shell
nohup /projects/p32534/mlx/bin/python -u lmrsd_post_pub.py --data-file lmrsd_postpub_outcomes.parquet --llm gptoss-120b --opfile lmrsd_exp4_postpub_gptoss_120b --content-choice post_pub --backend 1 --max-concurrent 32 > lmrsd_exp4_postpub_gptoss_120b.log 2>&1 &

nohup /projects/p32534/mlx/bin/python -u lmrsd_post_pub.py --data-file lmrsd_postpub_outcomes.parquet --llm qwen3-moe --opfile lmrsd_exp4_postpub_qwen3_80b_A3b --content-choice post_pub --backend 1 --max-concurrent 32 > lmrsd_exp4_postpub_qwen3_80b_A3b.log 2>&1 &
```

```shell
nohup /projects/p32534/sglang/bin/python -m sglang.launch_server \
  --model /kellogg/proj/dashun/LLM/HuggingFaceCache/Qwen3-32B \
  --tp 4 \
  --reasoning-parser qwen3 \
  --mem-fraction-static 0.88 \
  --context-length 32768 \
  --max-running-requests 64 \
  --max-total-tokens 800000 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 131072 \
  --kv-cache-dtype fp8_e5m2 \
  --schedule-policy lpm \
  --enable-torch-compile \
  > sglang_qwen3_32b_qgpu3024.log 2>&1 &
```

```shell
nohup python result_extraction_simple.py \
  --data-file resextract_bioarxiv_26k.parquet \
  --llm qwen3-32b \
  --opfile test_simple \
  --max-concurrent 64 \
  --backend 1 > logs/resextract_bioarxiv_26k_rn1.log 2>&1 &
```