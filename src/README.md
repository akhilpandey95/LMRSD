### Experiments
- Exp 1: idea generation with just abstract | Results across models that fit the context window.
- Exp 2: idea generation and paper review with full-text | Results across models that fit the context window.
- Exp 3: Create Post publications outcomes dataset and statistical comparisons with full-text reviews.
- Ablation 1: Comparing the post-publication outcomes with reasoning models
- Ablation 2: Comparing the post-publication outcomes with private models

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

### Run `LMRSD` experiment 1

**type="idea"**
```shell
python lmrsd_exp1.py --data-file lmrsd_abs_evaluation.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_idea --content-choice idea --backend 0 --max-concurrent 32
```

or

```shell
nohup python -u lmrsd_exp1.py --data-file lmrsd_abs_evaluation.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_idea --content-choice idea --backend 0 --max-concurrent 32 > lmrsd_exp1_qwen3_32b_idea.log 2>&1 &
```

### Run `LMRSD` experiment 2

**type="full_text"**
```shell
python lmrsd_exp1.py --data-file lmrsd_ft_evaluation.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_full_text --content-choice full_text --backend 0 --max-concurrent 32
```

or

```shell
nohup python -u lmrsd_exp1.py --data-file lmrsd_ft_evaluation.parquet --llm qwen3-32b --opfile lmrsd_exp1_qwen3_32b_full_text --content-choice full_text --backend 0 --max-concurrent 32 > lmrsd_exp1_qwen3_32b_full_text.log 2>&1 &
```

### Run `LMRSD` experiment 3
```shell
TBA
```