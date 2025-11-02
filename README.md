# LMRSD (Learning meaningful rewards on Scientific Documents)


### Core motivation
- Using Language models in scholarly peer review seems comes with significant risks surrounding safety, research integrity and validity of the review.
- Inevitably people utilize LLMs as pre-review agents if not fully autonomous peer-review agents.
- Lack of a systematic evaluation of LLMs generating reviews across science disciplines misses the mark on and assessing the alignment/misalignment question.

### Problem formulation
- Given a Paper P, field F, and peer-review R, a traditional learning framework would capture the decision function θ(R^ | P, F) through a training objective minimizing R by Mean Absolute Error.
	- Assumption 1: Representations from different pre-trained models capturing crucial information of P and F act as features to train the model θ.
	- Assumption 2: Peer-review R includes both a sequence of tokens Rtext [r1, r2, r3, ….rn] and a discrete value Rscore representing the score gauging the evaluation of the idea/manuscript on a scale of 1-10. 
- Utilizing large language models(LLMs) can provide a training-free framework to understand peer-review Rscore and assess the alignment/mis-alignment of LLMs over the real-world outcome such as hit-paper status in field F.
- Systematically assessing alignment of LLMs Rscore would help us gauge the safety risks involved with deploying large language models as agents for pre-review to help reviewers with peer-review.


### Research Agenda
- RQ-1: Understanding the joint distribution of idea review scores and paper review scores for a collection of language models.
- RQ-2: Apart from the accuracy, observe the alignment and misalignment of each model to observe which agrees/disagrees the most with the human label.
- RQ-3: Assessing reviews where humans/LLMs can gauge hit-paper 1%, 5%, and 10% outcomes.

### Data
<img src="./data/media/review_joint_distribution.png" width=500 height=400>

More about the data can be found [here](./data/README.md).
> `NOTE`: The datasets are available as parquet files on Google drive, and they can be found [here](https://drive.google.com/drive/folders/1nAPX7PFgCbhGVaHMhkMqbak9dCg4WxfL?usp=sharing).

### Repository structure

```shell
├── LICENSE
├── README.md
├── data
│   ├── README.md
│   ├── __init__.py
│   └── media
│       ├── review_idea_distribution.png
│       ├── review_joint_distribution.png
│       └── review_paper_distribution.png
└── src
    ├── __init__.py
    ├── icl.py
    ├── prompts.py
    └── schema.py
```

### Environment setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or via **pip**
```shell
pip install uv
```

Have the following packages installed to run `LMRSD`
```shell
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install bitsandbytes
uv pip install git+https://github.com/huggingface/transformers
uv pip install deepspeed
uv pip install sentencepiece
uv pip install vllm tiktoken outlines trl openai polars peft tqdm pydantic google-genai matplotlib scikit-learn ninja bs4
```

### Acknowledgement
Thanks to `@sumuks` and the huggingface repo **[sumuks/openreview-reviews-filtered](https://huggingface.co/datasets/sumuks/openreview-reviews-filtered)** which were crucial for the dataset, experiments, and meethodology of the paper.

### License
[MIT License](https://github.com/akhilpandey95/LMRSD/blob/main/LICENSE)

### Authors and Collaborators
[Akhil Pandey](https://github.com/akhilpandey95)
