# LMRSD (Learning meaningful rewards on Scientific Documents)

### About
LM's are execptional at generating texts across various domains.


The core research question we are trying to investigate surrounds the
topic of credit assignment problem for qualitative texts. More specifically,
if there exists a scientific task, <i>Creating reward signals for LLM
reasoning beyond math/programming domains is **hard**</i> is a well agreed
upon notion.

### Data
<img src="./data/media/review_joint_distribution.png" width=400 height=300>

More about the data can be found [here](./data/README.md).
> `NOTE`: The datasets are available as parquet files and can be found [here](https://drive.google.com/drive/folders/1nAPX7PFgCbhGVaHMhkMqbak9dCg4WxfL?usp=sharing).

### Repository structure

```shell
├── LICENSE
├── README.md
├── data
│   ├── README.md
│   ├── __init__.py
│   ├── data_raw.parquet
│   └── media
│       ├── review_idea_distribution.png
│       ├── review_joint_distribution.png
│       └── review_paper_distribution.png
└── src
    ├── __init__.py
    └── icl.py
```

### Environment setup

```shell
TBA
```

### Acknowledgement
Thanks to `@sumuks` and the huggingface repo **[sumuks/openreview-reviews-filtered](https://huggingface.co/datasets/sumuks/openreview-reviews-filtered)** which were crucial for the dataset, experiments, and meethodology of the paper.

### License
[MIT License](https://github.com/akhilpandey95/LMRSD/blob/main/LICENSE)

### Authors and Collaborators
[Akhil Pandey](https://github.com/akhilpandey95)
