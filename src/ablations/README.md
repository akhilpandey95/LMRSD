### Ablations for `LMRSD`
The core research ablations for `LMRSD` uses the post-publication outcomes dataset to answer the following:
- ABL 1 : How do the frontier of models reflect on the pre-training corpus when it comes to openreviews data ?
- ABL 2 : How similar/dissimilar are the reviews written by top language models over human reviewers ?
- ABL 3 : Does reasoning strength combined with adjusted prompt considerations deviate the peer-review scores of reasoning models ?

### Post-publications dataset for ablations
```plaintext
File: lmrsd_postpub_outcomes.parquet
Number of rows/papers: 1818
Number of columns: 20
Columns: ['paper_id', 'year', 'paper_title', \
          'paperid', 'fieldid', 'year_right', \
          'citation_count', 'Hit_1pct', 'Hit_5pct', 'Hit_10pct', \
          'median_ft_score', 'avg_ft_score', 'median_ft_score_cf', 'avg_ft_score_cf', \
          'type', 'paper_keywords', 'paper_abstract', 'paper_content', 'tkn_count', 'review_rating_description']
```

### Run ablations
**ablation 1**
> How do the frontier of models reflect on the pre-training corpus when it comes to openreviews data ?
> For LMRSD.c dataset, adjust the script to include calc_perplexity(), and include log_probs.
```shell
python lmrsd_abl1.py --dataset /projects/p32534/code/LMRSD/data/lmrsd_postpub_outcomes.parquet --llm qwen3-32b --opfile abl1_qwen3 --bs 2 --tensor-parallel-size 4
```

**ablation 2**
> How similar/dissimilar are the reviews written by top language models over human reviewers ?
> KL divergence + Jaccard + Cos of (human_review, LLM_review)
```shell
python lmrsd_abl2.py
```

**Relevant Columns**
```plaintext
------------Column Meta data------------
`paper_id`           - OpenReview randomly generated paperid
`paperid`            - OpenAlex paperid
`paper_content`      - Full text of the article
`fieldid`            - OpenAlex ConceptID/MAG Field Level-0 fieldID.
`Hit_1pct`           - Binary variable suggesting if a paper is in the top 1% of the cited papers per that respective Field.
`Hit_5pct`           - Binary variable suggesting if a paper is in the top 5% of the cited papers per that respective Field.
`Hit_10pct`          - Binary variable suggesting if a paper is in the top 10% of the cited papers per that respective Field.
`avg_ft_score`       - Average of all Review scores for a given paper.
`avg_ft_score_cf`    - Average of all Reviewer confidence scores for a given paper.
`median_ft_score`    - Median of all Review scores for a given paper.
`median_ft_score_cf` - Median of all Reviewer confidence scores for a given paper.
`tkn_count`          - Total # of tokens in the the input (system prompt + paper_content) when passed through a Qwen3 family tokenizer.
`review_rating_description` - List of reviews written by human reviews for a single paper.
```
