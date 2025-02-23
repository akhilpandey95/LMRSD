# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/LMRSD/blob/main/LICENSE.

import gc
import time
import torch
import tiktoken
import polars as pl
import seaborn as sns
import jsonlines as jl
from pprint import pprint
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

# read the ICLR dataset
df_open_reviews = pl.scan_parquet("../data/*.parquet")

# collect the data
df_open_reviews = df_open_reviews.collect()

# essential stats
print(f"Shape of df_open_reviews: {df_open_reviews.shape}")
print(f"Estimated memory pressure for the model: {df_open_reviews.estimated_size("gb")} GB")

# sneak peak
print(df_open_reviews.head())

# helper function to give token pass flag
def token_fits_128K(paper_content, return_length=False):
    """
    Boolean fn flag to check if the paper
    content fits 128K context length

    Parameters
    ------------
    arg1 | paper_content: str
        Full text of the paper
    arg2 | paper_content: bool [Optional]
        Boolean flag to return token count. Default set to `False`

    Returns
    ------------
        Boolean/Integer
    """
    # init tiktoken object
    tiktoken_enc = tiktoken.encoding_for_model("gpt-4o")

    # store token length
    tokens = len(tiktoken_enc.encode(paper_content))

    # token pass check ?
    if tokens < 128000 and return_length:
        return tokens
    else:
        return tokens < 128000

# helper function to generate review
def generate_review(model_name, input_text, max_tokens, device):
    """
    Use ICL to generate a simple review
    of a given scientific paper

    Parameters
    ------------
    arg1 | model_name: str
        Use model catalog to load local model weights
    arg2 | input_text: str
        Input text for the model with paper content and ICL instruction
    arg3 | max_tokens: int
        Maximum tokens to generate for writing a review
    arg4 | device: str
        Hardware acceleration, defaults to "cpu" if any errors arise

    Returns
    ------------
        Text
            str
    """
    # device for acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # null outpput
    output = None

    # set the model-id
    model_catalog = {
        "llama3.3": "/projects/p32534/llama3.3/Llama3.3-70B-Instruct/hf",
        "llama3.2-1b": "/projects/p32534/llama3.2/Llama3.2-1B-Instruct/hf",
        "llama3.2-3b": "/projects/p32534/llama3.2/Llama3.2-3B-Instruct/hf",
        "llama3.1-70b": "/projects/p32534/llama3.1/Meta-Llama-3.1-70B-Instruct/hf",
        "llama3.1-8b": "/projects/p32534/llama3.1/Meta-Llama-3.1-8B-Instruct/hf",
    }
    
    # set a model-id
    model_id = model_catalog[model_name]

    # log
    print("----------------------------------")
    print(f"Using {device} to load {model_id}")
    print("----------------------------------")
    
    # get model-tokenizer pair
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # based on model size switch quantization config
    if model_name == "llama3.1-70b":
        # 4-bit quantization config
        bnb_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16
        )

        # 4 bit quantization
        model = AutoModelForCausalLM.from_pretrained(model_id, \
                                                     quantization_config=bnb_4bit, \
                                                     trust_remote_code=True, \
                                                     device_map=device)
    else:
        # load bfloat16 weights
        model = AutoModelForCausalLM.from_pretrained(model_id, \
                                                     trust_remote_code=True, \
                                                     torch_dtype=torch.bfloat16, \
                                                     device_map=device)
    
    # is it a llama tokenizer ?
    if "llama" in model_name:
        # pad token if needed
        tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
        print(f"Setting <|finetune_right_pad_id|> token for {model_id}")
        model.resize_token_embeddings(len(tokenizer))
    
    # load time
    end = time.time()
    print("Model-tokenizer Load Time:", end - start)
    print("----------------------------------")
    
    # seed for reproducibility
    set_seed(2025)
    
    # set top_p and temperature to none
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    
    # tokenomics
    print(f"Total tokens in input_text: {token_fits_128K(input_text, return_length=True)} tokens")
    print("----------------------------------")
    
    # get attention mask and input ids
    input_encoded = tokenizer(input_text, padding=True, return_tensors="pt")
    input_encoded_ids = input_encoded.input_ids.to(device)
    input_encoded_attn_mask = input_encoded.attention_mask.to(device)
    input_shape = input_encoded_ids.shape[1]
    
    # shape check
    print(f"Encoded inputs: {input_shape}")
    print(f"Truncated tokens because of context window issues: {token_fits_128K(input_text, return_length=True) - input_shape}")
    print("----------------------------------")
    
    # model.generate()
    with torch.no_grad():
        # time check
        start = time.time()
        
        # routine model.generate()
        outputs = model.generate(
            input_ids=input_encoded_ids,
            attention_mask=input_encoded_attn_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # inference time
        end = time.time()
        inference = end - start
        
        # model.generate() output
        output = tokenizer.decode(outputs[0][input_shape:], skip_special_tokens=True)

        # empty cuda cache
        torch.cuda.empty_cache()

        # gc
        del model, tokenizer
        gc.collect()

    # return the output
    return output

# prompt to generate open review style review
input_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful intelligent peer review assistant. You have the ability to take scientific papers and write reviews

<|eot_id|><|start_header_id|>user<|end_header_id|>
SCIENTIFIC_PAPER_TITLE:
```
PAPER_TITLE_CONTENT
```

SCIENTIFIC_PAPER_FULL_TEXT:
```
PAPER_FULL_TEXT_CONTENT
```

Write a peer review for the above scientific work
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# str replace
input_text = input_text.replace("PAPER_TITLE_CONTENT", df_open_reviews.select("paper_title")[0].item())
input_text = input_text.replace("PAPER_FULL_TEXT_CONTENT", df_open_reviews.select("paper_content")[0].item())

# generate a peer review
output = generate_review("llama3.1-70b", input_text, max_tokens=1024, device="cuda")
print("Peer Review:")
print(output)
print("----------------------------------")

# tps logic
new_tokens = outputs[0].shape[0] - input_shape
tokens_per_second = new_tokens / inference if inference > 0 else float('inf')
print(f"Inference time: {inference:.2f} seconds")
print(f"New tokens generated: {new_tokens}")
print(f"Tokens per second: {tokens_per_second:.2f}")
