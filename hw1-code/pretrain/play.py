# %%
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# %%
model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

for k,v in sd_hf.items():
    print(k, v.shape)


