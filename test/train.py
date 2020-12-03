# from transformers import GPT2LMHeadModel
from transformers.configuration_gpt2 import GPT2Config
from mmdialoggpt import MMdialogGPT
from transformers import GPT2Model
import torch

configuration = GPT2Config()
print(configuration)
model = MMdialogGPT(configuration)
model2 = GPT2Model(configuration)

# input_emb = config.n_emb = 768
input = torch.rand(3, 8, 768)
mask = torch.ones(3,8)
mask[0][7] = 0
print(mask)
output = model(input, attention_mask=mask)
output1 = model2(inputs_embeds=input, attention_mask=mask)
print(output1[0])
print(output[0])