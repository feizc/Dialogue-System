from transformers import GPT2Config, BertTokenizer
import os
import sys
from model import MMdialog 
import torch 


if __name__ == "__main__":
    
    config = GPT2Config(vocab_size=13088)
    print(config)
    tokenizer = BertTokenizer.from_pretrained('./huggingface_ckpt/cdial-gpt', do_lower_case=True)
    input = '你好啊'
    input = tokenizer.tokenize(input)
    input = tokenizer.convert_tokens_to_ids(input)
    input = torch.LongTensor(input)
    # print(input)

    mmdialog = MMdialog(config)
    emb = mmdialog.transformer.wte(input)
    print(emb.shape)
    input = torch.rand(7,768)
    input = torch.cat((emb, input), 0)

    print(mmdialog(input).shape)
