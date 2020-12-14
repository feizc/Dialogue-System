from transformers import * 
import torch

from train import SPECIAL_TOKENS_DICT
from MMdialog import MMdialog 

device = torch.device("cuda" if use_cuda else "cpu")


# greedy search 


# beam search 

# top-k sampling 


# response generaton 


if __name__ == "__main__":

    ckpt_path = 'ckpt/mmgpt/model.bin'
    tokenizer = BertTokenizer.from_pretrained('ckpt/mmgpt', do_lower_case=True)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model_config = GPT2Config.from_pretrained('ckpt/mmgpt')

    model = MMdialog(model_config)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model']) 
    model = model.to(device)
    model.eval()







