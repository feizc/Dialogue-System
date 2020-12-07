from transformers import GPT2LMHeadModel, BertTokenizer
import torch 
import torch.nn.functional as F 

tokenizer = BertTokenizer.from_pretrained('./ckpt/cdial-gpt', do_lower_case=True)
model = GPT2LMHeadModel.from_pretrained('./ckpt/cdial-gpt')


input = ['[CLS]你好，你叫什么名字[SEP]']
history = tokenizer.tokenize(input[0])
history = tokenizer.convert_tokens_to_ids(history)
history = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input[0]))
print(history)
history = torch.LongTensor(history)
output, *_ = model(history)
probs = F.softmax(output, dim=-1)
output = torch.topk(probs, 1)[1].squeeze(1)
print(output)
output = tokenizer.decode(output)
print(output)

