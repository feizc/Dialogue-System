from transformers import * 
import os
import sys 
import torch  
from torch.utils.data import DataLoader 
import numpy 
import json 

from MMdialog import LabelPredict 
from dataset import LPDataset 
from utils import accuracy_compute, AverageMeter 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}



# Data parameters
train_data_path = 'data/label_valid.json' 
val_data_path = 'data/label_valid.json'

model_path = 'ckpt/label_predict' 
lr = 4e-5
ckpt_path = 'ckpt/label_predict/model.bin' 
use_cuda = torch.cuda.is_available() 
device = torch.device('cuda' if use_cuda else 'cpu') 

checkpoint_usage = True 
gradient_accumulation_steps = 5 
epochs = 40 
print_freq = 100



def main(): 
    if checkpoint_usage == True: 
        tokenizer = BertTokenizer.from_pretrained('ckpt/label_predict', do_lower_case=True)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    
        model_config = GPT2Config.from_pretrained('ckpt/label_predict') 
        model = LabelPredict(model_config) 
        ckpt = torch.load(ckpt_path, map_location='cpu') 
        model.load_state_dict(ckpt['model'])
        # model.resize_token_embeddings(len(tokenizer)) 
    else:
        tokenizer = BertTokenizer.from_pretrained('ckpt/cdial-gpt', do_lower_case=True) 
        model = LabelPredict('ckpt/cdial-gpt')
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
        model.resize_token_embeddings(len(tokenizer)) 

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr) 

    train_dataset = LPDataset(train_data_path, tokenizer) 
    train_loader = DataLoader(train_dataset, batch_size=5, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)) 
    val_dataset = LPDataset(val_data_path, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=1) 

    for epoch in range(epochs):
        train(model=model, optimizer=optimizer, dataset=train_loader, epoch=epoch)
        #print(his.size(), respond) 
        break

    return 
    torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},\
                '%s/model.bin'%(model_path))
    model.config.to_json_file(os.path.join(model_path, 'config.json'))
    tokenizer.save_vocabulary(model_path)


def train(model, optimizer, dataset, epoch):
    model.train() 

    avg_loss = AverageMeter()
    avg_acc = AverageMeter() 

    iteration = 1 

    for instance in dataset:
        his, respond = instance 
        his, respond = his.to(device), respond.to(device) 

        loss, logits = model(his, respond)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        if iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        acc = accuracy_compute(logits, respond, 5)
        avg_loss.update(loss.item()) 
        avg_acc.update(acc)
        # print(acc)
        # print(loss.item())
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc))
        
        # print(logits.size())

        iteration += 1 
        break 







def collate_fn(batch, pad_token):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq]) 
        if seq[0].size(0) == 1: 
            result = torch.ones(len(seq)) * pad_token 
            result = result.long() 
            # print(result, seq)
            for i in range(len(seq)):
                result[i] = seq[i]
        else:
            result = torch.ones((len(seq), max_len)) * pad_token 
            result = result.long()
            for i in range(len(seq)):
                result[i, :seq[i].size(0)] = seq[i]
        return result 

    his_ids_list, respond_list = [], []
    for i in batch: 
        his_ids_list.append(i[0])
        respond_list.append(i[1])
    his_ids_list = padding(his_ids_list, pad_token) 
    respond_list = padding(respond_list, pad_token)
    return his_ids_list, respond_list 



if __name__ == "__main__":
    main()



