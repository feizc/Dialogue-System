from transformers import * 
import os 
import sys 
import torch 
import json 
import numpy as np 
from torch.utils.data import DataLoader 

from dataset import VEIDDataset 
from MMdialog import VEID 
from utils import accuracy_compute, AverageMeter 

SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}



use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
train_data_path = 'data/small_train.json'
val_data_path = 'data/small_valid.json'


checkpoint_usage = False 
epochs = 20 
lr = 5e-5
gradient_accumulation_steps = 5 
print_freq = 1


def main():
       # 模型初始化 
    if checkpoint_usage == True: 
        ckpt_path = 'ckpt/mmgpt/model.bin'
        tokenizer = BertTokenizer.from_pretrained('ckpt/mmgpt', do_lower_case=True)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        model_config = GPT2Config.from_pretrained('ckpt/mmgpt')

        model = VEID(model_config)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model']) 
    
    else:
        tokenizer = BertTokenizer.from_pretrained('ckpt/cdial-gpt', do_lower_case=True)
        model = VEID.from_pretrained('ckpt/cdial-gpt')
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # 数据读取
    with open('data/id2feature_clip.json', 'r', encoding='utf-8') as f: 
        id2feature = json.load(f) 
    img_bank = torch.from_numpy(np.array(img_bank_construct(id2feature))).float().to(device) 
    img_bank = img_bank / torch.norm(img_bank, dim=1).unsqueeze(1)
    print(img_bank.size())
    
    train_dataset = VEIDDataset(train_data_path, tokenizer, id2feature) 
    #val_dataset = VEIDDataset(val_data_path, tokenizer, id2feature) 

    train_loader = DataLoader(train_dataset, batch_size=1)
    #val_loader = DataLoader(val_dataset, batch_size=1)

    ckpt_path = 'ckpt/VEID'
    for epoch in range(epochs): 
        acc = train(model=model, tokenizer=tokenizer, optimizer=optimizer, dataset=train_loader, epoch=epoch, img_bank=img_bank) 
        torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},\
                    '%s/epoch_%d_acc_%.3f'%(ckpt_path, epoch, acc))
        model.config.to_json_file(os.path.join(ckpt_path, 'config.json'))
        tokenizer.save_vocabulary(ckpt_path) 
        break 




# training process implementation 
def train(model, tokenizer, optimizer, dataset, epoch, img_bank):

    # 模型的训练
    model.train()

    total_avg_loss = AverageMeter()
    text_avg_loss = AverageMeter()
    img_avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    iteration = 1

    for instance in dataset: 
        input_ids, target_ids, token_type_ids, feature, img_id = instance 
        if token_type_ids.size(0) > 450:
            continue
        input_ids, target_ids, token_type_ids, feature, img_id = input_ids.to(device), target_ids.to(device), token_type_ids.to(device), feature.to(device), img_id.to(device) 
        #print(feature.size())
        loss,text_loss, img_loss, fenmu = model(input_ids, token_type_ids, target_ids, feature, img_bank)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        #print(img_id)
        _, idx = torch.topk(fenmu, k=60)
        acc = idx.eq(img_id.view(-1).expand_as(idx)).float().sum().item()
        avg_acc.update(acc)
        total_avg_loss.update(loss.item())
        text_avg_loss.update(text_loss.item())
        img_avg_loss.update(img_loss.item())
        
        
        # print status 
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
            'Total Loss {loss.val:.4f} ({loss.avg:.4f})\tText {text_loss.val:.4f} ({text_loss.avg:.4f})\timage {img_loss.val:.4f} ({img_loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, iteration, len(dataset),loss=total_avg_loss, acc=avg_acc, text_loss=text_avg_loss, img_loss=img_avg_loss))
        
        iteration += 1 
        break
    return avg_acc.avg

def img_bank_construct(id2feature):
    img_bank = []
    for k in id2feature.keys():
        img_bank.append(id2feature[k])
    return img_bank 


if __name__ == "__main__":
    main() 
