import torch 
from transformers import * 
import os 

from dataset import get_data, MMDataset, SPECIAL_TOKENS, SPECIAL_TOKENS_DICT  
from MMdialog import MMdialog 
from train import input_construct 


# 将回复中不含情感标记的对话删去 
def refine_dialog_list(dialog_list):
    new_dialog_list = []
    for dialog in dialog_list:
        answer = dialog['answer']
        if 'emotion_id' in answer.keys():
            new_dialog_list.append(dialog)
    return new_dialog_list 


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(): 
    emotion_num = 62 
    data_path = 'data/data.json'
    feature_path = 'data/id2feature.json' 
    ckpt_path = 'ckpt/emotion'
    epochs = 10 
    grad_accumulation_steps = 5 
    
    
    tokenizer = BertTokenizer.from_pretrained('./ckpt/cdial-gpt', do_lower_case=True) 
    model = MMdialog.from_pretrained('ckpt/cdial-gpt')
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4) 

    # 数据集读取
    dialog_list, id2feature = get_data(tokenizer, data_path, feature_path) 
    dialog_list = refine_dialog_list(dialog_list)
    # print(dialog_list[0])

    dataset = MMDataset(dialog_list, id2feature, tokenizer, 'emotion') 
    model.train()

    for epoch in range(epochs): 
        iteration = 1 
        for instance in dataset: 
            history_txt, history_img, token_type_ids, labels = instance 
            if token_type_ids.size(0) > 500:
                continue
            history_txt, history_img, token_type_ids, labels  = history_txt.to(device), history_img.to(device), token_type_ids.to(device), labels.to(device)

            history_txt_embs = model.transformer.wte(history_txt) 
            history_img_embs = model.image_off(history_img).squeeze(1)

            input_embs, img_features = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer) 
            input_embs, img_features = input_embs[:-1, :].to(device), img_features.to(device) 
            input_embs = torch.cat([input_embs, img_features], dim=0).to(device)
            # print(input_embs.size(), img_features.size(), token_type_ids.size())
            loss = model(input_embs, token_type_ids, labels, img_features, 'emotion') 

            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if iteration % grad_accumulation_steps == 0:
                optimizer.step() 
                optimizer.zero_grad()
                print(loss.item())
                break
            iteration += 1
        
        torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},\
                '%s/epoch_%d'%(ckpt_path, epoch))
        model.config.to_json_file(os.path.join(ckpt_path, 'config.json'))
        break 



if __name__ == "__main__":
    train()

