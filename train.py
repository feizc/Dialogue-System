from transformers import * 
import os
import sys
from MMdialog import MMdialog 
from dataset import MMDataset, build_input_from_segments, get_data
import torch 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}

train_data_path = 'data/data.json'
feature_path = 'data/id2feature.json'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

lr = 6.5e-5

def train():
    
    # 读取数据集
    tokenizer = BertTokenizer.from_pretrained('ckpt/cdial-gpt', do_lower_case=True)
    dialogs, id2feature = get_data(tokenizer, train_data_path, feature_path)
    dataset = MMDataset(dialogs, id2feature, tokenizer)

    # 模型初始化
    config = GPT2Config(vocab_size=13092)
    model = MMdialog(config)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # 模型的训练
    model.train()
    for instance in dataset:
        history_txt, history_img, token_type_ids, labels = instance
        history_txt, history_img, token_type_ids, labels  = history_txt.to(device), history_img.to(device), token_type_ids.to(device), labels.to(device)
        #print(token_type_ids)
        #print(labels)

        history_txt_embs = model.transformer.wte(history_txt)
        history_img_embs = model.image_off(history_img).squeeze(1)
        input_embs, img_features = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer)
        _, loss = model(input_embs, token_type_ids, labels, img_features)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())

        break


# 将输入拼接成符合要求的tensor
def input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer):
    
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    emb_length = token_type_ids.size(-1)
    emb_dim = history_txt_embs.size(-1)
    img_num = history_img_embs.size(0)
    # txt_num = history_txt_embs.size(0)
    input_embs = torch.zeros((emb_length, emb_dim))

    txt_idx = 0
    img_idx = 0
    left_idx = 0
    right_idx = 0
    while right_idx < emb_length:
        if right_idx == emb_length-1 and token_type_ids[right_idx] == img:
            break
        if token_type_ids[right_idx] == img: 
            txt_length = right_idx - left_idx
            input_embs[left_idx:right_idx, :] = history_txt_embs[txt_idx:txt_idx + txt_length, :]
            txt_idx += txt_length
            input_embs[right_idx, :] = history_img_embs[img_idx, :]
            img_idx += 1
            left_idx = right_idx + 1
        right_idx += 1
    txt_length = right_idx - left_idx
    if txt_length > 0:
        input_embs[left_idx:right_idx, :] = history_txt_embs[txt_idx:txt_idx + txt_length, :]
    
    image_feature = torch.zeros((1, emb_dim)).to(device)
    if img_idx < img_num :
        image_feature[0,:] = history_img_embs[img_idx,:]
    return input_embs, image_feature
        





if __name__ == "__main__":
    
    train()


    '''
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
    '''
