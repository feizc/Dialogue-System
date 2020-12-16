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

epochs = 20
lr = 6.5e-5 
model_path = 'ckpt/mmgpt'
gradient_accumulation_steps = 5 



def accuracy_compute(lm_logits, targets, k):
    _, idx = torch.topk(lm_logits, k, 1)
    correct = idx.eq(targets.view(-1,1).expand_as(idx))
    #print(correct)
    correct_total = correct.view(-1).float().sum().item()
    nums = targets.view(-1).detach().cpu().numpy()
    length = 0
    for num in nums:
        if num != -100:
            length += 1
    return correct_total / float(length)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train():
    
    # 模型初始化
    tokenizer = BertTokenizer.from_pretrained('ckpt/cdial-gpt', do_lower_case=True)
    model = MMdialog.from_pretrained('ckpt/cdial-gpt')
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # 数据读取
    dialogs, id2feature = get_data(tokenizer, train_data_path, feature_path)
    dataset = MMDataset(dialogs, id2feature, tokenizer)


    # 模型的训练
    model.train()
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        iteration = 1
        for instance in dataset: 
            history_txt, history_img, token_type_ids, labels = instance 
            if token_type_ids.size(0) > 500:
                continue
            history_txt, history_img, token_type_ids, labels  = history_txt.to(device), history_img.to(device), token_type_ids.to(device), labels.to(device)
            #print(token_type_ids)
            #print(labels)

            # print(history_txt.size())
            history_txt_embs = model.transformer.wte(history_txt)
            history_img_embs = model.image_off(history_img).squeeze(1)
            input_embs, img_features = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer)
            input_embs, img_features = input_embs.to(device), img_features.to(device)
            lm_logits, loss = model(input_embs, token_type_ids, labels, img_features)
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if iteration % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                avg_loss.update(loss.item() / gradient_accumulation_steps)
                break
            acc = accuracy_compute(lm_logits, labels, 5)
            avg_acc.update(acc)
            iteration += 1
            # print(loss.item())
            print('acc:', acc)

        torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},\
                    '%s/epoch_%d_acc_%.3f'%(model_path, epoch, avg_acc.avg))
        model.config.to_json_file(os.path.join(model_path, 'config.json'))
        tokenizer.save_vocabulary(model_path)
        loss_list.append(avg_loss.avg)
        acc_list.append(avg_acc.avg)
        break
        
    print(loss_list)
    print(acc_list)


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
