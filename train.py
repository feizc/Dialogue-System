from transformers import * 
import os
import sys
from MMdialog import MMdialog 
from dataset import MMDataset, build_input_from_segments, get_data
import torch 
from utils import accuracy_compute, AverageMeter 
from nltk.translate.bleu_score import corpus_bleu 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}

# Data parameters
train_data_path = 'data/eval.json'
feature_path = 'data/id2feature.json'


# model parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
epochs = 20
lr = 6.5e-5 
model_path = 'ckpt/mmgpt'
gradient_accumulation_steps = 5 
checkpoint_usage = True
print_freq = 1


# training and validation 
def main():

    # 模型初始化 
    if checkpoint_usage == True: 
        ckpt_path = 'ckpt/mmgpt/model.bin'
        tokenizer = BertTokenizer.from_pretrained('ckpt/mmgpt', do_lower_case=True)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        model_config = GPT2Config.from_pretrained('ckpt/mmgpt')

        model = MMdialog(model_config)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model']) 
    
    else:
        tokenizer = BertTokenizer.from_pretrained('ckpt/cdial-gpt', do_lower_case=True)
        model = MMdialog.from_pretrained('ckpt/cdial-gpt')
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # 数据读取
    dialogs, id2feature = get_data(tokenizer, train_data_path, feature_path)    
    dataset = MMDataset(dialogs, id2feature, tokenizer) 
    
    # Epochs 
    for epoch in range(epochs):

        # one epoch's training 
        train(model=model, tokenizer=tokenizer, optimizer=optimizer, dataset=dataset, epoch=epoch)
        break





def train(model, tokenizer, optimizer, dataset, epoch):

    # 模型的训练
    model.train()

    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    iteration = 1

    for instance in dataset: 
        history_txt, history_img, token_type_ids, labels = instance 
        if token_type_ids.size(0) > 500:
            continue
        history_txt, history_img, token_type_ids, labels  = history_txt.to(device), history_img.to(device), token_type_ids.to(device), labels.to(device)
            
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
            
        acc = accuracy_compute(lm_logits, labels, 5)
        avg_acc.update(acc)
        
        # print status 
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc))
        
        iteration += 1 
        break

         
        # print(loss.item())
        # print('acc:', acc)

        #torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},\
        #            '%s/epoch_%d_acc_%.3f'%(model_path, epoch, avg_acc.avg))
        #model.config.to_json_file(os.path.join(model_path, 'config.json'))
        #tokenizer.save_vocabulary(model_path)
        #loss_list.append(avg_loss.avg)
        #acc_list.append(avg_acc.avg)
        
    #print(loss_list)
    #print(acc_list)


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
    
    main()


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
