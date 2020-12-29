from transformers import * 
import torch 
import torch.nn.functional as F 
import json 
import copy 
import numpy as np 


from train import SPECIAL_TOKENS_DICT, SPECIAL_TOKENS
from MMdialog import MMdialog 
from dataset import MMDataset, get_data, build_input_from_segments
from train import input_construct 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# greedy search 
def greedy_decode(input_embs, token_type_ids, model, tokenizer, max_len=20):
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    res = []
    for i in range(max_len):
        logits = model.inference(input_embs, token_type_ids) # (input_length, vocab_size)
        logits = logits.cpu().data.numpy()
        next_word = np.argsort(logits[-1])[-1]
        if next_word == eos:
            break
        res.append(next_word)
        clone_id = token_type_ids[0].clone().view(-1)
        token_type_ids = torch.cat((token_type_ids, clone_id), 0)
        word_emb = model.transformer.wte(torch.LongTensor([next_word]))
        input_embs = torch.cat((input_embs, word_emb), 0)
        
    return res


# beam search 

# top-k sampling 
def sample_sequence(input_embs, token_type_ids, model, tokenizer, max_len=20): 
    temperature = 0.7 
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    res = []
    for i in range(max_len):
        logits = model.inference(input_embs, token_type_ids) 
        logits = logits[-1] / temperature 
        logits = top_filtering(logits, top_k=0, top_p=0.9)
        probs = F.softmax(logits, dim=-1)
        next_word = torch.multinomial(probs, 1).item()
        if next_word == eos: 
            break
        res.append(next_word)
        clone_id = token_type_ids[0].clone().view(-1)
        token_type_ids = torch.cat((token_type_ids, clone_id), 0)
        word_emb = model.transformer.wte(torch.LongTensor([next_word]))
        input_embs = torch.cat((input_embs, word_emb), 0) 
        break 
    return res 


# select top-k or top-p candidates
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')): 
    assert logits.dim()==1 
    top_k = min(top_k, logits.size(-1)) 
    if top_k > 0:
        idxs_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[idxs_to_remove] = filter_value 
    if top_p > 0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True) 
        cummulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) 

        sorted_idx_to_remove = cummulative_probs > top_p 
        sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[...,:-1].clone()
        sorted_idx_to_remove[...,0] = 0 

        idxs_to_remove = sorted_idx[sorted_idx_to_remove]
        logits[idxs_to_remove] = filter_value 
    idxs_to_remove = logits < threshold 
    logits[idxs_to_remove] = filter_value
    # print(logits.size())
    return logits



# response generaton 
def generate_responce(model, dialog_list, id2feature, tokenizer):
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    with torch.no_grad():
        for dialog in dialog_list:
            if len(dialog['history']) < 2:
                continue
            history = copy.deepcopy(dialog['history'][:-1])
            answer = copy.deepcopy(dialog['history'][-1])
            for i in range(len(history)):
                if 'img_id' in history[i].keys():
                    history[i]['img_id'] = id2feature[history[i]['img_id']]
            if 'img_id' in answer.keys():
                answer['img_id'] = id2feature[answer['img_id']]
            history_txt, history_img, token_type_ids, _ = build_input_from_segments(history, answer, tokenizer)
            if history_txt[-1] == tag:
                history_txt[-1] = img
            if answer['speaker_id'] == '[speaker1]':
                history_txt += [speaker2]
                token_type_ids += [speaker2] * 2
            else:
                history_txt += [speaker1]
                token_type_ids += [speaker1] * 2
            history_txt += [bos]

            history_txt = torch.LongTensor(history_txt)
            history_img = torch.from_numpy(np.array(history_img)).float()
            token_type_ids = torch.Tensor(token_type_ids).long()

            history_txt, history_img, token_type_ids  = history_txt.to(device), history_img.to(device), token_type_ids.to(device)
            history_txt_embs = model.transformer.wte(history_txt)
            history_img_embs = model.image_off(history_img).squeeze(1)

            input_embs, img_features = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer)
            input_embs, img_features = input_embs.to(device), img_features.to(device)
            print(input_embs.size(), token_type_ids.size())

            res = sample_sequence(input_embs, token_type_ids, model, tokenizer)
            #res = greedy_decode(input_embs, token_type_ids, model, tokenizer)
            #res = tokenizer.decode(res, skip_special_tokens=True)
            #print(res)
            break





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

    test_path = 'data/eval.json'
    feature_path = 'data/id2feature.json'
    test_data = json.load(open(test_path, 'r', encoding='utf-8'))
    # print(test_data)

    dialog_list, id2feature = get_data(tokenizer, test_path, feature_path)
    print(dialog_list[-1])
    result = generate_responce(model, dialog_list, id2feature, tokenizer)







