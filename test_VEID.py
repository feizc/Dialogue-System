import torch
import json
import torch.nn as nn
import numpy as np
import copy
from transformers import *

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu 
from nltk.translate.bleu_score import SmoothingFunction 

from train_VEID import SPECIAL_TOKENS, SPECIAL_TOKENS_DICT
import torch.nn.functional as F
from MMdialog import VEID 
from utils import AverageMeter 


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def build_input_from_input(conv, current_output, tokenizer):
    eos = tokenizer.eos_token_id
    conv_seq = []
    temp_conv = []
    temp_len = 0
    token_type_seq = []
    for i in range(len(conv)):
        if temp_len + len(conv[i]) < 1000:
            temp_conv.append(conv[i])
            temp_len += len(conv[i])
        else:
            while temp_len + len(conv[i]) >= 1000:
                a = len(temp_conv[0])
                temp_conv = temp_conv[1:]
                temp_len -= a
            temp_conv.append(conv[i])
            temp_len += len(conv[i])

    for i in range(len(temp_conv)):
        conv_seq.extend(temp_conv[i][:128])
        conv_seq.append(eos)

    conv_seq = torch.LongTensor(conv_seq[:-1]).unsqueeze(0)
    conv_seq = conv_seq.expand(args.batch_size, -1)
    conv_seq = torch.cat([conv_seq, current_output], dim=-1)
    return conv_seq


def sample_sequence(src, tokenizer, model, args):
    special_tokens_ids = [tokenizer.bos_token, tokenizer.eos_token_id]
    eos = tokenizer.eos_token_id
    final_output = torch.LongTensor([[eos]] * args.batch_size)
    past_key_values = None
    src = tokenize(src, tokenizer)
    input_ids  = build_input_from_input(src, final_output, tokenizer)
    input_ids = input_ids.to(args.device)
    for i in range(args.max_length):
        if past_key_values is not None:
            input_ids = current_output
            logits, past_key_values = model(input_ids, past=past_key_values, use_cache=True)
        else:
            logits, past_key_values = model(input_ids, use_cache=True)

        logits = logits[:, -1, :] / args.temperature
        if i < args.min_length:
            logits[:, eos] = -1e9
        logits = torch.cat([top_filtering(logits[j].squeeze(), top_k=args.top_k, top_p=args.top_p).unsqueeze(0) for j in range(args.batch_size)], dim=0)
        probs = F.softmax(logits, dim=-1)
        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        current_output = prev
        final_output = torch.cat([final_output, prev.cpu()], dim=-1)
    decode_result = []
    for i in range(0, args.batch_size):
        temp = final_output[i, 1:].cpu().tolist()
        temp1 = []
        for j in temp:
            if j in special_tokens_ids:
                break
            temp1.append(j)
        decode_result.append(tokenizer.decode(temp1, skip_special_tokens=True).replace("\n", "") + "\n")
    return decode_result 




def build_input_from_input_beam(conv, current_output, tokenizer):
    eos = tokenizer.eos_token_id
    conv_seq = []
    temp_conv = []
    temp_len = 0
    token_type_seq = []
    for i in range(len(conv)):
        if temp_len + len(conv[i]) < 1000:
            temp_conv.append(conv[i])
            temp_len += len(conv[i])
        else:
            while temp_len + len(conv[i]) >= 1000:
                a = len(temp_conv[0])
                temp_conv = temp_conv[1:]
                temp_len -= a
            temp_conv.append(conv[i])
            temp_len += len(conv[i])

    for i in range(len(temp_conv)):
        conv_seq.extend(temp_conv[i][:128])
        conv_seq.append(eos)
    conv_seq.extend(current_output)

    conv_seq = torch.LongTensor(conv_seq).unsqueeze(0)
    return conv_seq


def beam_search(src, tokenizer, model, args):
    special_tokens_ids = [tokenizer.bos_token, tokenizer.eos_token_id]
    current_output = []
    hyplist = [([], 0., current_output)]
    best_state = None
    comp_hyplist = []

    instance = create_history(src, tokenizer)

    for i in range(args.max_length):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            input_ids, token_type_ids  = create_input_clip_beam(instance, st, tokenizer)
            input_ids = input_ids.to(args.device)
            token_type_ids = token_type_ids.to(args.device) 
            #print(input_ids)
            #print(token_type_ids)
            
            logits = model.infer(input_ids=input_ids.unsqueeze(0), token_type_ids=token_type_ids.unsqueeze(0)) 
            #print(logits.size())
            logp = F.log_softmax(logits, dim=-1)[:, -1, :]
            #print(logp.size())
            
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
            
            if i >= args.min_length:
                new_lp = lp_vec[tokenizer.eos_token_id] + args.penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]: 
                if o == tokenizer.unk_token_id or o == tokenizer.eos_token_id:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        return [tokenizer.decode(maxhyps[0][0], skip_special_tokens=True).replace("\n", "")]
    else:
        return [([], 0)]


# def meme_retrieval()


class Config():
    def __init__(self):
        self.max_length = 30
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.top_k = 1
        self.top_p = 0.0
        self.min_length = 6
        self.no_sample = False
        self.temperature = 1
        self.model_checkpoint = "ckpt/VEID"
        self.batch_size = 2
        self.beam_size = 5
        self.penalty = 0.1



def create_history(history, tokenizer):
    
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    input_ids = []
    token_type_ids = []
    instance = {}
    for i in range(len(history)):
        inp = [bos] + tokenize(history[i]['txt'], tokenizer) + [eos] 
        t_sp = speaker1 if i%2 == 0 else speaker2  
        inp = [t_sp] + inp
        token_type = [t_sp] * len(inp) 
        input_ids += inp
        token_type_ids += token_type 
    instance['input_ids'] = input_ids 
    instance['token_type_ids'] = token_type_ids 
    return instance 



def create_input_clip_beam(instance, answer, tokenizer):
    
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    
    t_sp = speaker1 if instance['token_type_ids'][-1] == speaker2 else speaker2 
    
    input_ids = copy.deepcopy(instance['input_ids']) + answer  
    token_type_ids = copy.deepcopy(instance['token_type_ids']) + [t_sp] * len(answer)
    #print(len(input_ids), len(target_ids), len(token_type_ids))
    input_ids = torch.from_numpy(np.array(input_ids)).long()
    token_type_ids = torch.from_numpy(np.array(token_type_ids)).long() 

    return input_ids, token_type_ids


args = Config() 
ckpt_path = 'ckpt/VEID/model.bin' 
tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint, do_lower_case=True) 
tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
model_config = GPT2Config.from_pretrained(args.model_checkpoint)

model = VEID(model_config) 
ckpt = torch.load(ckpt_path, map_location='cpu') 
model.load_state_dict(ckpt['model'])

smooth = SmoothingFunction() 
meter = AverageMeter()

with open('data/new_small_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f) 

total_num = len(data)
current_num = 0 

case_study = []

for dia in data: 
    #print(dia) 
    case = {}  
    print(current_num, '/', total_num)
    case['history'] = dia['history']
    reference = [[dia['answer']['txt']]] 
    print(reference)
    case['reference'] = reference 
    hypothesis = [beam_search(dia['history'], tokenizer, model, args)[0].replace(' ', '')]
    print(hypothesis)
    case['hypothesis'] = hypothesis 
    bleu = corpus_bleu(reference, hypothesis, weights=(1,0,0,0), smoothing_function=smooth.method1)
    meter.update(bleu) 
    case['bleu'] = bleu 
    print(bleu, meter.avg)
    current_num += 1 
    case_study.append(case)

    break 

#print(case_study)
with open('case_study.json', 'w', encoding='utf-8') as f: 
    json.dump(case_study, f, indent=4)

# model = torch.load("log_pretrain/DPKS_model5490.bin")
'''
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")
model.cuda()
model.eval()
print("finish loading model")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
with open("test.refs.txt") as f:
    data = f.readlines()

data_results = []
for i in data[3258:]:
    temp = i.split("\t")
    history = temp[0].split(" EOS ")
    responses = temp[1:]
    hypstr = beam_search(history, tokenizer, model, args)
    with open("dialoGPT_results_large_beam_01.txt", "a+", encoding="utf-8") as f:
        f.writelines(hypstr)
'''

