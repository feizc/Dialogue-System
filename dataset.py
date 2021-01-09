import torch 
from torch.utils.data import Dataset 
import os 
import json 
#from skimage import io, transform 
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt  
from transformers import BertTokenizer 
import copy


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}


# 预训练表情和对应情感的标签分类
class ExpressionDataset(Dataset):
    def __init__(self, data_folder):
        self.image_path = os.path.join(data_folder, 'image')
        files = os.listdir(self.image_path)
        self.image_name_list = []
        for file in files:
            if file[-3:] == 'npy':
                self.image_name_list.append(file)
        with open(os.path.join(data_folder, 'img2label.json'), 'r', encoding='utf-8') as f:
            self.img2label_dict = json.load(f)

        
    def __getitem__(self, i):
        current_image_path = os.path.join(self.image_path, self.image_name_list[i])
        tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = np.load(current_image_path, allow_pickle=True, encoding='bytes')
        #print(img[0])
        #plt.imshow(img)
        #plt.show()
        img = tfms(Image.fromarray(img))
        label = self.img2label_dict[self.image_name_list[i][:-4]]
        label = np.array(int(label))
        label = torch.from_numpy(label)
        return img, label
        
    
    def __len__(self):
        return len(self.image_name_list)


def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n,o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


# 将训练集拆分成 history + answer 形式
def get_data(tokenizer, data_path, feature_path):
    dialog_data = json.load(open(data_path, 'r'))
    dialog_list = []
    for idx in dialog_data.keys():
        dialog = dialog_data[idx]
        history = []
        for i in range(len(dialog)):
            if 'txt' in dialog[i].keys():
                dialog[i]['txt'] = tokenize(dialog[i]['txt'], tokenizer)
            if i == 0:
                history.append(dialog[i])
                continue
            item = {'history': copy.deepcopy(history), 'answer': copy.deepcopy(dialog[i])}
            dialog_list.append(item)
            history.append(dialog[i])
    id2feature = json.load(open(feature_path, 'r'))

    return dialog_list, id2feature


class MMDataset(Dataset): 
    # mode: dialog / emotion 
    def __init__(self, dialogs, id2feature, tokenizer, mode='dialog'):
        self.dialogs = dialogs
        self.id2feature = id2feature
        self.tokenizer = tokenizer
        self.mode = mode
    
    def __len__(self):
        return len(self.dialogs)
    
    def __getitem__(self, index):
        his = copy.deepcopy(self.dialogs[index]['history'])
        ans = copy.deepcopy(self.dialogs[index]['answer'])

        for i in range(len(his)):
            if 'img_id' in his[i].keys():
                his[i]['img_id'] = self.id2feature[his[i]['img_id']]
                
        if 'img_id' in ans.keys():
            ans['img_id'] = self.id2feature[ans['img_id']]
        
        history_txt, history_img, token_type_ids, labels = build_input_from_segments(his, ans, self.tokenizer) 
        # print(history_img)
        history_txt = torch.LongTensor(history_txt)
        history_img = torch.from_numpy(np.array(history_img)).float()
        
        token_type_ids = torch.Tensor(token_type_ids).long()
        labels = torch.Tensor(labels).long() 

        if self.mode == 'emotion': 
            new_labels = [-100] * labels.size(0)
            new_labels[-1] = ans['emotion_id']
            new_labels = torch.Tensor(new_labels).long() 
            return history_txt, history_img, token_type_ids, new_labels 

        return history_txt, history_img, token_type_ids, labels



class LPDataset(Dataset): 
    def __init__(self, path, tokenizer):
        with open(path, 'r', encoding='utf-8') as f: 
            self.data = json.load(f) 
        self.tokenizer = tokenizer 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        his = self.data[index]['history']
        respond = self.data[index]['respond'] 
        his = tokenize(his, self.tokenizer) 
        his_ids = concate_input(his, self.tokenizer) 
        his_ids = torch.LongTensor(his_ids)
        respond = torch.LongTensor([respond])
        return his_ids, respond 


def concate_input(history, tokenizer):
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    his_ids = [bos] 
    for sentence_id in history:
        his_ids += sentence_id 
    his_ids += [bos, tag]
    return his_ids 



# 将文字和图像分开拼接，toekn type记录位置
def build_input_from_segments(history, answer, tokenizer):
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    history_txt = []
    history_img = []
    token_type_ids = []
    labels = []
    his_length = len(history)
    for i in range(his_length):
        idx = his_length - 1 - i
        if len(token_type_ids) > 450:
            break

        if history[idx]['speaker_id'] == '[speaker1]':
            speaker_id = speaker1
        else:
            speaker_id = speaker2

        if 'img_id' in history[idx].keys():
            history_img.append(history[idx]['img_id'])
            #print(np.array(history[idx]['img_id']).shape)
            token_type_ids = [img] + token_type_ids
            labels = [-100] + labels
        if 'txt' in history[idx].keys():
            content = [bos] + history[idx]['txt'] + [eos]
            history_txt = content + history_txt
            token_type_ids = [speaker_id]*len(content) + token_type_ids
            labels = [-100]*len(content) + labels

        history_txt = [speaker_id] + history_txt
        token_type_ids = [speaker_id] + token_type_ids 
        labels = [-100] + labels
    
    history_img.reverse()
    #print(np.array(history_img).shape)
    if answer['speaker_id'] == '[speaker1]':
        speaker_id = speaker1
    else:
        speaker_id = speaker2
    
    history_txt += [speaker_id]
    token_type_ids += [speaker_id]

    if 'txt' in answer.keys():
        content = [bos] + answer['txt'] + [eos]
        history_txt += content
        token_type_ids += [speaker_id]*len(content)
        labels += content
    labels +=[-100]
    history_txt += [tag]
    token_type_ids += [img]
    if 'img_id' in answer.keys():
    #    print(len(answer['img_id'][0]))
        history_img.append(answer['img_id'])
    else:
        history_img.append([[0.0]*62720])
    # print(history_img)
    
    return history_txt, history_img, token_type_ids, labels





# 需要转为Tensoer的拼接
def build_inputs(history, answer, tokenizer, word_embedding, image_embedding): 
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    inp_embs = [], token_type_ids = [], labels = []
    for turn in history:
        if turn['speaker_id'] == 'speaker1':
            speaker_id = speaker1
        else:
            speaker_id = speaker2
        inp_embs.append(word_embedding(speaker_id))
        token_type_ids.append(speaker_id)
        labels.append(-100.0)
        if 'txt' in turn.keys():
            inp_embs.append(word_embedding(bos))
            token_type_ids.append(speaker_id)
            labels.append(-100.0)

            for word_id in turn['txt']:
                inp_embs.append(word_embedding(word_id))
                token_type_ids.append(speaker_id)
                labels.append(-100.0)
            
            inp_embs.append(word_embedding(eos))
            token_type_ids.append(speaker_id)
            labels.append(-100.0)

        if 'img_id' in turn.keys():
            inp_embs.append(image_embedding(turn['img_id']))
            token_type_ids.append(img)
            labels.append(-100.0)
    
    # 将答案编码
    if answer['speaker_id'] == 'speaker1':
        speaker_id = speaker1
    else:
        speaker_id = speaker2
    inp_embs.append(word_embedding(speaker_id))
    token_type_ids.append(speaker_id)
    labels.append(-100.0)
    if 'txt' in answer.keys():
        inp_embs.append(word_embedding(bos))
        token_type_ids.append(speaker_id)

        for word_id in turn['txt']:
            inp_embs.append(word_embedding(word_id))
            token_type_ids.append(speaker_id)
            labels.append(word_id)
            
        inp_embs.append(word_embedding(eos))
        token_type_ids.append(speaker_id)
        labels.append(eos)
        labels.append(-100.0)
    
    image_flag = False
    target_feature = None
    if 'img_id' in answer.keys():
        target_feature = image_embedding(turn['img_id'])
        image_flag = True

    inp_embs.append(tag)
    token_type_ids.append(img)

    return inp_embs, token_type_ids, labels, image_flag, target_feature




if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('./ckpt/cdial-gpt', do_lower_case=True) 
    train_data = LPDataset('./data/label_train.json', tokenizer)
    img, label = train_data[0]
    print(img, label)
    '''

    data_path = './data/data.json'
    feature_path = './data/id2feature.json'
    
    tokenizer = BertTokenizer.from_pretrained('./ckpt/cdial-gpt', do_lower_case=True)
    dialogs, id2feature = get_data(tokenizer, data_path, feature_path)
    print(dialogs[0])
    dataset = MMDataset(dialogs, id2feature, tokenizer)
    dataset[0]

    
    for item in dataset:
        _, _, _, labels = item
        break
    '''
