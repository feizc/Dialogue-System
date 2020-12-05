import torch 
from torch.utils.data import Dataset 
import os 
import json 
from skimage import io, transform 
import numpy as np
import torchvision.transforms as transformers
from PIL import Image
import matplotlib.pyplot as plt  
from transformers import BertTokenizer


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[TXT]', '[IMG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TXT]'], 'pad_token':'[PAD]'}


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
        tfms = transformers.Compose([transformers.Resize(224), transformers.ToTensor(), transformers.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = np.load(current_image_path, allow_pickle=True, encoding='bytes')
        # plt.imshow(img)
        # plt.show()
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
        return dict((n, tokenize(o)) for n,o in obj.items())
    return list(tokenize(o) for i in obj)


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
                print(history)
                continue
            item = {'history': history, 'answer':dialog[i]}
            dialog_list.append(item)
            history.append(dialog[i])
    #print(dialog_list[0])









if __name__ == "__main__":
    '''
    train_data = ExpressionDataset('./data/pretrain_data')
    img, label = train_data[0]
    print(img, label)
    '''

    data_path = './data/data.json'
    feature_path = './data/id2feature.json'
    
    tokenizer = BertTokenizer.from_pretrained('./ckpt/cdial-gpt', do_lower_case=True)
    get_data(tokenizer, data_path, feature_path)
