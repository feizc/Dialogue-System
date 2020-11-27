import torch 
from torch.utils.data import Dataset 
import os 
import json 
from skimage import io, transform 
import numpy as np
import torchvision.transforms as transformers
from PIL import Image
import matplotlib.pyplot as plt  

class ExpressionDataset(Dataset):
    def __init__(self, data_folder):
        self.image_path = os.path.join(data_folder, 'image')
        self.image_name_list = os.listdir(self.image_path)
        # print(self.image_name_list)
        with open(os.path.join(data_folder, 'img2tag.json'), 'r', encoding='utf-8') as f:
            self.img2tag_dict = json.load(f)
        with open(os.path.join(data_folder, 'tag2label.json'), 'r', encoding='utf-8') as f:
            self.tag2label_dict = json.load(f)
        
    def __getitem__(self, i):
        current_image_path = os.path.join(self.image_path, self.image_name_list[i])
        tfms = transformers.Compose([transformers.Resize(224), transformers.ToTensor(), transformers.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = np.load(current_image_path)
        # plt.imshow(img)
        # plt.show()
        img = tfms(Image.fromarray(img))
        tag = self.img2tag_dict[self.image_name_list[i][:-4]]
        label = np.array(int(self.tag2label_dict[tag]))
        label = torch.from_numpy(label)
        return img, label
        
    
    def __len__(self):
        return len(self.image_name_list)





if __name__ == "__main__":
    train_data = ExpressionDataset('./npy_image')
    img, label = train_data[0]
    print(img, label)