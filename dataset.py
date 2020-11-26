import torch 
from torch.utils.data import Dataset 
import os 
import json 


class ExpressionDataset(Dataset):
    def __init__(self, data_folder):
        self.image_path = os.path.join(data_folder, 'image')
        self.image_name_list = os.listdir(self.image_path)
        with open(os.path.join(data_folder, 'img_label.json'), 'r', encoding='utf-8') as f:
            self.img_tag_dict = json.load(f)
        print(self.img_tag_dict)




if __name__ == "__main__":
    train_data = ExpressionDataset('./npy_image')