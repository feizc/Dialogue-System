import os 
import shutil 
import json 
import emoji 
import torch 



# 将图片和标记放到一个文件夹，方便读取
def image_file_construct(source_path, target_path, emoji_label_dict):
    image_target_path = os.path.join(target_path, 'image')
    json_target_path = os.path.join(target_path, 'img2label.json')

    if not os.path.exists(image_target_path):
        os.mkdir(image_target_path)
    files = os.listdir(source_path)

    img_label_dict = {}
    max_imag_num = 30000
    i = 0 
    for file in files:
        if file[0] == '.':
            continue
        image_file = os.path.join(source_path, file)
        name_list = os.listdir(image_file)
        for name in name_list:
            name_path = os.path.join(image_file, name)

            # 读取图片对应的标签
            if name_path[-3:] == 'txt':
                with open(name_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.split('\t')
                    if emoji.emojize(line[1][:-1]) in emoji_label_dict.keys():
                        img_label_dict[line[0]] = emoji_label_dict[line[1][:-1]]
            else:
                shutil.copy(name_path, image_target_path)
                i += 1
        if i >= max_imag_num:
            break
    # 将标注信息存为json
    with open(json_target_path, 'w', encoding='utf-8') as f:
        json.dump(img_label_dict, f, indent=4)
    return img_label_dict 


# 读取emoji_vocab文件并返回dict, 这里只对频率前300的emoji做预训练
def emoji_vocab_read(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    i = 0 
    emoji_label_dict = {}
    for line in lines:
        emoji, _ = line.split()
        emoji_label_dict[emoji] = i 
        i += 1
        if i > 299:
            break
    return emoji_label_dict


# 删除没有标注的图片
def remove_img(path, dict):
    img_path = os.path.join(path, 'image')
    img_list = os.listdir(img_path)
    for name in img_list:
        if name[:-4] not in dict.keys() and name[-3:] == 'npy':
            path = os.path.join(img_path, name)
            os.remove(path)



# 预训练数据调整
def pretrain_data_process():
    pretrain_data_path = os.path.join(os.getcwd(), 'data')
    pretrain_data_path = os.path.join(pretrain_data_path, 'pretrain_data')
    source_path = os.path.join(pretrain_data_path, 'npy_stickers')
    target_path = pretrain_data_path

    # 筛选top300的表情，生成dict
    emoji_label_dict = emoji_vocab_read(os.path.join(pretrain_data_path, 'emoji_vocab'))
    print(emoji_label_dict)
    # 选择有标记的图片，放到\image 文件夹下面 
    img_label_dict = image_file_construct(source_path, target_path, emoji_label_dict)
    remove_img(target_path, img_label_dict)



# calculate the accuracy of respond 
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



# class for evaluation metric 
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



if __name__ == "__main__": 
    pretrain_data_process()

