import os 
import shutil 
import json 

# 将图片和标记放到一个文件夹，方便读取
def image_file_construct(source_path, target_path):
    image_target_path = os.path.join(target_path, 'image')
    json_target_path = os.path.join(target_path, 'img2tag.json')

    if not os.path.exists(image_target_path):
        os.mkdir(image_target_path)
    files = os.listdir(source_path)
    # print(files)
    img_label_dict = {}
    for file in files:
        if file[0] == '.':
            continue
        image_file = os.path.join(source_path, file)
        name_list = os.listdir(image_file)
        for name in name_list:
            name_path = os.path.join(image_file, name)

            # 读取图片对应的标签
            if name_path[-3:] == 'txt':
                # print(name_path)
                with open(name_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.split('\t')
                    img_label_dict[line[0]]=line[1][:-1]
            else:
                shutil.copy(name_path,image_target_path)
    
    return img_label_dict
    # with open(json_target_path, 'w', encoding='utf-8') as f:
    #     json.dump(img_label_dict, f, indent=4)
    # print(img_label_dict)


# 将vocab文件转为json格式方便读取
def vocab2json(fold_path):
    tag2label_dict = {}
    with open(os.path.join(fold_path, 'emoji_vocab'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # print(lines)
    for line in lines:
        line = line.split()
        tag2label_dict[line[0]] = line[1]
    return tag2label_dict
    # with open(os.path.join(fold_path, 'tag2label.json'), 'w', encoding='utf-8') as f:
    #     json.dump(tag2label_dict, f, indent=4)


# 生成图片对应label的json文件
def label2json(img_tag_dict, tag_label_dict, output_fold):
    img_label_dict = {}
    for image_name in img_tag_dict.keys():
        tag = img_tag_dict[image_name]
        if tag in tag_label_dict.keys():
            img_label_dict[image_name] = tag_label_dict[tag]
        else:
            print(tag)
    # print(img_label_dict)
    with open(os.path.join(output_fold, 'img2label.json'), 'w', encoding='utf-8') as f:
        json.dump(img_label_dict, f, indent=4)
    
    return img_label_dict 


# 去掉部分无效的表情图片
def remove_img(img_fold, dict):
    img_list = os.listdir(img_fold)
    for name in img_list:
        if name[:-4] not in dict.keys() and name[-3:] == 'npy':
            path = os.path.join(img_fold, name)
            os.remove(path)


if __name__ == "__main__":
    img_tag_dict = image_file_construct('./npy_image/npy_stickers', './npy_image')
    tag2label_dict = vocab2json('./npy_image')
    img_label_dict = label2json(img_tag_dict, tag2label_dict, './npy_image')
    remove_img('./npy_image/image', img_label_dict)