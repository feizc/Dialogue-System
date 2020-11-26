import os 
import shutil 
import json 

# 将图片和标记放到一个文件夹，方便读取
def image_file_construct(source_path, target_path):
    image_target_path = os.path.join(target_path, 'image')
    json_target_path = os.path.join(target_path, 'img_label.json')

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
                print(name_path)
                with open(name_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.split('\t')
                    img_label_dict[line[0]]=line[1][:-1]
            else:
                shutil.copy(name_path,image_target_path)
    
    with open(json_target_path, 'w', encoding='utf-8') as f:
        json.dump(img_label_dict, f, indent=4)
    print(img_label_dict)

# 将vocab文件转为json格式方便读取
def vocab2json(fold_path):
    tag2label_dict = {}
    with open(os.path.join(fold_path, 'emoji_vocab'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # print(lines)
    for line in lines:
        line = line.split()
        tag2label_dict[line[0]] = line[1]
    with open(os.path.join(fold_path, 'tag2label.json'), 'w', encoding='utf-8') as f:
        json.dump(tag2label_dict, f, indent=4)


if __name__ == "__main__":
    print('haha')
    # image_file_construct('./npy_image/npy_stickers', './npy_image')
    # vocab2json('./npy_image')