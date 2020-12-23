import os 
import xlrd 
import json
from efficientnet_pytorch import EfficientNet
from PIL import Image 
from collections import Counter 
#import torchvision
#import torchvision.transforms as transformers

SPEAKER = ['[speaker1]', '[speaker2]']


# 将表情名映射为对应id
def img2id(path):
    img_path = os.path.join(path, 'image')
    img_name_list = os.listdir(img_path)
    img2id_dict = {}
    i = 1000
    for img in img_name_list:
        s = str(i)[1:]
        i += 1
        img2id_dict[img] = s
    return img2id_dict


# 读取excel对话文件
def excel_read(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows

    lines = []
    for i in range(nrows):
        if i == 0: 
            continue
        lines.append(table.row_values(i))
    return lines


# 结构调整，将表情放置于对话的后面
def format_modify(lines, img2id, img_dot_name_dict):
    for i in range(len(lines)):
        lines[i][1] = lines[i][1].strip()
        if lines[i][1] != "":
            left_idx = lines[i][1].find('[')
            right_idx = lines[i][1].find(']')
            img_name = lines[i][1][left_idx+1:right_idx]
            # 表情标注错误
            img_name = img_name.split('.')[0]
            if img_name in img_dot_name_dict.keys():
                img_name = img_dot_name_dict[img_name]
            if img_name[:-1] in img_dot_name_dict.keys():
                img_name = img_dot_name_dict[img_name[:-1]]

            if img_name not in img2id.keys():
                print(img_name) 
          
            # 把中间的表情移到后面
            if left_idx == 0:
                continue
            elif right_idx == len(lines[i][1])-1:
                continue
            else:
                tmpline = lines[i][1].split('[')[0]
                try:
                    tmpline += lines[i][1].split(']')[1]
                except:
                    print(lines[i][1])
                    print('not right')
                tmpline += lines[i][1][left_idx:right_idx+1]
                lines[i][1] = tmpline
    return lines
            


# 对每个句子取消分词
def delete_space(data_dict):
    for dialog in data_dict.keys():
        for respond in data_dict[dialog]:
            if 'txt' in respond.keys():
                tmp = str(respond['txt'])
                respond['txt'] = tmp.replace(' ', '')
    return data_dict


# 生成json数据集文件
def data2json(lines, img2id, img_dot_name_dict, emotion_dict):
    data = {}

    '''
    # 格式错误进行差错
    i = 2
    for line in lines:
        try:
            if 'dialogue' in line[0]:
                line[0]
        except:
            print(i, line)
        i += 1
    '''
    i = 0

    while i < len(lines):
        if 'dialogue' in str(lines[i][0]):
            dialogue_id = lines[i][0]
            dialogue_content = []
            j = 0
            while True:
                i += 1
                if i >= len(lines) or lines[i][0] == '' or lines[i][0] == ' ':
                    break
                sentence = {}
                sentence['speaker_id'] = SPEAKER[j%2]
                j += 1
                if lines[i][1] != '': 
                    emotion = lines[i][2].strip() 
                    if emotion in emotion_dict.keys(): 
                        sentence['emotion_id'] = emotion_dict[emotion]
                    left_idx = lines[i][1].find('[')
                    right_idx = lines[i][1].find(']')
                    img_name = lines[i][1][left_idx+1:right_idx]
                    img_name = img_name.split('.')[0]
                    if img_name in img_dot_name_dict.keys():
                        img_name = img_dot_name_dict[img_name]
                    if img_name[:-1] in img_dot_name_dict.keys():
                        img_name = img_dot_name_dict[img_name[:-1]]
                    try:
                        sentence['img_id'] = img2id[img_name]
                    except:
                        print(lines[i][1])
                    if left_idx == 0 and right_idx == len(lines[i][1])-1:
                        continue
                    elif left_idx == 0:
                        sentence['txt'] = lines[i][1][right_idx+1:]
                    else:
                        sentence['txt'] = lines[i][1][:left_idx]
                else:
                    sentence['txt'] = lines[i][0]
                dialogue_content.append(sentence) 
            data[dialogue_id] = dialogue_content
        i += 1
    # 删除字符间的空格 
    data = delete_space(data)
    # print(data['dialogue 1'])
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    


# 对每张表情提取特征，存入json
def image_process(path, model, img2id_dict):
    tsfm = transformers.Compose([transformers.Resize((224,224)),\
        transformers.ToTensor(),\
        transformers.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225]),])
    image_path = os.path.join(path, 'image')
    image_list = os.listdir(image_path)

    id2feature = {}
    for img_name in image_list:
        img_path = os.path.join(image_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img = tsfm(img).unsqueeze(0)
        feature = model.extract_features(img)
        feature = feature.squeeze(0).view(1, -1).detach().numpy()
        print(feature.shape)
        feature = feature.tolist()
        id2feature[img2id_dict[img_name]] = feature

    with open('id2feature.json', 'w', encoding='utf-8') as f:
        json.dump(id2feature, f)



# 为了避免后缀出错做出修正 
def img_dot_name_create(img2id_dict):
    img_dot_name_dict = {}
    for key in img2id_dict.keys():
        dot_name = key.split('.')[0]
        img_dot_name_dict[dot_name] = key
        img_dot_name_dict[dot_name[:-1]] = key
    # print(img_dot_name_dict) 
    return img_dot_name_dict 


# 统计情感相关的标签, 这里可能需要限制情感种类 
def emotion_label_calculate(lines):
    emotion_list = []
    emotion_counter = Counter()
    for line in lines:
        if line[2] == '':
            continue
        if line[2] not in emotion_list:
            emotion_list.append(line[2])
        emotion = line[2].strip()
        emotion_counter.update([emotion])
    print(emotion_counter) 
    emotion_dict = {}
    i = 0 
    for emotion in emotion_list:
        emotion_dict[emotion] = i 
        i += 1 
    return emotion_dict 




if __name__ == "__main__":

    
    # excel 文字处理
    data_path = os.getcwd()
    excel_path = os.path.join(data_path, 'dialogue.xlsx')
    img2id_dict = img2id(data_path)
    img_dot_name_dict = img_dot_name_create(img2id_dict)


    lines = excel_read(excel_path)
    emotion_dict = emotion_label_calculate(lines) 
    print(emotion_dict)
    lines = format_modify(lines, img2id_dict, img_dot_name_dict)
    data2json(lines, img2id_dict, img_dot_name_dict, emotion_dict)
    
    '''
    # 表情特征处理
    ckpt_path = 'ckpt/classifier_ckpt/model.bin'
    model = EfficientNet.from_pretrained('efficientnet-b0')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model']) 
    # print(model)
    image_process(data_path, model, img2id_dict)
    '''
