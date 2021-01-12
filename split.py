import json 
import random 
import numpy as np 
from transformers import * 
from collections import Counter 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib 
import random 

data_path = 'data/data.json' 
reference_path = 'data/LCCC-base_train.json'


# calculate the static for VEID dataset 
def dataset_static(data):
    tokenizer = BertTokenizer.from_pretrained('ckpt/mmgpt', do_lower_case=True) 

    vocab_counter = Counter() 
    utterence_list = [] 
    visual_expression_list = []
    emotion_list = []
    for dialog_id in data:
        for utterance in data[dialog_id]:
            if 'txt' in utterance.keys():
                sentence_list = tokenizer.tokenize(utterance['txt'])
                utterence_list.append(sentence_list)
                vocab_counter.update(sentence_list) 
            if 'img_id' in utterance.keys():
                visual_expression_list.append(int(utterance['img_id'])) 
            if 'emotion_id' in utterance.keys():
                emotion_list.append(int(utterance['emotion_id']))

    print('token number:', len(vocab_counter.keys()))
    print('utterance number:', len(utterence_list))
    print('Total visual expression number:', len(visual_expression_list))

    token_num = 0 
    for sentence in utterence_list:
        token_num += len(sentence)
    print('avg tokens in an utterance:', float(token_num/len(utterence_list))) 
    frequency_plot(visual_expression_list, 'visual expression id', 'blue')
    frequency_plot(emotion_list, 'emotion id', 'red')


# plot the frequency figure 
def frequency_plot(data, x_label, color):
    # 设置matplotlib正常显示中文和负号
    #matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    #matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
    # 随机生成（10000,）服从正态分布的数据
    # bin_size = len(data)
    bin_size = 40 
    data = np.array(data)
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    plt.hist(data, bins=bin_size, facecolor=color, edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel(x_label)
    # 显示纵轴标签
    plt.ylabel("frequency")
    # 显示图标题
    # plt.title("频数/频率分布直方图")
    plt.show()


# easy split for dataset
def easy_split(data, reference):
    
    reference_list = []
    for reference_sentence in reference: 
        tmp_reference = reference_sentence[0].replace(' ', '').strip()
        #print(tmp_reference)
        reference_list.append(tmp_reference) 
    
    # train: valid: test = 8:1:1
    train_id = []
    res_id = []
    train_data = {}
    valid_data = {}
    test_data = {}

    for dialog_id in data.keys():
        try: 
            if data[dialog_id][0]['txt'] in reference_list:
                train_id.append(dialog_id) 
                print(len(train_id))
            else:
                res_id.append(dialog_id)
        except:
            print(dialog_id)
            print(data[dialog_id])
    print(len(train_id))

    vaild_id = res_id[:1000]
    test_id = res_id[1000:2000]
    train_id = train_id + res_id[2000:]
    '''
    valid_id = []
    test_id = []
    for id in res_id:
        p = random.random()
        if p < 0.28:
            valid_id.append(id)
        elif p >= 0.28 and p < 0.56:
            test_id.append(id) 
        else:
            train_id.append(id)
    '''

    print('begin to split dataset!')
    for dialog_id in data.keys():
        if dialog_id in train_id:
            train_data[dialog_id] = data[dialog_id]
        elif dialog_id in vaild_id:
            valid_data[dialog_id] = data[dialog_id]
        else:
            test_data[dialog_id] = data[dialog_id]
            
    print('write data split!')
    with open('data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    with open('data/valid.json', 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=4)
    with open('data/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)





if __name__ == "__main__":
    
    with open(data_path, 'r', encoding='utf-8') as f: 
        data = json.load(f) 
    print(len(data.keys()))
    
    # dataset_static(data)

    with open(reference_path, 'r', encoding='utf-8') as f:
        reference = json.load(f) 
    
    easy_split(data, reference)



 

