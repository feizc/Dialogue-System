import os 
import xlrd 
import json


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
def format_modify(lines, img2id):
    for i in range(len(lines)):
        lines[i][1] = lines[i][1].strip()
        if lines[i][1] != "":
            left_idx = lines[i][1].find('[')
            right_idx = lines[i][1].find(']')
            img_name = lines[i][1][left_idx+1:right_idx]
            # 表情标注错误
            if img_name not in img2id.keys():
                print(img_name) 
          
            # 把中间的表情移到后面
            if left_idx == 0:
                continue
            elif right_idx == len(lines[i][1])-1:
                continue
            else:
                tmpline = lines[i][1].split('[')[0]
                tmpline += lines[i][1].split(']')[1]
                tmpline += lines[i][1][left_idx:right_idx+1]
                lines[i][1] = tmpline
    return lines
            


# 生成json数据集文件
def data2json(lines, img2id):
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
        if 'dialogue' in lines[i][0]:
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
                    left_idx = lines[i][1].find('[')
                    right_idx = lines[i][1].find(']')
                    img_name = lines[i][1][left_idx+1:right_idx]
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
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    



if __name__ == "__main__":
    data_path = os.getcwd()
    excel_path = os.path.join(data_path, 'dialogue.xlsx')
    img2id_dict = img2id(data_path)
    lines = excel_read(excel_path)
    lines = format_modify(lines, img2id_dict)
    data2json(lines, img2id_dict)
