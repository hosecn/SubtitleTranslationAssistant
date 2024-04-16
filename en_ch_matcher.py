import re
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

max_len = 100
critical_value = 0.3

en_file_name = 'en_book.txt'
cn_file_name = 'cn_book.txt'

start_time = time.perf_counter()

# 初始化模型和分词器
model_path = "./model/m2.pkl"

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)





def en_split_text(para, max_len=100):
    para = re.sub('(\.{6})([^”’\'\"]) ?', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’\'\"]) ?', r"\1\n\2", para)  # 中文省略号
    para = re.sub('(?<!\b(?:Mr|Dr|Ms)\.)\.(?=\s+[A-Z]|$)', '.\n', para)
    para = re.sub('([。！？；;\?\!]+ ?[”’]? ?)([^，。！？\?\!”’\'\"])', r'\1\n\2', para)
    para = re.sub('([^。！？；;\?\!]+ ?[”’]? ?)(”’\'\")', r'\1\n', para)
    sentences = [item for item in re.split('\n ?', para) if item != '']
    return sentences





def cn_split_text(para, max_len=20):
    para = re.sub('(\.{6})([^”’\'\"]) ?', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’\'\"]) ?', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?\!]+ ?[”’]? ?)([^，。！？\?\!”’\'\"])', r'\1\n\2', para)
    para = re.sub('([^。！？\?\!]+ ?[”’]? ?)(”’\'\")', r'\1\n', para)
    para = para.strip()  #去除首尾空格
    
    sentences = [item for item in re.split('\n', para) if item != '']

    return sentences




def get_sentence_embeddings(sentences):
    """
    计算句子的嵌入向量
    """
    return model.encode(sentences)



def get_queue():
    res_list = [[] for _ in range(4950)]
    idx = 0

    for i in range(1, 100):
        for j in range(i, 0, -1):
            res_list[idx] = (i, j)
            idx += 1

    return res_list




def split_to_short(text):
    sentences = []
    for para in text:
        para = re.sub('([,，])', r"\1\n", para)
        para = para.strip()  #去除首尾空格
    
        tmp_sentences = [(item, idx) for item in re.split('\n', para) if item != '']  

    sentences = sentences + tmp_sentences

    return sentences





def position_similarity(pos1, pos2, sigma=0.1):
    """
    根据两个句子在各自文本中的相对位置计算距离相似度。
    :param pos1: 第一个句子的相对位置（0到1之间的小数）
    :param pos2: 第二个句子的相对位置（0到1之间的小数）
    :param sigma: 高斯函数的标准差，控制相似度随距离变化的速度
    :return: 距离的相似度
    """
    relative_distance = abs(pos1 - pos2)
    return np.exp(-relative_distance**2 / (2 * sigma**2))





def match_sentences(english_text, chinese_text, window_size=3):
    """
    匹配中英文句子
    """
    en_sentences = en_split_text(english_text)
    cn_sentences = cn_split_text(chinese_text)

    if input("从本地读取?(N/Y) >>>") in ['n', 'N']:
        en_embeddings = [get_sentence_embeddings([sentence])[0] for sentence in tqdm(en_sentences, desc="正在处理英文句子")]
        cn_embeddings = [get_sentence_embeddings([sentence])[0] for sentence in tqdm(cn_sentences, desc="正在处理中文句子")]
        en_embeddings = np.array(en_embeddings)
        cn_embeddings = np.array(cn_embeddings)

        np.savetxt('en_embeddings.txt', en_embeddings)
        np.savetxt('cn_embeddings.txt', cn_embeddings)

    # 从文件中读取数组数据
    en_embeddings = np.loadtxt('en_embeddings.txt')
    cn_embeddings = np.loadtxt('cn_embeddings.txt')


    similaritys = en_embeddings @ cn_embeddings.T
    matches = []
    last_cn_index = -1  # 记录上一次匹配的中文句子索引
    last_en_index = -1  # 记录上一次匹配的英文句子索引
    queue = get_queue()
    
    # 初次匹配
    for idx_en in range(len(en_sentences)):
        if idx_en < last_en_index:
            continue

        for d_en, d_cn in queue:
            if last_en_index + d_en >= len(en_sentences):
                continue
            if last_cn_index + d_cn >= len(cn_sentences):
                continue
            
            en_sentance = en_sentences[last_en_index + d_en]
            cn_sentence = cn_sentences[last_cn_index + d_cn]
            similarity = similaritys[last_en_index + d_en][last_cn_index + d_cn]
            if similarity > critical_value:
                last_en_index += d_en   
                last_cn_index += d_cn
                matches.append((last_en_index, last_cn_index, en_sentance, cn_sentence, similarity))
                break


    # 二次匹配

    last_benchmark_en_idx = matches[0][last_en_index]
    last_benchmark_cn_idx = matches[0][last_cn_index]
    for benchmark_en_idx, benchmark_cn_idx, en_sentance, cn_sentence, similarity in matches[1:]:
        slice_en_sentances = en_sentance[benchmark_en_idx : last_benchmark_en_idx]
        slice_cn_sentances = cn_sentence[benchmark_cn_idx : last_benchmark_cn_idx]

        cn_short_sentences = split_to_short(slice_cn_sentances)
        en_short_sentences = split_to_short([en_sentance])
    # # 首先，初始化一个与 similarity 同形状的矩阵来存储位置相似度
    # position_sim_matrix = np.zeros_like(similaritys)

    # # 计算位置相似度矩阵
    # en_len = len(en_sentences)
    # cn_len = len(cn_sentences)

    # for i in range(en_len):
    #     for j in range(cn_len):
    #         en_pos = i / en_len
    #         cn_pos = j / cn_len
    #         position_sim_matrix[i, j] = position_similarity(en_pos, cn_pos)

    # # 计算最终的相似度矩阵
    # final_sim_matrix = similarity * position_sim_matrix

            


        for en_idx, row in enumerate(similaritys):
            cn_idx = row.argmax()
            max_value = row.max()
            matches.append((en_idx, cn_idx, en_sentences[en_idx], cn_sentences[cn_idx], max_value))
        
    return matches





en_file_name = 'en_book.txt'
cn_file_name = 'cn_book.txt'

with open(en_file_name, 'r', encoding='utf-8') as file:
    english_text = file.read()

with open(cn_file_name, 'r', encoding='utf-8') as file:
    chinese_text = file.read()

cn = cn_split_text(chinese_text)
en = en_split_text(english_text)
with open('try1.txt', 'w', encoding='utf-8') as file:
    for line in cn:
        file.write(line + '\n')
with open('try2.txt', 'w', encoding='utf-8') as file:
    for line in en:
        file.write(line + '\n')

matches = match_sentences(english_text, chinese_text)
with open('try3.txt', 'w', encoding='utf-8') as file:
    for en_index, cn_index, en_sentance, cn_sentance, similarity in matches:
        file.write(f"{en_index} {cn_index}\n{en_sentance}\n{cn_sentance}\n{similarity}\n\n")

    

end_time = time.perf_counter()

print(f'用时:{end_time - start_time:.2f}秒')
