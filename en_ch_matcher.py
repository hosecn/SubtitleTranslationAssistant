import re
import os
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import text_api_processor

import concurrent.futures
import threading

first_match_min_value = 0.6
third_match_min_value = 0 
forth_match_min_value = 0
# GET_EMBEDDINGS_FROM_FILE = False
GET_EMBEDDINGS_FROM_FILE = True
max_lenth = 33

en_file = 'input_data/en_book.txt'
cn_file = 'input_data/cn_book.txt'

start_time = time.perf_counter()

model_name = 'distiluse-base-multilingual-cased-v2'
# model_name = 'intfloat/multilingual-e5-small'
# model_name = 'mixedbread-ai/mxbai-embed-large-v1'


# 初始化模型和分词器
model_path = f"./model/{model_name}.pkl"

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = SentenceTransformer(model_name)
    try:
        os.makedirs(f"./model/")
    except:
        pass

    if '/' in model_name:
        try:
            os.makedirs(f"./model/{model_name[:model_name.rfind('/')]}")
        except:
            pass

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)





def en_split_text(para):
    para = re.sub('(\.{6})([^”’\'\"]) ?', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’\'\"]) ?', r"\1\n\2", para)  # 中文省略号
    para = re.sub('(?<!\b(?:Mr|Dr|Ms)\.)\.(?=\s+[A-Z]|$)', '.\n', para)
    para = re.sub('([。！？；;\?\!]+ ?[”’]? ?)([^，。！？\?\!”’\'\"])', r'\1\n\2', para)
    para = re.sub('([^。！？；;\?\!]+ ?[”’]? ?)(”’\'\")', r'\1\n', para)
    para = para.strip()  #去除首尾空格
    l1 = ['', '-', '***']
    sentences = [item for item in re.split('\n ?', para) if item not in l1]
    return sentences





def cn_split_text(para):
    para = re.sub('(\.{6})([^”’\'\"]) ?', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’\'\"]) ?', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。;；！？\?\!]+ ?[”’]? ?)([^，。！？\?\!”’\'\"])', r'\1\n\2', para)
    para = re.sub('([^。;；！？\?\!]+ ?[”’]? ?)(”’\'\")', r'\1\n', para)
    para = para.strip()  #去除首尾空格
    
    sentences = [item for item in re.split('\n', para) if item != '']

    return sentences




def get_similaritys(en_sentences, cn_sentences):
    """
    计算句子之间的相似度
    """
    global en_embeddings, cn_embeddings
    if not GET_EMBEDDINGS_FROM_FILE:
        
        en_embeddings = [get_sentence_embeddings([sentence])[0] for sentence in tqdm(en_sentences, desc="正在预处理英文句子")]
        cn_embeddings = [get_sentence_embeddings([sentence])[0] for sentence in tqdm(cn_sentences, desc="正在预处理中文句子")]
        en_embeddings = np.array(en_embeddings)
        cn_embeddings = np.array(cn_embeddings)

        np.savetxt('data/en_embeddings.txt', en_embeddings)
        np.savetxt('data/cn_embeddings.txt', cn_embeddings)

    # 从文件中读取数组数据
    en_embeddings = np.loadtxt('data/en_embeddings.txt')
    cn_embeddings = np.loadtxt('data/cn_embeddings.txt')

    similaritys = cosine_similarity(en_embeddings, cn_embeddings)

    return similaritys





def get_sentence_embeddings(sentences):
    """
    计算句子的嵌入向量
    """
    return model.encode(sentences)





def get_queue():
    res_list = [[] for _ in range(4950)]
    res_list = [[] for _ in range(4950)]
    idx = 0

    for i in range(1, 100):
        for j in range(i, 0, -1):
            res_list[idx] = (i, j)
            idx += 1

    return res_list



    

def match(en_sentences, cn_sentences, similaritys, min_value, serial_num=1):

    matches = []

    last_cn_index = -1  # 记录上一次匹配的中文句子索引
    last_en_index = -1  # 记录上一次匹配的英文句子索引
    queue = get_queue()

    for idx_en in range(len(en_sentences)):
        if idx_en < last_en_index:
            continue

        for d_en, d_cn in queue:
            if last_en_index + d_en >= len(en_sentences):
                continue
            if last_cn_index + d_cn >= len(cn_sentences):
                continue
            
            en_sentence = en_sentences[last_en_index + d_en]
            cn_sentence = cn_sentences[last_cn_index + d_cn]
            similarity = similaritys[last_en_index + d_en][last_cn_index + d_cn]
            if similarity > min_value:
                last_en_index += d_en   
                last_cn_index += d_cn
                matched = ([last_en_index + 1, last_en_index + 1], [last_cn_index + 1, last_cn_index + 1], en_sentence, cn_sentence, similarity, serial_num)
                matches.append(matched)
                break
    
    return matches




def get_1_similaritys(en_sentences, cn_sentences):
    en_embeddings = [get_sentence_embeddings([en_sentences])[0]]
    cn_embeddings = [get_sentence_embeddings([cn_sentences])[0]]
    en_embeddings = np.array(en_embeddings)
    cn_embeddings = np.array(cn_embeddings)
    similaritys = cosine_similarity(en_embeddings, cn_embeddings)[0][0]
    return similaritys




def rematch(en_index, cn_index, last_en_index, last_cn_index, sentence_list):
    en_sentence, cn_sentence, back_en_sentence, back_cn_sentence, front_en_sentence, front_cn_sentence = sentence_list

    back_en = back_en_sentence + en_sentence
    back_cn = back_cn_sentence + cn_sentence
    front_en = en_sentence + front_en_sentence
    front_cn = cn_sentence + front_cn_sentence

    similarity_1 = get_1_similaritys(back_en, back_cn)
    similarity_2 = get_1_similaritys(front_en, front_cn)

    if similarity_1 > similarity_2:
        merged_matches.pop()
        merged_matches.append(([last_en_index + 1, en_index[0] - 1], [last_cn_index + 1, cn_index[0] - 1], back_en, back_cn, similarity_1))
        merged_matches.append((en_index, cn_index, front_en_sentence, front_cn_sentence, 0))
    
    else:
        merged_matches.append((en_index, cn_index, front_en, front_cn, similarity_2))





def split_to_short(para):
    """
    尝试将长度过长的句子按照中英文对应的方式进行分割，分割位置尽量在长句子的中间位置。
    """
    para = re.sub('(\.{6})([^”’\'\"]) ?', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’\'\"]) ?', r"\1\n\2", para)  # 中文省略号
    para = re.sub('(?<!\b(?:Mr|Dr|Ms)\.)\.(?=\s+[A-Z]|$)', '.\n', para)
    para = re.sub('([。;；！？\?\!，,]+ ?[”’]? ?)([^，。！？\?\!”’\'\"])', r'\1\n\2', para)
    para = re.sub('([^。;；！？\?\!，,]+ ?[”’]? ?)(”’\'\")', r'\1\n', para)
    para = para.strip()  #去除首尾空格
    
    sentences = [item for item in re.split('\n', para) if item != '']

    return sentences






def merge_unmatched_sentences(matches, en_sentences, cn_sentences):
    global merged_matches
    merged_matches = []
    last_en_index = 0
    last_cn_index = 0

    for en_index_list, cn_index_list, en_sentence, cn_sentence, similarity, serial_num in tqdm(matches, desc="正在进行二次匹配"):
        en_index = en_index_list[0]
        en_index_end = en_index_list[1]
        cn_index = cn_index_list[0]
        cn_index_end = cn_index_list[1]
        # 合并上一个匹配句子与当前匹配句子之间的未匹配句子
        unmatched_en_list = en_sentences[last_en_index:en_index-1]
        unmatched_cn_list = cn_sentences[last_cn_index:cn_index-1]
        if unmatched_en_list == [] and unmatched_cn_list == []:
            merged_matches.append((en_index_list, cn_index_list, en_sentence, cn_sentence, similarity))
            last_en_index = en_index_end
            last_cn_index = cn_index_end
            continue
        unmatched_en_sentence = ' '.join(unmatched_en_list)
        unmatched_cn_sentence = ''.join(unmatched_cn_list)
        en_new_idx_list = [last_en_index + 1, en_index - 1]
        cn_new_idx_list = [last_cn_index + 1, cn_index - 1]

        if unmatched_en_list and unmatched_cn_list:
            merged_matches.append((en_new_idx_list, cn_new_idx_list, unmatched_en_sentence, unmatched_cn_sentence, 0))
            merged_matches.append((en_index_list, cn_index_list, en_sentence, cn_sentence, similarity))

            # en_short = split_to_short(unmatched_en_sentence)
            # cn_short = split_to_short(unmatched_cn_sentence)
            # output_text = f'english_text = [0:"{en_short[0]}"'
            # for i in range(1, len(en_short)):
            #     output_text = f'{output_text},{str(i)}:"{en_short[i]}"'
            
            # output_text = f'{output_text}], chinese_text = ["{cn_short[0]}"'
            # for i in range(1, len(cn_short)):
            #     output_text = f'{output_text},{str(i)}:"{cn_short[i]}"'
            # output_text = f'{output_text}]'
                
            # print(output_text)

        else:
            last_en_sentence = en_sentences[last_en_index-1]
            last_cn_sentence = cn_sentences[last_cn_index-1]
            front_en_sentence = en_sentences[en_index-1]
            front_cn_sentence = cn_sentences[cn_index-1]
            sentences_list = [unmatched_en_sentence, unmatched_cn_sentence, last_en_sentence, 
                              last_cn_sentence, front_en_sentence, front_cn_sentence]

            if unmatched_en_list and not unmatched_cn_list:
                rematch(en_new_idx_list, cn_new_idx_list, last_en_index, last_cn_index, sentences_list)

            if unmatched_cn_list and not unmatched_en_list:
                rematch(en_new_idx_list, cn_new_idx_list, last_en_index, last_cn_index, sentences_list)

        last_en_index = en_index
        last_cn_index = cn_index

    # 合并最后一个匹配句子之后的未匹配句子
    unmatched_en_list = en_sentences[last_en_index+1:]
    unmatched_cn_list = cn_sentences[last_cn_index+1:]
    if unmatched_en_list or unmatched_cn_list:
        en_sentence = ' '.join(unmatched_en_list)
        cn_sentence = ''.join(unmatched_cn_list)
        en_new_index_list = [last_en_index + 1, len(en_sentences) + 1]
        cn_new_index_list = [last_cn_index + 1, len(cn_sentences) + 1]
        merged_matches.append((en_new_index_list, cn_new_index_list, en_sentence, cn_sentence, 0))

    return merged_matches






def match_embeddings(similaritys_slice, cn_index, min_value=0.5):
    matched_indices = []
    last_cn_index = -1
    for idxi, i in enumerate(similaritys_slice):
        max_similarity = 0
        max_cn_index = 0
        for idx, j in enumerate(i[cn_index[0]:cn_index[1] + 1]):
            if j > max_similarity:
                max_similarity = j
                max_cn_index = idx
        matched_indices.append((idxi, idxi, max_cn_index, max_cn_index, max_similarity))

    return matched_indices





# 初始化计数器、锁和时间戳
call_count = 0
lock = threading.Lock()
last_reset_time = time.time()

def safe_process_text(input_text):
    global call_count, last_reset_time
    
    with lock:  # 获取锁以安全地访问和修改共享资源
        current_time = time.time()
        
        # 检查是否需要重置计数器
        if current_time - last_reset_time >= 60:
            call_count = 0
            last_reset_time = current_time
        
        # 检查是否达到调用上限
        while call_count >= 100:
            remaining_time = 60 - (current_time - last_reset_time)
            if remaining_time > 0:
                print(f"达到调用上限，等待 {remaining_time:.2f} 秒...")
                time.sleep(remaining_time)  # 等待直到下一分钟或剩余时间
                current_time = time.time()  # 更新当前时间
            else:  # 防止浮点运算误差导致的负数
                time.sleep(1)
            
            # 再次检查是否已经重置了计数器
            if current_time - last_reset_time >= 60:
                call_count = 0
                last_reset_time = current_time
                
        call_count += 1  # 增加调用计数

    # 实际的处理文本逻辑
    completion = text_api_processor.process_text(input_text)
    completion = re.sub(r"\\\\", r'\\', completion)

    return completion





def third_match(matches, similaritys, max_lenth):
    new_matches = []
    requiste_texts = []
    results = []

    for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
        if len(cn_sentence) <= max_lenth:
            # new_matches.append((en_index, cn_index, en_sentence, cn_sentence, similarity))
            continue
        
        else:
            requiste_texts.append(f"english: ```{en_sentence}```, chinese: ```{cn_sentence}```")
    
    results = [None] * len(requiste_texts)  # 创建一个结果列表，用于存储处理后的文本

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(safe_process_text, text): idx for idx, text in tqdm(enumerate(requiste_texts), desc="正在进行三次匹配")
        }
        
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]  # 获取future的原始索引
            output_text = future.result()
            output_text = output_text.lstrip('```json').strip('```')
            output_text = re.sub(r'"english":[^\",\s][^"](.*),?\n', r'"english":"$1"\n', output_text)
            output_text = re.sub(r'"chinese":[^\",\s][^"](.*),?\n', r'"chinese":"$1"\n', output_text)
            output_text = re.sub(r"'english': ?'(.*)', ", r'"english": "$1"', output_text)
            output_text = re.sub(r"'chinese': ?'(.*)'(,? ?)", r'"chinese": "$1"$2', output_text)
            
            try:
                results[index] = json.loads(rf"{output_text}")# 将结果存储在正确的位置
            except:
                print(f"处理文本时出错:{output_text}")
                results[index] = {"english":"", "chinese":""}
                

    results_idx = 0
    for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
        if len(cn_sentence) <= max_lenth:
            new_matches.append((en_index, cn_index, en_sentence, cn_sentence, similarity))
        else:
            
                result = results[results_idx]
                # print(result)
                results_idx += 1
                num = 0
                try:
                    for item in result:
                        new_matches.append((en_index, cn_index, result[item]["english"], result[item]["chinese"], 1))
                        num += 1
                except:
                    print(f"处理文本时出错")
                    while num > 0:
                        new_matches.pop()
                        results_idx -= 1

                    new_matches.append((en_index, cn_index, en_sentence, cn_sentence, 1))

    return new_matches


def forth_match(matches, max_lenth):
    new_matches = []
    requiste_texts = []
    results = []

    for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
        if len(cn_sentence) <= max_lenth:
            # new_matches.append((en_index, cn_index, en_sentence, cn_sentence, similarity))
            continue
        
        else:
            requiste_texts.append(f"请分割成以逗号分割的短句，每句中文不超过{str(max_lenth)}个字。请保留每个短句后面的标点，如逗号。english: ```{en_sentence}```, chinese: ```{cn_sentence}```")
    
    results = [None] * len(requiste_texts)  # 创建一个结果列表，用于存储处理后的文本

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(safe_process_text, text): idx for idx, text in tqdm(enumerate(requiste_texts), desc="正在进行三次匹配")
        }
        
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]  # 获取future的原始索引
            output_text = future.result()
            output_text = output_text.lstrip('```json').strip('```')
            output_text = re.sub(r'"english":[^\",\s][^"](.*),?\n', r'"english":"$1"\n', output_text)
            output_text = re.sub(r'"chinese":[^\",\s][^"](.*),?\n', r'"chinese":"$1"\n', output_text)
            output_text = re.sub(r"'english': ?'(.*)', ", r'"english": "$1"', output_text)
            output_text = re.sub(r"'chinese': ?'(.*)'(,? ?)", r'"chinese": "$1"$2', output_text)
            output_text = re.sub(r"'chinese': ?'(.*)'(,? ?)", r'"chinese": "$1"$2', output_text)
            
            try:
                results[index] = json.loads(rf"{output_text}")# 将结果存储在正确的位置
            except:
                print(f"处理文本时出错")
                results[index] = {"english":"", "chinese":""}
                

    results_idx = 0
    for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
        if len(cn_sentence) <= max_lenth:
            new_matches.append((en_index, cn_index, en_sentence, cn_sentence, similarity))
        else:
            
                result = results[results_idx]
                # print(result)
                results_idx += 1
                num = 0
                try:
                    for item in result:
                        new_matches.append((en_index, cn_index, result[item]["english"], result[item]["chinese"], 1))
                        num += 1
                except:
                    print(f"处理文本时出错")
                    while num > 0:
                        new_matches.pop()
                        results_idx -= 1

                    new_matches.append((en_index, cn_index, en_sentence, cn_sentence, 0.8))

    return new_matches
    # new_matches = []
    # for en_index, cn_index, en_sentence, cn_sentence, similarity in tqdm(matches, desc="正在进行四次匹配"):
    #     if len(cn_sentence) <= max_lenth:
    #         new_matches.append((en_index, cn_index, en_sentence, cn_sentence, similarity))
    #         continue

    #     if len(cn_sentence) > max_lenth:
    #         cn_sentences_list = split_to_short(cn_sentence, max_lenth)
    #         en_sentences_list = split_to_short(en_sentence, max_lenth)
        
    #     en_embeddings = [get_sentence_embeddings([sentence])[0] for sentence in en_sentences_list]
    #     cn_embeddings = [get_sentence_embeddings([sentence])[0] for sentence in cn_sentences_list]
        
    #     matched_indices = match_embeddings(en_embeddings, cn_embeddings, en_sentences_list, cn_sentences_list, third_match_min_value)
        
    #     for en_start, en_end, cn_start, cn_end, similarity in matched_indices:
    #         en_sentence = ' '.join(en_sentences_list[en_start:en_end+1])
    #         cn_sentence = ''.join(cn_sentences_list[cn_start:cn_end+1])
    #         new_matches.append((en_index, cn_index, en_sentence, cn_sentence, similarity))

    # return new_matches
    




def match_sentences(english_text, chinese_text, max_lenth = max_lenth):
    """
    匹配中英文句子
    """
    en_sentences = en_split_text(english_text)
    cn_sentences = cn_split_text(chinese_text)
    print(f'英文句数：{len(en_sentences)} 中文句数：{len(cn_sentences)}')

    similaritys = get_similaritys(en_sentences, cn_sentences)

    # 初次匹配
    matches = match(en_sentences, cn_sentences, similaritys, first_match_min_value)

    # 合并未匹配的句子
    matches = merge_unmatched_sentences(matches, en_sentences, cn_sentences)

    matches = third_match(matches, similaritys, max_lenth)

    matches = forth_match(matches, max_lenth)

    return matches




if __name__ == '__main__':

    with open(en_file, 'r', encoding='utf-8') as file:
        english_text = file.read()

    with open(cn_file, 'r', encoding='utf-8') as file:
        chinese_text = file.read()

    with open('data/cn_sentences.txt', 'w', encoding='utf-8') as file:
        for line in cn_split_text(chinese_text):
            file.write(line + '\n')

    with open('data/en_sentences.txt', 'w', encoding='utf-8') as file:
        for line in en_split_text(english_text):
            file.write(line + '\n')

    matches = match_sentences(english_text, chinese_text) # 匹配句子
    with open('data/matches.txt', 'w', encoding='utf-8') as file:
        for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
            file.write(f"{en_index} {cn_index}\n{en_sentence}\n{cn_sentence}\n{similarity}\n\n")



    end_time = time.perf_counter()
    print(f'用时:{end_time - start_time:.2f}秒')
