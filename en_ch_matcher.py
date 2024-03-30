import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

max_len = 30

# 初始化模型和分词器
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def en_split_text(para, max_len=20):
    # 使用正则表达式进行分句，同时处理英文和中文的标点
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|。|？|！|,)\s'
    sentences = re.split(pattern, para)
    refined_sentences = []

    for sentence in sentences:
        # 检查句子长度
        if len(sentence) <= max_len:
            refined_sentences.append(sentence)
            continue

        # 对于超长句子，尝试根据逗号进一步拆分（同时考虑英文和中文的逗号）
        chunks = re.split('，|,|、', sentence)
        temp_chunk = []
        for chunk in chunks:
            if len(','.join(temp_chunk + [chunk])) > max_len and temp_chunk:
                refined_sentences.append(','.join(temp_chunk).strip())
                temp_chunk = [chunk]
            else:
                temp_chunk.append(chunk)
        # 添加最后一部分
        if temp_chunk:
            refined_sentences.append(','.join(temp_chunk).strip())

    return refined_sentences





def cn_split_text(para, max_len=20):
    para = re.sub('([。！？\?；])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    
    sentences = [item for item in para.split("\n") if item != '']


    refined_sentences = []

    for sentence in sentences:
        # 检查句子长度
        if len(sentence) <= max_len:
            refined_sentences.append(sentence)
            continue

        # 对于超长句子，尝试根据逗号进一步拆分（同时考虑英文和中文的逗号）
        chunks = re.split('，|,|、', sentence)
        temp_chunk = []
        for chunk in chunks:
            if len(','.join(temp_chunk + [chunk])) > max_len and temp_chunk:
                refined_sentences.append(','.join(temp_chunk).strip())
                temp_chunk = [chunk]
            else:
                temp_chunk.append(chunk)
        # 添加最后一部分
        if temp_chunk:
            refined_sentences.append(','.join(temp_chunk).strip())

    return refined_sentences




def get_sentence_embeddings(sentences):
    """
    批量计算句子的嵌入向量，并使用GPU加速（如果可用）。
    """
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128, return_token_type_ids=False, return_attention_mask=True).to(device)
    with torch.no_grad():  # 不计算梯度，减少内存/显存占用
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # 将嵌入向量移回CPU
    return embeddings





def match_sentences(english_text, chinese_text, window_size=3):
    """
    动态调整搜索范围以提高匹配准确度。
    """
    english_sentences = cn_split_text(english_text)
    chinese_sentences = en_split_text(chinese_text)

    english_embeddings = get_sentence_embeddings(english_sentences)
    chinese_embeddings = get_sentence_embeddings(chinese_sentences)

    matches = []
    last_matched_chi_index = -1  # 记录上一次匹配的中文句子索引

    for i, eng_emb in enumerate(english_embeddings):
        max_similarity = 0
        best_match = None
        # 动态调整搜索起始点，基于上一次匹配的中文句子索引
        start_index = max(0, last_matched_chi_index + 1)
        for j in range(start_index, len(chinese_sentences)):
            chi_emb = chinese_embeddings[j]
            similarity = cosine_similarity([eng_emb], [chi_emb])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = (i, j, english_sentences[i], chinese_sentences[j], similarity)
            # 当搜索到足够远离上次匹配位置时，停止搜索以避免过大的错位
            if j >= start_index + window_size:
                break
        
        if best_match:
            matches.append(best_match)
            last_matched_chi_index = best_match[1]  # 更新上一次匹配的中文句子索引

    # 保存结果
    with open("dynamic_flexible_matched_pairs.txt", "w", encoding="utf-8") as f:
        for match in matches:
            f.write(f"Index English: {match[0]}, Index Chinese: {match[1]}\nEnglish Sentence: {match[2]}\nChinese Sentence: {match[3]}\nSimilarity: {match[4]:.4f}\n\n")

    print(f"Processed {len(matches)} dynamically flexible matched sentence pairs.")





en_file_name = 'en_book.txt'
cn_file_name = 'cn_book.txt'

with open(en_file_name, 'r', encoding='utf-8') as file:
    english_text = file.read()

with open(cn_file_name, 'r', encoding='utf-8') as file:
    chinese_text = file.read()

# match_sentences(english_text, chinese_text)
cn = cn_split_text(chinese_text)
with open('try3.txt', 'w', encoding='utf-8') as file:
    for line in cn:
        file.write(line + '\n')

    # with open('try2.txt', 'w', encoding='utf-8') as file:
    #     file.write(str(english_sentences))