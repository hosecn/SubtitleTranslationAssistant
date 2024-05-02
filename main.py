import re
import en_ch_matcher
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import os
from tqdm import trange
import difflib

en_file = 'en_book.txt'
cn_file = 'cn_book.txt'



def read_vtt(filename):

    # 打开并读取VTT文件
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    words_lines = []

    for i in range(2, len(lines)):
        line = lines[i].lower()

        time_match_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})'
        time_match = re.match(time_match_pattern, lines[i - 2])

        # [a-zA-Z'-]+ 匹配一个或多个英文字母、连字符和撇号
        first_word_match = re.match(r"(\b[a-zA-Z'-]+\b)(<.*>.*)", line) 

        if time_match and first_word_match:
            words_lines.append('<' + time_match.group(1) + '><c> ' + first_word_match.group(1) \
                + '</c>' + first_word_match.group(2) + '<' + time_match.group(2) + '>')
            
        
    with open('out.vtt', 'w', encoding='utf-8') as file:
        for line in words_lines:
            file.write(line)
            file.write('\n')
            
    # 存放解析后的字幕信息
    words = []
    i = 0
    for line in words_lines:
        parts = re.split(r'[<>]', line)
        for j in range(len(parts) // 6):
            words.append({})
            words[i]['text'] = parts[j * 6 + 4][1:]
            words[i]['start'] = parts[j * 6 + 1]
            words[i]['end'] = parts[j * 6 + 7]
            i += 1

    return words





def book_timing(book_file, patch_file):
    with open(patch_file, 'r', encoding='utf-8') as file:
        patch_words = file.readlines()
    
    with open(book_file, 'r', encoding='utf-8') as file:
        book_words = file.readlines()
        book = []
        for word in book_words:
            book.append({'text' : word.rstrip('\n')})

    tmp = 0
    for patch in patch_words:
        time_match = re.match(r' *(\d+) +(\d+) (.*)', patch)
        if time_match:
            tmp += 1
            subtitle_idx = int(time_match.group(1)) - 1
            book_idx = int(time_match.group(2)) - 1
            try:
                book[book_idx]['start'] = subtitle[subtitle_idx]['start']
                book[book_idx]['end'] = subtitle[subtitle_idx]['end']
            except:
                pass

    for idx, book_list in enumerate(book):
        if 'start' not in book_list:
            book[idx]['start'] = book[idx-1]['start']
            book[idx]['end'] = book[idx-1]['end']

    return book
    # with open('timing_book.txt', 'w', encoding='utf-8') as file:
    #     for line in book:
    #         file.write(str(line) + '\n')
            
    



subtitle = read_vtt('sub.vtt')

with open('subtitle.txt', 'w', encoding='utf-8') as file:
    for word in subtitle:
        file.write(word["text"] + '\n')

timing_book = book_timing('book.txt', 'report.txt')

with open(en_file, 'r', encoding='utf-8') as file:
    english_text = file.read()

with open(cn_file, 'r', encoding='utf-8') as file:
    chinese_text = file.read()

if not os.path.exists('matches.json'):
    matches = en_ch_matcher.match_sentences(english_text, chinese_text)
    # with open('matches.json', 'w') as file:
    #     json.dump(matches, file)

# 从json文件中读取列表
# with open('matches.json', 'r') as file:
#     matches = json.load(file)





# def get_timing():
#     timing_words = [item['text'] for item in timing_book]
#     matches_sentences = [en_sentence.lower() for _, _, en_sentence, _, _ in matches]
#     matches_words = []

#     for sentence in matches_sentences:
#         for word in word_tokenize(sentence):
#             matches_words.append(word)

#     s_t = '\n'.join(timing_words)
#     s_m = '\n'.join(matches_words)

#     diff_result = difflib.ndiff(s_m.splitlines(), s_t.splitlines())
def get_timing():
    # 将matches的内容转换为方便处理的结构
    matches_words = [{'en_index': en_idx, 'en_sentence': sent.lower()} for en_idx, _, sent, _, _ in matches]
    timing_words = [{'text': item['text'], 'index': idx} for idx, item in enumerate(timing_book)]
    
    timing_indices = []  # 用于存储每个匹配英文句子在timing_book中的起始和结束索引
    
    # 遍历matches_words中的每个英文句子
    for match in matches_words:
        en_words = word_tokenize(match['en_sentence'])
        current_idx = 0  # 当前正在检查的timing_book中的单词索引
        found_start = False
        
        # 尝试找到句子的起始位置
        for en_word in en_words:
            while current_idx < len(timing_words) and timing_words[current_idx]['text'] != en_word:
                current_idx += 1
            if current_idx == len(timing_words):
                break  # 如果找不到下一个单词，则跳出循环
            if not found_start:
                start_idx = current_idx
                found_start = True
            else:
                # 更新到当前单词的索引，用于计算句子的结束位置
                current_idx += 1
                
        if found_start:  # 确保找到了句子的起始位置
            end_idx = current_idx - 1  # end_idx是最后一个匹配单词的下一个单词的索引（如果存在）
            timing_indices.append((start_idx, end_idx))
    
    return timing_indices
            





def extract_timestamps(start_idx, end_idx):
    start_time = timing_book[start_idx]['start']
    end_time = timing_book[end_idx]['end']
    return start_time, end_time





with open('output.srt', 'w') as f:
    matches_indices = get_timing()
    
    for i, (match_idx, timing_idx) in enumerate(matches_indices):
        start_idx = timing_idx
        try:
            end_idx = timing_idx + len(matches[match_idx][2].split()) - 1
        except:
            print(f"Error: {match_idx} {timing_idx}")
            continue

        if end_idx >= len(timing_book):
            continue

        start_time, end_time = extract_timestamps(start_idx, end_idx)
        
        subtitle = f"{i+1}\n{start_time} --> {end_time}\n{matches[match_idx][3]}\n\n"
        f.write(subtitle)



# with open('output.srt', 'w') as f:
#     for i, (en_index, cn_index, en_sentence, cn_sentence, similarity) in enumerate(matches):
#         en_words = word_tokenize(en_sentence)
#         en_word_durations = [entry['duration'] for entry in timing_book[timing_book_idx : timing_book_idx + len(en_words)]]

#         en_sentence_duration = sum(en_word_durations)

#         cn_start_time, cn_end_time = get_adjusted_timestamps(cn_sentence, en_sentence_duration, en_word_durations)

#         try:
#             subtitle = f"{i+1}\n{cn_start_time} --> {cn_end_time}\n{cn_sentence}\n\n"
#             f.write(subtitle)
#         except Exception as e:
#             print(f"Error at match {i}: {e}")
#             print(f"English index: {en_index}, Chinese index: {cn_index}")
#             print(f"English sentence: {en_sentence}")
#             print(f"Chinese sentence: {cn_sentence}")


# with open('output.srt', 'w') as f:
#     timing_book_idx = 0
#     start_time = 0
#     end_time = 0
#     for i, (en_index, cn_index, en_sentence, cn_sentence, similarity) in enumerate(matches):
#         for en_word_idx, en_word in enumerate(word_tokenize(en_sentence)):
#             en_word = en_word.lower()

#             while timing_book_idx < len(timing_book) and timing_book[timing_book_idx]['text'] != en_word:
#                 timing_book_idx += 1

#             if timing_book_idx >= len(timing_book):
#                 break
            
#             last_timing_book_idx = timing_book_idx

#             if en_word_idx == 0:
#                 start_time = timing_book[timing_book_idx]['start']

#             if en_word_idx == len(word_tokenize(en_sentence)) - 1:
#                 end_time = timing_book[timing_book_idx]['end']

#         try:
#             subtitle = f"{i+1}\n{start_time} --> {end_time}\n{cn_sentence}\n\n"
#             f.write(subtitle)
        
#         except:
#             print(f"Error: {en_index} {cn_index}")
