import re
import en_ch_matcher
from nltk.tokenize import word_tokenize
import os
import difflib
from collections import Counter

import argparse


parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument("--en-file", help="Path to English book file", required=True)
parser.add_argument("--cn-file", help="Path to Chinese book file", required=True)
parser.add_argument("--output-file", help="Output SRT file path", required=True)
parser.add_argument("--vtt-file", help="Input VTT file path", required=True)
parser.add_argument("--html-file", help="Output diff HTML file path", required=True)
parser.add_argument("--get-embeddings-from-file", type=bool, help="Flag to get embeddings from file", required=True)

args = parser.parse_args()

en_file = args.en_file
cn_file = args.cn_file
output_file = args.output_file
vtt_file = args.vtt_file
html_file = args.html_file
GET_EMBEDDINGS_FROM_FILE = args.get_embeddings_from_file


if not os.path.isdir('./data'):
    os.mkdir('./data')

if not os.path.isdir('./output'):
    os.mkdir('./output')


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
            
        
    with open('data/out.vtt', 'w', encoding='utf-8') as file:
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





def book_timing(book_file, subtitle_words):
    with open(book_file, 'r', encoding='utf-8') as file:
        book_sentences = en_ch_matcher.en_split_text(file.read())
        book_words = []
        for sentence in book_sentences:
            for word in word_tokenize(sentence):
                book_words.append(word.lower())
        book = []
        for word in book_words:
            book.append({'text' : word.rstrip('\n'), 'start' : None, 'end' : None})

    diff = difflib.ndiff(subtitle_words, book_words)
    file = difflib.HtmlDiff().make_file(subtitle_words, book_words)
    with open(html_file, "w", encoding='utf-8') as f:
        f.write(file)

    matcher = difflib.SequenceMatcher(None, subtitle_words, book_words)
    for idx, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes()):
        # print(f"{tag}: {i1+1} to {i2} -> {j1+1} to {j2}")
        if tag == 'equal':
            for i in range(i1, i2):
                j = i - i1 + j1
                book[j]['start'] = subtitle[i]['start']
                book[j]['end'] = subtitle[i]['end']

        if tag == 'replace':
            book[j1]['start'] = subtitle[i1]['start']
            book[j1]['end'] = subtitle[i1]['end']

        # elif tag == 'insert':
        #     try:
        #         next_matcher = matcher.get_opcodes()[idx + 1]
        #         if next_matcher[0] == ['delete']:
        #             book[i2]['start'] = subtitle[next_matcher[1]]['start']
        #             book[i2]['end'] = subtitle[next_matcher[1]]['end']
        #     except:
        #         pass
        # elif tag in ['replace', 'delete', 'insert']:
        #     print(f"{tag}: {i1+1} to {i2} -> {j1+1} to {j2}")       

    for idx, book_line in enumerate(book):
        if book_line['start'] == None:
            try:
                tmp = idx + 1
                while book[tmp]['start'] == None:
                    tmp += 1
                book[idx]['start'] = book[tmp]['start']

                tmp = idx - 1
                while book[tmp]['end'] == None:
                    tmp -= 1          
                book[idx]['end'] = book[tmp]['end']
            except:
                pass

    with open('data/book_timing.txt', 'w', encoding='utf-8') as file:
        for line in book:
            file.write(str(line) + '\n')
    return book
            




subtitle = read_vtt(vtt_file)
subtitle_words = []

with open('data/subtitle.txt', 'w', encoding='utf-8') as file:
    for word in subtitle:
        file.write(word["text"] + '\n')
        subtitle_words.append(word["text"])

timing_book = book_timing(en_file, subtitle_words)

with open(en_file, 'r', encoding='utf-8') as file:
    english_text = file.read()

with open(cn_file, 'r', encoding='utf-8') as file:
    chinese_text = file.read()

if GET_EMBEDDINGS_FROM_FILE:
    if not os.path.exists('data/matches.txt'):
        matches = en_ch_matcher.match_sentences(english_text, chinese_text)
        #保存文件
        with open('data/matches.txt', 'w', encoding='utf-8') as file:
            for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
                file.write(f"{en_index} {cn_index}\n{en_sentence}\n{cn_sentence}\n{similarity}\n\n")

    #读取文件
    with open('data/matches.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        matches = []

        for idx in range(len(lines) // 5):
            en_idx, cn_idx = lines[idx*5].strip('\n').split()
            en_idx, cn_idx = int(en_idx), int(cn_idx)
            matches.append((en_idx, cn_idx, lines[idx*5+1], lines[idx*5+2], float(lines[idx*5+3])))
            
else:
    matches = en_ch_matcher.match_sentences(english_text, chinese_text)
    with open('data/matches.txt', 'w', encoding='utf-8') as file:
        for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
            file.write(f"{en_index} {cn_index}\n{en_sentence}\n{cn_sentence}\n{similarity}\n\n")





def char_tokenize(text_list):
    char_list = []
    for line in text_list:
        for char in line:
            char_list.append(char)
    return char_list




def get_dics(text_list):
    dics = {}
    for line in text_list:
        for char in line:
            dics.setdefault(char, 0)
            dics[char] += 1

    print(dics)
    return dics





def calculate_similarity(counter1, counter2):
    """计算两个Counter对象的加权Jaccard相似度"""
    intersection_sum = sum((counter1 & counter2).values())  # 交集元素的频次和
    union_sum = sum((counter1 | counter2).values())  # 并集元素的频次和
    return intersection_sum / union_sum if union_sum else 0





def get_timing():
    book_sentences_timing = []

    for match in matches:
        _, _, en_sentence, cn_sentence, _ = match
        en_words = [word.lower() for word in word_tokenize(en_sentence)]
        en_chars = char_tokenize(en_words)
        # en_count = Counter(en_sentence.lower())

        current_timing_chars = []
        current_similarity = 0

        # 初始化相似度为第一个单词的相似度
        if len(timing_book):
            for char in timing_book[0]['text']:
                current_timing_chars.append(char)

            current_similarity = calculate_similarity(Counter(current_timing_chars), Counter(en_chars))
            start = timing_book[0]['start']
            end = timing_book[0]['end']
            timing_book.pop(0)
        
        else:
            break

        # 尝试添加更多单词直到相似度不再增加
        for i in range(1, len(timing_book)):
            prev_similarity = current_similarity
            for char in timing_book[0]['text']:
                current_timing_chars.append(char)

            current_similarity = calculate_similarity(Counter(current_timing_chars), Counter(en_chars))
            if current_similarity <= prev_similarity:  # 相似度没有增加，撤销最后一个添加的单词
                break

            else:
                end = timing_book[0]['end']
                timing_book.pop(0)

        # 构建句子和时间戳

        book_sentences_timing.append({'text': cn_sentence, 'start': start, 'end': end})

    return book_sentences_timing



            

with open(output_file, 'w', encoding='utf-8') as f:
    matches_list = get_timing()

    for i, match in enumerate(matches_list):
        text = match['text']
        start_time = match['start']
        end_time = match['end']

        subtitle = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
        f.write(subtitle)
