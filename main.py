import re
import en_ch_matcher
from nltk.tokenize import word_tokenize
import json
import os
import difflib

en_file = '1_en.txt'
cn_file = '1_cn.txt'
# GET_EMBEDDINGS_FROM_FILE = False
GET_EMBEDDINGS_FROM_FILE = True


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
            book.append({'text' : word.rstrip('\n'), 'start' : None, 'end' : None})

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

    with open('timing_book.txt', 'w', encoding='utf-8') as file:
        for line in book:
            file.write(str(line) + '\n')
    return book
            
    



subtitle = read_vtt('sub.vtt')

with open('subtitle.txt', 'w', encoding='utf-8') as file:
    for word in subtitle:
        file.write(word["text"] + '\n')

timing_book = book_timing('book.txt', 'report.txt')

with open(en_file, 'r', encoding='utf-8') as file:
    english_text = file.read()

with open(cn_file, 'r', encoding='utf-8') as file:
    chinese_text = file.read()

if GET_EMBEDDINGS_FROM_FILE:
    if not os.path.exists('matches.txt'):
        matches = en_ch_matcher.match_sentences(english_text, chinese_text, timing_book)
        #保存文件
        with open('matches.txt', 'w', encoding='utf-8') as file:
            for en_index, cn_index, en_sentance, cn_sentence, similarity in matches:
                file.write(f"{en_index} {cn_index}\n{en_sentance}\n{cn_sentence}\n{similarity}\n\n")

    #读取文件
    with open('matches.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        matches = []

        for idx in range(len(lines) // 5):
            en_idx, cn_idx = lines[idx*5].strip('\n').split()
            en_idx, cn_idx = int(en_idx), int(cn_idx)
            matches.append((en_idx, cn_idx, lines[idx*5+1], lines[idx*5+2], float(lines[idx*5+3])))
            
else:
    matches = en_ch_matcher.match_sentences(english_text, chinese_text, timing_book)





def get_timing():
    timing_words = [item['text'] for item in timing_book]
    book_sentences = [en_sentence.lower() for _, _, en_sentence, _, _ in matches]
    book_words = []

    for sentence in book_sentences:
        for word in word_tokenize(sentence):
            book_words.append(word)

    book_words_timing = []
    book_sentences_timing = []
    diff_result = difflib.ndiff(book_words, timing_words)
    with open('diff.txt', 'w', encoding='utf-8') as file:
        for line in diff_result:
            file.write(line + '\n')


    flag = 0
    for idx, line in enumerate(diff_result):
        if flag:
            flag = 0
            continue

        the_word = line[2:]

        if line.startswith(' '):
            book_words_timing.append((the_word, timing_book[0]['start'], timing_book[0]['end']))
            matches.pop(0)
            timing_book.pop(0)

        elif line.startswith('-'):
            try:
                if diff_result[idx+1].startswith('+'):
                    book_words_timing.append((the_word, timing_book[1]['start'], timing_book[1]['end']))
                    matches.pop(0)
                    timing_book.pop(0)
                    flag = 1

                else:
                    book_words_timing.append((the_word, None, None))
            
            except:
                pass
        
        else:
            book_words_timing.append((the_word, None, None))

    return book_words_timing



            

with open('output.srt', 'w') as f:
    matches_list = get_timing()

    for i, match in enumerate(matches_list):
        text = match['text']
        start_time = match['start']
        end_time = match['end']

        subtitle = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
        f.write(subtitle)
