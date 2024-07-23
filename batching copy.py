import re
import os
import difflib
from nltk.tokenize import word_tokenize
import en_ch_matcher
import text_api_processor
from main import get_timing

# 文件路径
en_file = 'input_data/en_book.txt'
cn_file = 'input_data/cn_book.txt'
vtt_folder = 'input_data/vtt_files/'
output_folder = 'output/'

# 文件输出路径
output_file = os.path.join(output_folder, 'output.srt')
html_file = os.path.join(output_folder, 'diff.html')

# 参数设置
max_len_per_sentence = 38
GET_EMBEDDINGS_FROM_FILE = False

other_words = []

def read_vtt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    words_lines = []
    for i in range(2, len(lines)):
        line = lines[i].lower()
        time_match_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})'
        time_match = re.match(time_match_pattern, lines[i - 2])
        first_word_match = re.match(r"(\b[a-zA-Z'-]+\b)(<.*>.*)", line)

        if time_match and first_word_match:
            words_lines.append('<' + time_match.group(1) + '><c> ' + first_word_match.group(1) \
                + '</c>' + first_word_match.group(2) + '<' + time_match.group(2) + '>')

    words = []
    for line in words_lines:
        parts = re.split(r'[<>]', line)
        for j in range(len(parts) // 6):
            words.append({
                'text': parts[j * 6 + 4][1:],
                'start': parts[j * 6 + 1],
                'end': parts[j * 6 + 7]
            })

    return words

def book_timing(book_file, subtitle_words):
    with open(book_file, 'r', encoding='utf-8') as file:
        book_sentences = en_ch_matcher.split_en_text(file.read())
        book_words = [word.lower() for sentence in book_sentences for word in word_tokenize(sentence)]
        book = [{'text': word.rstrip('\n'), 'start': None, 'end': None} for word in book_words]

    with open(html_file, "w", encoding='utf-8') as f:
        f.write(difflib.HtmlDiff().make_file(subtitle_words, book_words))
    matcher = difflib.SequenceMatcher(None, subtitle_words, book_words)
    for idx, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes()):
        # print(f"{tag}: {i1+1} to {i2} -> {j1+1} to {j2}")   
        if tag == 'equal':
            for i in range(i1, i2):
                j = i - i1 + j1
                book[j]['start'] = subtitle_words[i]['start']
                book[j]['end'] = subtitle_words[i]['end']


        elif tag == 'replace':
            book[j1]['start'] = subtitle_words[i1]['start']
            book[j1]['end'] = subtitle_words[i1]['end']
            if i2 - i1 > 3:
                try:
                    other_word_text = ""
                    for word in subtitle_words[i1:i2+1]:
                        other_word_text += " " + word['text']
                    other_word_start = subtitle_words[i1]['start']
                    other_word_end = subtitle_words[i2]['end']
                    other_words.append({"text":other_word_text, "start":other_word_start, "end":other_word_end})

                except:
                    pass


        elif tag == 'insert':
            try:
                next_matcher = matcher.get_opcodes()[idx + 1]
                if next_matcher[0] == ['delete']:
                    book[i2]['start'] = subtitle_words[next_matcher[1]]['start']
                    book[i2]['end'] = subtitle_words[next_matcher[1]]['end']
            except:
                pass


        elif tag == 'delete':
            book[j1]['start'] = subtitle_words[i1]['start']
            book[j1]['end'] = subtitle_words[i1]['end']
            if i2 - i1 > 3:
                try:
                    other_word_text = ""
                    for word in subtitle_words[i1:i2+1]:
                        other_word_text += " " + word['text']
                    other_word_start = subtitle_words[i1]['start']
                    other_word_end = subtitle_words[i2]['end']
                    other_words.append({"text":other_word_text, "start":other_word_start, "end":other_word_end})
                except:
                    pass

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
    
    return matches, book

def get_timing(matches, book, subtitle):
    for match in matches:
        start, end, length = match
        for i in range(length):
            book[start + i]['start'] = subtitle[start + i]['start']
            book[start + i]['end'] = subtitle[start + i]['end']
    
    for idx in range(len(book)):
        if book[idx]['start'] is None:
            try:
                next_idx = next(i for i in range(idx + 1, len(book)) if book[i]['start'] is not None)
                book[idx]['start'] = book[next_idx]['start']
            except StopIteration:
                prev_idx = next(i for i in range(idx - 1, -1, -1) if book[i]['end'] is not None)
                book[idx]['end'] = book[prev_idx]['end']

    return book

def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    subtitle_words = []
    subtitle = []
    
    for vtt_file in os.listdir(vtt_folder):
        if vtt_file.endswith(".vtt"):
            vtt_path = os.path.join(vtt_folder, vtt_file)
            subtitle.extend(read_vtt(vtt_path))

    subtitle_words = [word["text"] for word in subtitle]
    
    matches, book = book_timing(en_file, subtitle_words)
    
    timed_book = get_timing(matches, book, subtitle)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(timed_book):
            text = entry['text']
            start_time = entry['start']
            end_time = entry['end']
            if start_time and end_time:
                subtitle_entry = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
                f.write(subtitle_entry)

if __name__ == '__main__':
    main()









# from main import *

# en_file = 'batching/input_data/en_book.txt'
# cn_file = 'batching/input_data/cn_book.txt'
# en_sentences_file = 'batching/data/en_sentences.txt'
# cn_sentences_file = 'batching/data/cn_sentences.txt'
# en_words_file = 'batching/data/en_words.txt'
# output_file = 'batching/output/output.srt'
# output_file2 = 'batching/output/output2.srt'
# output_file3 = 'batching/output/output3.srt'
# vtt_file = 'batching/input_data/sub.vtt'
# html_file = 'batching/output/diff.html'


# def split_book(en_file):
#     with open(en_file, 'r', encoding='utf-8') as file:
#         book_sentences = en_ch_matcher.en_split_text(file.read())
#     with open(en_sentences_file, 'r', encoding='utf-8') as file:
#         for word in word_tokenize(sentence):
#             file.write(word.lower() + '\n')


# if not os.path.exists(en_sentences_file):
#     split_book(en_file)

# files = sorted(os.listdir("./batching/vtt"))
# for file_index, vtt_file in enumerate(files):
#     if not vtt_file.endswith(".vtt"):
#         continue

#     subtitle = read_vtt(vtt_file)
#     subtitle_words = []

#     with open(f'batching/data/subtitle{file_index}.txt', 'w', encoding='utf-8') as file:
#         for word in subtitle:
#             file.write(word["text"] + '\n')
#             subtitle_words.append(word["text"])

    
#     timing_book = book_timing(en_file, subtitle_words)

#     other_words_translate = text_api_processor.process_text_thread(other_words, 2)
#     # print(other_words_translate)

#     with open(en_file, 'r', encoding='utf-8') as file:
#         english_text = file.read()

#     with open(cn_file, 'r', encoding='utf-8') as file:
#         chinese_text = file.read()

#     if GET_EMBEDDINGS_FROM_FILE:
#         if not os.path.exists('data/matches.txt'):
#             matches = en_ch_matcher.match_sentences(english_text, chinese_text)
#             #保存文件
#             with open('data/matches.txt', 'w', encoding='utf-8') as file:
#                 for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
#                     file.write(f"{en_index} {cn_index}\n{en_sentence}\n{cn_sentence}\n{similarity}\n\n")

#         #读取文件
#         with open('data/matches.txt', 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#             matches = []

#             for idx in range(len(lines) // 5):
#                 en_idx = cn_idx = 0
#                 # idx_match_pattern = r'([\d*, \d*] [\d*, \d*]'
#                 # idx_match = re.match(idx_match_pattern, lines[idx*5].strip('\n'))
                
#                 # en_idx, cn_idx = idx_match.group(0), idx_match.group(1)
#                 # en_idx, cn_idx = int(en_idx), int(cn_idx)
#                 matches.append((en_idx, cn_idx, lines[idx*5+1], lines[idx*5+2], float(lines[idx*5+3])))
                
#     else:
#         matches = en_ch_matcher.match_sentences(english_text, chinese_text, max_len_per_sentence)
#         with open('data/matches.txt', 'w', encoding='utf-8') as file:
#             for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
#                 file.write(f"{en_index} {cn_index}\n{en_sentence}\n{cn_sentence}\n{similarity}\n\n")
                

#     with open(output_file, 'w', encoding='utf-8') as f:
#         matches_list = get_timing()

#         for i, match in enumerate(matches_list):
#             text = match['text']
#             start_time = match['start']
#             end_time = match['end']

#             subtitle = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
#             f.write(subtitle)


#     with open(output_file2, 'w', encoding='utf-8') as f:
#         for i, sentence in enumerate(other_words):
#             start_time = sentence["start"]
#             end_time = sentence["end"]
#             text = sentence["text"]

#             subtitle = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
#             f.write(subtitle)


#     with open(output_file3, 'w', encoding='utf-8') as f:
#         for i, sentence in enumerate(other_words_translate):
#             start_time = sentence["start"]
#             end_time = sentence["end"]
#             text = sentence["text"]

#             subtitle = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
#             f.write(subtitle)