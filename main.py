import re
import en_ch_matcher
from nltk.tokenize import word_tokenize
import os
import difflib
import text_api_processor

with open("data.txt", "r+") as file:
    content = file.readlines()  # 读取所有行
    file_index = content[0]

en_file = f'batching/input_data/{file_index}_en.txt'
cn_file = f'batching/input_data/{file_index}_cn.txt'
output_file = f'batching/output/{file_index}_main.srt'
output_file2 = f'batching/output/{file_index}_en.srt'
output_file3 = f'batching/output/{file_index}_cn.srt'
vtt_file = f'batching/input_data/{file_index}.vtt'
html_file = f'batching/output/{file_index}_diff.html'

# en_file = 'input_data/en_book.txt'
# cn_file = 'input_data/cn_book.txt'
# output_file = 'output/output.srt'
# output_file2 = 'output/output2.srt'
# output_file3 = 'output/output3.srt'
# vtt_file = 'input_data/sub.vtt'
# html_file = 'output/diff.html'

max_len_per_sentence = 38
GET_EMBEDDINGS_FROM_FILE = False
# GET_EMBEDDINGS_FROM_FILE = True
other_words = []


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
        book_sentences = en_ch_matcher.split_en_text(file.read())
        book_words = []
        for sentence in book_sentences:
            for word in word_tokenize(sentence):
                book_words.append(word.lower())
        book = []
        for word in book_words:
            book.append({'text' : word.rstrip('\n'), 'start' : None, 'end' : None})

    # diff = difflib.ndiff(subtitle_words, book_words)
    with open(html_file, "w", encoding='utf-8') as f:
        f.write(difflib.HtmlDiff().make_file(subtitle_words, book_words))

    matcher = difflib.SequenceMatcher(None, subtitle_words, book_words)
    matches = matcher.get_matching_blocks()
    for idx, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes()):
        # print(f"{tag}: {i1+1} to {i2} -> {j1+1} to {j2}")   
        if tag == 'equal':
            for i in range(i1, i2):
                j = i - i1 + j1
                book[j]['start'] = subtitle[i]['start']
                book[j]['end'] = subtitle[i]['end']


        elif tag == 'replace':
            book[j1]['start'] = subtitle[i1]['start']
            book[j1]['end'] = subtitle[i1]['end']
            if i2 - i1 > 3:
                try:
                    other_word_text = ""
                    for word in subtitle[i1:i2+1]:
                        other_word_text += " " + word['text']
                    other_word_start = subtitle[i1]['start']
                    other_word_end = subtitle[i2]['end']
                    other_words.append({"text":other_word_text, "start":other_word_start, "end":other_word_end})

                except:
                    pass


        elif tag == 'insert':
            try:
                next_matcher = matcher.get_opcodes()[idx + 1]
                if next_matcher[0] == ['delete']:
                    book[i2]['start'] = subtitle[next_matcher[1]]['start']
                    book[i2]['end'] = subtitle[next_matcher[1]]['end']
            except:
                pass


        elif tag == 'delete':
            book[j1]['start'] = subtitle[i1]['start']
            book[j1]['end'] = subtitle[i1]['end']
            if i2 - i1 > 3:
                try:
                    other_word_text = ""
                    for word in subtitle[i1:i2+1]:
                        other_word_text += " " + word['text']
                    other_word_start = subtitle[i1]['start']
                    other_word_end = subtitle[i2]['end']
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
    return book





def get_timing():
    book_sentences_timing = [{'text' : match[3], 'start' : None, 'end' : None} for match in matches]
    matches_words = []
    matches_words_to_sentence_index = []
    for i, match in enumerate(matches):
        en_sentence = match[2]
        words = word_tokenize(en_sentence)
        for word in words:
            matches_words.append(word.lower())
            matches_words_to_sentence_index.append(i)

    timing_book_words = []
    for word in timing_book:
        timing_book_words.append(word['text'])


    # with open("try.html", "w", encoding='utf-8') as f:
    #     f.write(difflib.HtmlDiff().make_file(matches_words, timing_book_words))

    matcher = difflib.SequenceMatcher(None, matches_words, timing_book_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # print(f"{tag}: {i1+1} to {i2} -> {j1+1} to {j2}")   
        if tag == 'equal':
            for i in range(i1, i2):
                j = i - i1 + j1
                # book_sentences_timing[j]['start'] = subtitle[i]['start']
                # book_sentences_timing[j]['end'] = subtitle[i]['end']
                sentence_index = matches_words_to_sentence_index[i]
                if book_sentences_timing[sentence_index]['start'] == None:
                    book_sentences_timing[sentence_index]['start'] = timing_book[j]['start']

                book_sentences_timing[sentence_index]['end'] = timing_book[j]['end']


    for idx in range(len(book_sentences_timing)):
        if book_sentences_timing[idx]['start'] == None:
            try:
                book_sentences_timing[idx]['start'] = book_sentences_timing[idx - 1]['start']
                book_sentences_timing[idx]['end'] = book_sentences_timing[idx - 1]['end']
            except:
                pass

        if book_sentences_timing[idx]['end'] == None:
            book_sentences_timing[idx]['end'] = book_sentences_timing[idx]['start']



    return book_sentences_timing



if __name__ == '__main__':
    subtitle = read_vtt(vtt_file)
    subtitle_words = []

    with open('data/subtitle.txt', 'w', encoding='utf-8') as file:
        for word in subtitle:
            file.write(word["text"] + '\n')
            subtitle_words.append(word["text"])

    timing_book = book_timing(en_file, subtitle_words)

    other_words_translate = text_api_processor.process_text_thread(other_words, 2)
    # print(other_words_translate)

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
                en_idx = cn_idx = 0
                # idx_match_pattern = r'([\d*, \d*] [\d*, \d*]'
                # idx_match = re.match(idx_match_pattern, lines[idx*5].strip('\n'))
                
                # en_idx, cn_idx = idx_match.group(0), idx_match.group(1)
                # en_idx, cn_idx = int(en_idx), int(cn_idx)
                matches.append((en_idx, cn_idx, lines[idx*5+1], lines[idx*5+2], float(lines[idx*5+3])))
                
    else:
        matches = en_ch_matcher.match_sentences(english_text, chinese_text, max_len_per_sentence)
        with open('data/matches.txt', 'w', encoding='utf-8') as file:
            for en_index, cn_index, en_sentence, cn_sentence, similarity in matches:
                file.write(f"{en_index} {cn_index}\n{en_sentence}\n{cn_sentence}\n{similarity}\n\n")
                

    with open(output_file, 'w', encoding='utf-8') as f:
        matches_list = get_timing()

        for i, match in enumerate(matches_list):
            text = match['text']
            start_time = match['start']
            end_time = match['end']

            subtitle = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
            f.write(subtitle)


    with open(output_file2, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(other_words):
            start_time = sentence["start"]
            end_time = sentence["end"]
            text = sentence["text"]

            subtitle = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
            f.write(subtitle)


    with open(output_file3, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(other_words_translate):
            start_time = sentence["start"]
            end_time = sentence["end"]
            text = sentence["text"]

            subtitle = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
            f.write(subtitle)