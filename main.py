import re





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
            print(patch)
            tmp += 1
            subtitle_idx = int(time_match.group(1)) - 1
            book_idx = int(time_match.group(2)) - 1
            try:
                book[book_idx]['start'] = subtitle[subtitle_idx]['start']
                book[book_idx]['end'] = subtitle[subtitle_idx]['end']
            except:
                print(book_idx)

    with open('timing_book.txt', 'w', encoding='utf-8') as file:
        for line in book:
            file.write(str(line) + '\n')
            
    



subtitle = read_vtt('sub.vtt')

with open('subtitle.txt', 'w', encoding='utf-8') as file:
    for word in subtitle:
        file.write(word["text"] + '\n')

book_timing('book.txt', 'report.txt')