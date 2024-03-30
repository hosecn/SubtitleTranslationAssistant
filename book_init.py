import re
import nltk

filename = 'en2.txt'
outfilename = 'book.txt'

with open(filename, 'r', encoding='utf-8') as file:
    text = file.read().lower()

tokens = nltk.word_tokenize(text)

with open(outfilename, 'w', encoding='utf-8') as file:
    file.write('\n'.join(tokens))