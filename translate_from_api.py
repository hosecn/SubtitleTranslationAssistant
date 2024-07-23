# -*- coding: utf-8 -*-

from googletrans import Translator

text = '''
 Past the near meadows, over the still stream, 
  Up the hill-side; and now ‘tis buried deep 
     In the next valley-glades: 
 Was it a vision, or a waking dream? 
  Fled is that music:—Do I wake or sleep?
'''
def translateEnText(text : str):
    translator = Translator()
    result = translator.translate(text, src='en', dest='zh-cn')
    return result.text

# _translator = Translator()

# def translateEnText(text : str):
#     if text == None:
#         return None
    
#     translatedText = None
#     try:
#         result = _translator.translate(text, src='en', dest='zh-cn')
#         translatedText = result.text
#     except Exception as e:
#         print("************translated error*****************")
#         print(e)

#     return translatedText


print(translateEnText(text))