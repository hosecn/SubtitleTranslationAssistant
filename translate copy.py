
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

text = '''
入梦的，带不走，
初醒的，看不透。
重逢前，临别后，
拨雪寻春，烧灯续昼。
此身葬风波，还以为相忘旧山河，
你我往生客，谁才是痴狂者。
百鬼过荒城，第几次将横笛吹彻，
而此刻，又何以为歌？
是跌碎尘埃的孤魂，在天涯永夜处容身，
听谁唱世外光阴，洞中朝暮只一瞬。
是生死不羁的欢恨，问琴弦遥祝了几程，
就用这无名一曲诺此生。
长行的，不停留，
归来的，飘零久。
临别前，重逢后，
林泉渡水，白云载酒。
此身赴风波，还以为今时不识我，
惆怅人间客，谁才是忘情者。
清风过故城，又一次将横笛吹彻，
而此刻，又何以为歌?
是跌碎尘埃的孤魂，在天涯永夜处容身，
听谁唱世外光阴，洞中朝暮只一瞬。
是生死不羁的欢恨，问琴弦遥祝了几程，
就用这无名一曲诺此生。
长行的，不停留，
归来的，飘零久。
临别前，重逢后，
林泉渡水，白云载酒。
是风云浴血的故人，在天地静默处启唇，
低唱过世外光阴，洞中朝暮只一瞬。
是出鞘即斩的霜刃，避不开心头旧红尘，
就用这无名一曲诺此生。
是跌碎尘埃的孤魂，在天涯永夜处容身，
听谁唱世外光阴，洞中朝暮只一瞬。
是生死不羁的欢恨，问琴弦遥祝了几程，
就用这无名一曲诺此生。
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# prefix = 'translate to zh: '
prefix = "translate to en: "
src_text = prefix + text

# translate Russian to Chinese
input_ids = tokenizer(src_text, return_tensors="pt")

generated_tokens = model.generate(**input_ids.to(device))

result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(result)

# import uuid
# import hashlib
# import time
# import json
# import requests

# def trans_from_youdao(word = words):
#     YOUDAO_URL = 'https://openapi.youdao.com/api'
#     with open("api_data.json", "r") as f:
#         data = json.load(f)
#         APP_KEY = data['APP_KEY']
#         APP_SECRET = data['APP_SECRET']
#         # APP_KEY = '您的应用ID'
#         # APP_SECRET = '您的应用密钥'


#     def encrypt(signStr):
#         hash_algorithm = hashlib.sha256()
#         hash_algorithm.update(signStr.encode('utf-8'))
#         return hash_algorithm.hexdigest()


#     def truncate(q):
#         if q is None:
#             return None
#         size = len(q)
#         return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


#     def do_request(data):
#         headers = {'Content-Type': 'application/x-www-form-urlencoded'}
#         return requests.post(YOUDAO_URL, data=data, headers=headers)


#     def connect():
#         q = words

#         data = {}
#         data['from'] = 'en'
#         data['to'] = 'zh-CHS'
#         data['signType'] = 'v3'
#         curtime = str(int(time.time()))
#         data['curtime'] = curtime
#         salt = str(uuid.uuid1())
#         signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
#         sign = encrypt(signStr)
#         data['appKey'] = APP_KEY
#         data['q'] = q
#         data['salt'] = salt
#         data['sign'] = sign
#         # data['vocabId'] = "您的用户词表ID"

#         response = do_request(data)
#         contentType = response.headers['Content-Type']
#         if contentType == "audio/mp3":
#             millis = int(round(time.time() * 1000))
#             filePath = "合成的音频存储路径" + str(millis) + ".mp3"
#             fo = open(filePath, 'wb')
#             fo.write(response.content)
#             fo.close()
#         else:
#             result = json.loads(response.content.decode('utf-8'))
#         return result['translation']
    
#     result = connect()
#     print(result[0])
#     return result