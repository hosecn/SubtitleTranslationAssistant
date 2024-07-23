import json
import re

json_str = r"""
{"2": {"english": "And thus, the crown prince replied with his now-famous line: \"Body in the abyss, heart in paradise.\"","chinese": "太子就说了著名的八个字：\"身在无间，心在桃源。\""}}"""

json_str = re.sub(r"\\\\", r'\\', json_str)
# json_str = re.sub(r"asdfasdf", r"aaaa", json_str)
# print(json_str)
# json_str = re.sub(r"\\", r'\\\\', json_str)
# print(json_str)

result = json.loads(json_str)

try:
    for item in result:
        print(result[item]["english"], result[item]["chinese"], 1)
except:
    # print(f"处理文本时出错：{en_sentence}")
    pass