import openai
import os
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
client = openai.OpenAI(
    api_key=dashscope_api_key,  # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
)

prompt = '''
请将输入翻译成中文
'''

def translateEnText(text : str):
    completion = client.chat.completions.create(
        model="qwen-long",
        messages=[
            {
                'role': 'system',
                'content': prompt
            },
            {
                'role': 'user',
                'content': f"输入文本：{text}"
            }
        ],
        stream=True
    )
    res = r""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].model_dump()["delta"]["content"], end="")
            res += chunk.choices[0].delta.content
            
    return res
