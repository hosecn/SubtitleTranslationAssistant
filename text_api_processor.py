import openai
import os
import concurrent.futures
import threading
import time
import re

max_lenth = 38

dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
client = openai.OpenAI(
    api_key=dashscope_api_key,  # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
)

# 初始化计数器、锁和时间戳
call_count = 0
lock = threading.Lock()
last_reset_time = time.time()

def safe_process_text(input_text, prompt):
    global call_count, last_reset_time
    
    with lock:  # 获取锁以安全地访问和修改共享资源
        current_time = time.time()
        
        # 检查是否需要重置计数器
        if current_time - last_reset_time >= 60:
            call_count = 0
            last_reset_time = current_time
        
        # 检查是否达到调用上限
        while call_count >= 100:
            remaining_time = 60 - (current_time - last_reset_time)
            if remaining_time > 0:
                print(f"达到调用上限，等待 {remaining_time:.2f} 秒...")
                time.sleep(remaining_time)  # 等待直到下一分钟或剩余时间
                current_time = time.time()  # 更新当前时间
            else:  # 防止浮点运算误差导致的负数
                time.sleep(1)
            
            # 再次检查是否已经重置了计数器
            if current_time - last_reset_time >= 60:
                call_count = 0
                last_reset_time = current_time
                
        call_count += 1  # 增加调用计数

    while True:
        try:
            completion = process_text(input_text, prompt)
            completion = re.sub(r"\\\\", r'\\', completion)
            return completion
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting and retrying...")
            time.sleep(5)  # 等待5秒后重试

        except openai.BadRequestError as e:
            print(f"不适当内容：{input_text}")
            return None
        
        except:
            return None

prompt1 = f'''
作为高级语义处理任务，你的任务是理解并处理一段中英文文本，遵循以下具体步骤：

1. **语义配对与合并**：确保每对相邻的中英文句子在语义上完全对应，合并处理后使它们的意义相互匹配。
2. **额外内容处理**：若英文句子开头或结尾包含了中文句子未提及的内容，或反之，应将这部分额外信息独立识别并标注。
3. **全面处理与排序**：对文本中的所有句子执行上述两项操作，并按原始出现顺序输出处理结果，每一中英句子对作为一个条目。
4. **长句分割**：对中文多于{str(max_lenth)}个字的句子进行分割。

输出结果必须严格遵循指定格式示例("|"为分隔符)：
```
[英文句子1|中文句子1]
[英文句子2|中文句子2]...
``` 
请注意，每个句子对的索引从0开始计数。句子的界定依据是英文中的句点作为结束标志。无论如何英文句子一定要从前到后完整列出。确保最终输出中无多余文本。
'''

prompt2 = '''
请将输入翻译成中文，并每句间换行。
'''

def process_text(input_text, prompt):

    completion = client.chat.completions.create(
        model="qwen-long",
        messages=[
            {
                'role': 'system',
                'content': prompt
            },
            {
                'role': 'user',
                'content': f"输入文本：{input_text}"
            }
        ],
        top_p=0.1,
        stream=True
    )
    res = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            res += chunk.choices[0].delta.content
            
    return res

def process_text_thread(requiste_texts, prompt_index):
    results = [None] * len(requiste_texts)  # 创建一个结果列表，用于存储处理后的文本

    with concurrent.futures.ThreadPoolExecutor() as executor:
        if prompt_index == 1:
            futures = {executor.submit(safe_process_text, text, prompt1): idx for idx, text in enumerate(requiste_texts)}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]  # 获取future的原始索引
                output_text = future.result()
                if future == None:
                    print(f"处理文本时出错:{output_text}")
                    results[index] = {"english": "", "chinese": ""}    
                    continue

                try:
                    output_text = re.sub(r'\[|\]', '', output_text)
                    matches = output_text.split('\n')
                    matches = [match.split('|') for match in matches]
                    results[index] = matches  # 将结果存储在正确的位置
                except:
                    print(f"处理文本时出错:{output_text}")
                    results[index] = {"english": "", "chinese": ""}
        
        else:
            futures = {executor.submit(safe_process_text, text['text'], prompt2): idx for idx, text in enumerate(requiste_texts)}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]  # 获取future的原始索引
                output_text = future.result()
                results[index] = {"text": output_text, "start": requiste_texts[index]['start'], "end": requiste_texts[index]['end']}

        return results