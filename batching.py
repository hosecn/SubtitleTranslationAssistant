import subprocess
import os

def run_script_and_update_file(times):
    for i in range(times):
        # 运行main.py
        subprocess.run(["python", "main.py"])
        
        # 读取data.txt第一行，转换为整数并加1，然后再写回文件
        with open("data.txt", "r+") as file:
            content = file.readlines()  # 读取所有行
            number = int(content[0])  # 将第一行转换为整数
            number += 1  # 数字加1
            content[0] = f"{number}"  # 更新第一行的内容
            
            # 将内容重写回文件，注意这里会清空原文件后写入
            file.seek(0)  
            file.writelines(content)
            file.truncate()  # 确保多余的字符被删除

# 运行main.py 30次，并更新data.txt
run_script_and_update_file(30)