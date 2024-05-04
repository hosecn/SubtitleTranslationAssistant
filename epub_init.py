import re
import os
import zipfile
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import shutil


epub_input_folder = 'input_data'
ebook_folder = 'output/ebook_folder'
epub_output_folder = 'output/epub'
txt_output_folder = 'output/txt'


def unzip_epub(epub_file, output_folder):
    shutil.rmtree(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        # 使用zipfile模块打开EPUB文件
        with zipfile.ZipFile(epub_file, 'r') as zip_ref:
            # 解压所有内容到指定文件夹
            zip_ref.extractall(output_folder)
            
        print(f"EPUB文件已成功解压到：{output_folder}")





def process_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.endswith(".xhtml"):
                continue
            
            with open(file_path, "r", encoding="utf-8") as f:
                html_file = f.read()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_file)





def compress_folder_to_epub(folder_path, epub_output_file_path):
    if not os.path.exists(epub_output_folder):
        os.makedirs(epub_output_folder)
    
    # 创建一个ZipFile对象，在写入模式下
    with zipfile.ZipFile(epub_output_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 创建文件的绝对路径
                file_path = os.path.join(root, file)
                # 创建在ZIP文件中的相对路径
                arcname = os.path.relpath(file_path, folder_path)
                # 向ZIP文件添加文件
                zipf.write(file_path, arcname)





def epub_to_txt(epub_file_path, txt_file_path):
    if not os.path.exists(txt_output_folder):
        os.makedirs(txt_output_folder)

    book = epub.read_epub(epub_file_path)
    
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        # 遍历文档中的每个项目
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # 使用BeautifulSoup解析HTML内容
                soup = BeautifulSoup(item.content, 'html.parser')
                text = soup.get_text()
                text = re.sub(r"https://oceanofpdf.com", "", text)
                text = re.sub(r"OceanofPDF.com", "", text)                
                txt_file.write(text + '\n\n')
                
    print(f"EPUB文件 '{epub_file_path}' 已成功转换为TXT文件：'{txt_file_path}'")







for root, dirs, files in os.walk(epub_input_folder):
    for file in files:
        file_path = os.path.join(root, file)
        if not file.endswith(".epub"):
            continue

        file_name = file.rstrip('.epub')

        unzip_epub(file_path, ebook_folder)
        process_folder(ebook_folder)
        epub_output_file_path = os.path.join(epub_output_folder, file)
        compress_folder_to_epub(ebook_folder, epub_output_file_path)
        txt_output_path = os.path.join(txt_output_folder, file_name + '.txt')
        epub_to_txt(epub_output_file_path, txt_output_path)
