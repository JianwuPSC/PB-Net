import os
import sys
# 指定包含文件的目录
directory = sys.argv[1]
key_word = sys.argv[2]

# 列出目录中的所有文件
files = os.listdir(directory)
# 遍历文件列表，并对每个文件执行重命名操作
for index, file in enumerate(files):
    
    if str(file).find(str('pdb')) != -1:
        # 定义原始文件路径
        original_file_path = os.path.join(directory, file)
        # 定义新文件名（例如，添加前缀或后缀）
        new_file_name = f'{key_word}_{file}'
        new_file_path = os.path.join(directory, new_file_name)
    
        # 重命名文件
        try:
            os.rename(original_file_path, new_file_path)
            print(f"file '{file}' rename '{new_file_name}'")
        except OSError as e:
            print(f"renamed '{file}' err: {e}")
