import os
def check_dataset_format(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines, start=1):
            parts = line.strip().split('\t')
            if len(parts) != 2 or not parts[1].isdigit():
                print(f"第 {i} 行格式错误: {line.strip()}")
    print(f"{file_path} 格式检查完成！")

# 检查所有数据文件
base_path = "../ERNIE/datas/data"
for file_name in ["train.txt", "dev.txt", "test.txt"]:
    file_path = os.path.join(base_path, file_name)
    check_dataset_format(file_path)
