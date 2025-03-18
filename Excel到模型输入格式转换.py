import pandas as pd

# 读取 Excel 文件
file_path = "path_to_your_excel_file.xlsx"  # Excel 文件路径
data = pd.read_excel(file_path)

# 检查数据列名是否正确
# 假设 Excel 表格有两列："句子" 和 "标签"
if "句子" not in data.columns or "标签" not in data.columns:
    raise ValueError("Excel 文件缺少必要的列：句子 或 标签")

# 处理数据，转换为 '句子\t标签' 格式
processed_data = data.apply(lambda row: f"{row['句子']}\t{row['标签']}", axis=1)

# 保存为文本文件
output_path = "output_path.txt"  # 输出的文本文件路径
processed_data.to_csv(output_path, index=False, header=False, sep="\n", encoding="utf-8")
print(f"数据已成功保存到 {output_path}")
