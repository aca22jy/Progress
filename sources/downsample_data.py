import pandas as pd
import numpy as np

# 定义文件路径和类别列
file_path = 'sources/ProgressTrainingCombined.tsv'
category_columns = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']
reference_category = 'Religion'

# 读取 TSV 文件
try:
    df = pd.read_csv(file_path, sep='\t')
except FileNotFoundError:
    print(f"错误：文件 {file_path} 未找到。")
    exit()

# 确保所有类别列都存在于 DataFrame 中
missing_cols = [col for col in category_columns if col not in df.columns]
if missing_cols:
    print(f"错误：以下列在文件中缺失：{', '.join(missing_cols)}")
    exit()

# 将类别列转换为数值类型，非数值转为 NaN，然后填充为 0 (假设原始数据中 0/1 代表类别)
for col in category_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# 计算参照类别 ('Religion') 中值为 1 的样本数量
try:
    religion_positive_count = df[reference_category].sum()
    print(f"'{reference_category}' 类别中值为 1 的样本数量: {religion_positive_count}")
except KeyError:
    print(f"错误：参照类别 '{reference_category}' 在文件中未找到。")
    exit()


# 创建 DataFrame 的副本进行修改
df_modified = df.copy()

# 遍历其他类别列进行下采样
for category in category_columns:
    if category == reference_category:
        continue  # 跳过参照类别本身

    current_positive_count = df_modified[category].sum()
    print(f"处理前 '{category}' 类别中值为 1 的样本数量: {current_positive_count}")

    if current_positive_count > religion_positive_count:
        num_to_change = current_positive_count - religion_positive_count
        
        # 获取当前类别值为 1 的所有行的索引
        positive_indices = df_modified[df_modified[category] == 1].index
        
        # 随机选择需要将 1 变为 0 的行的索引
        indices_to_change = np.random.choice(positive_indices, size=num_to_change, replace=False)
        
        # 将这些选定行的对应类别值从 1 修改为 0
        df_modified.loc[indices_to_change, category] = 0
        print(f"处理后 '{category}' 类别中值为 1 的样本数量: {df_modified[category].sum()} (已减少 {num_to_change} 个)")
    else:
        print(f"'{category}' 类别中值为 1 的样本数量 ({current_positive_count}) 未超过 '{reference_category}' ({religion_positive_count})，无需更改。")

# 验证更改
print("\n修改后各类别值为 1 的样本数量:")
for category in category_columns:
    print(f"'{category}': {df_modified[category].sum()}")

# 建议将修改后的 DataFrame 保存到新文件
output_file_path = 'sources/ProgressTrainingCombined_downsampled.tsv'
df_modified.to_csv(output_file_path, sep='\t', index=False)
print(f"\n已将下采样后的数据保存到: {output_file_path}")
