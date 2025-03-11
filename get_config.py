import pandas as pd
from random import randint
df = pd.read_csv("./datasets/real/response_tuda.csv",header=None)
last_column = df.iloc[:, 2]
new_column = []

for value in last_column:
     new_column.append(randint(0,4))

df.iloc[:, 2] = new_column

# 保存修改后的 CSV 文件
df.to_csv('datasets/real/response_tuda.csv', index=False)