import pandas as pd
from random import randint
df1 = pd.read_csv("./datasets/real/response.csv",header=None)
df2=pd.read_csv("./datasets/real/real_R_tuda.csv",header=None)
last_column = df1.iloc[:, 0]
new_column = []
for item in last_column:
    new_column.append(item)

df2.iloc[:,0] = new_column

df2.to_csv('datasets/real/real_R_tuda.csv', index=False)