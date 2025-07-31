import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 匯入excel
List = pd.read_excel("排程最佳化.xlsx", sheet_name="20j5m", usecols=range(1, 6))
List = List.values
print(List)

# 初始變數
X_U = 100
X_L = 0
m = 30                                                              # 總粒子數
w_init = 1                                                          # 初始權重
w_final = 0.4                                                       # 終止權重
c1 = 2                                                              # 認知參數
c2 = 2                                                              # 社群參數
v_init = 0                                                          # 初始速度
v_max = 1                                                           # 最大移動速度
v_min = -1                                                          # 最小移動速度
Iter = 10

x = np.random.uniform(X_L, X_U, m)                                  # m個粒子的目前解
p_best = x.copy()                                                   # 個人最佳解
g_best = p_best[objective(p_best).argmin()]                         # 群體最佳解，把p_best陣列中的每個x帶入目標函式並回傳代入函式結果的陣列，接著在此陣列找出最小值
k = 0                                                               # 第k次迭代
v = np.zeros(m)                                                     # 初始化速度矩陣

