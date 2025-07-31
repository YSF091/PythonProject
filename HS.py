import numpy as np

# 目標函數
def objective (x):
    return np.sin(x) + np.cos(2 * x)

# 定義氣泡排序法
def bubble(HM, fitness):
    for i in range(a - 1):                                        # 會執行a - 1次，因為最後一個數字會在前面數字排好的時候排完
        for j in range(a - 1 - i):                                # 要比較的範圍是0 ~ a - 2，所以用range的話會是a - 2 + 1，隨著i增加，i的元素已經被排好了。可以不用比較，因而減去
            if fitness[j] > fitness[j + 1]:                       # 若元素較後者大，則交換位置 -> 小到大排列
                HM[j], HM[j + 1] = HM[j + 1], HM[j]
                fitness[j], fitness[j + 1] = fitness[j + 1], fitness[j]
    return HM, fitness

# 初始變數
X_U = 6.28
X_L = 0
HMS = 5
HMCR = 0.6
PAR = 0.4
BW = (X_U - X_L) * 0.05
Iter = 20

#HM = np.random.uniform(X_L, X_U, HMS)                            # 在限定範圍內隨機抽取的數是均勻分布的
HM = X_L + (X_U - X_L) * np.random.rand(HMS)                      # 在限定範圍內隨機抽取的數是均勻分布的
a = len(HM)                                                       # 得知HM中有幾個元素
fitness = np.array([objective(x) for x in HM])                    # 將HM當中每個值帶入適應值公式，並存入fitness中

for i in range(Iter):                                             # 20次迭代
    for j in range(a):                                            # HM當中的每個解，a是HM的維度大小
        if np.random.rand() <= HMCR:                              # 隨機生成0~1的數，選擇記憶庫中現有解的機率
            x = np.random.choice(HM)                              # 從HM裡面選出一個對應維度的值

            if np.random.rand() <= PAR:                           # 對選取的解進行調音的機率
                x = x - BW + np.random.rand() * 2 * BW            # 調音的幅度
                if x > X_U:                                       # 調音的範圍限制
                    x = X_U
                elif x < X_L:
                    x = X_L

        else:                                                     # 不從記憶庫選擇，隨機選擇的機率
            x = X_L + np.random.rand() * (X_U - X_L)              # 重新生成一個x值，並限定範圍

        now_fitness = objective(x)                                # 把目前解的適應值存到now_fitness

        worst_fitness = objective(HM[-1])                         # HM[-1]記憶庫的最後一個值(最差的)

        if now_fitness < worst_fitness:                           # 如果目前解比最差解小
            HM[-1] = x                                            # 取代掉
            fitness[-1] = now_fitness
    HM, fitness = bubble(HM, fitness)                             # 呼叫氣泡排序法排序

for x, fit in zip(HM, fitness):                                   # zip()將HM跟fitness逐一配對
    print(f"[{x}, {fit}]")                                        # 印出目前解以及適應值


