import numpy as np

# 決策變數上下界、初始參數
T_max = 500
T_min = 1
alpha = 0.99
R = 1
D = 30
X_L = np.full(D, -100)
X_U = np.full(D, 100)

# 目標函數
def objective5(x):
    Z5 = np.sum(x ** 2)
    return Z5

# 限制式
x = np.round(np.random.uniform(X_L, X_U, size=D))                     # 目前解
x_best = x.copy()                                                     # 最佳解
y_best = objective5(x)                                                # 最佳適合值
k = 0                                                                 # 第k次迭代
T = T_max                                                             # 設溫度T
Iter = 5

# 模擬退火過程（記錄軌跡和溫度）
while T > T_min:                                                      # 在T達到最小溫度之前
    for I in range(Iter):                                             # 用for迴圈執行Iter_temp，在每個溫度I下產生三個試探解
        x_new = x + np.random.uniform(-R, R, size=D)                  # x(目前解) + -1 ~ 1之間 = 試探解
        x_new = np.clip(x_new, X_L, X_U)                              # 設定x的值不超過上下界範圍

        delta = objective5(x_new) - objective5(x)                     # 變化量
        if delta < 0 or np.random.rand() < np.exp(-delta / T):        # 是否接受試探解
            x = x_new                                                 # 接受的話取代原本的目前解
            if objective5(x) < y_best:                                # 新適合值較優，更新最佳解跟最佳適合值
                x_best = x.copy()
                y_best = objective5(x)
        k += 1
    T *= alpha                                                        # 降溫

print(f'best x = {x_best}')
print(f'Z* = {y_best:.5f}')

