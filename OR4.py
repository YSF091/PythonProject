import numpy as np

# 決策變數上下界、初始參數
T_max = 100
T_min = 1
alpha = 0.99
R = 1
D = 4
coef = np.array([5, 6, 9, 8])

# 目標函數
def objective4(x):
    Z4 = np.dot(coef, x)
    return Z4

# 限制式
def restrict(x):
    return (
        x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3] <= 5 and
        x[0] + x[1] + 2 * x[2] + 3 * x[3] <= 3 and
        x[0] >= 0 and
        x[1] >= 0 and
        x[2] >= 0 and
        x[3] >= 0
    )

while True:
    x = np.round(np.random.uniform(0, 10, size=D)).astype(int)
    if restrict(x):
        break

x_best = x.copy()                                                          # 最佳解
y_best = objective4(x)                                                     # 最佳適合值
k = 0                                                                      # 第k次迭代
T = T_max                                                                  # 設溫度T
Iter = 3

while T > T_min:                                                           # 在T達到最小溫度之前
    for I in range(Iter):                                                  # 用for迴圈執行Iter_temp，在每個溫度I下產生三個試探解
        x_new = x + np.random.uniform(-R, R, size=D)                       # x(目前解) + -1 ~ 1之間 = 試探解
        x_new = np.round(x_new)
        x_new = x_new.astype(int)
        if not restrict(x_new):
            continue

        delta = objective4(x_new) - objective4(x)                         # 變化量
        if delta < 0 or np.random.rand() < np.exp(-delta / T):            # 是否接受試探解
            x = x_new                                                     # 接受的話取代原本的目前解
            if objective4(x) > y_best:                                    # 新適合值較優，更新最佳解跟最佳適合值
                x_best = x
                y_best = objective4(x)
        k += 1
    T *= alpha                                                            # 降溫

print(f'best x = {x_best}')
print(f'Z* = {y_best:.0f}')
