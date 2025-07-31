import numpy as np

# 決策變數上下界、初始參數
T_max = 500
T_min = 1
alpha = 0.99
R = 1
D = 3
coef = np.array([5, 12, 4])

# 目標函數
def objective3(x):
    Z3 = np.dot(coef, x)
    return Z3

# 限制式
def restrict(x):

    return (
        x[0] + 2 * x[1] + x[2] <= 10 and
        abs(2 * x[0] - x[1] + 2 * x[2] - 8) <= 1e-1 and                    # 誤差在0.1內就滿足等式，abs是絕對值
        x[0] >= 0 and
        x[1] >= 0 and
        x[2] >= 0
    )

while True:                                                                # 不斷生成x直到符合限制式
    x = np.random.uniform(0, 10, size=D)
    if restrict(x):                                                        # 如果x符合限制式
        break                                                              # 跳出while

x_best = x.copy()                                                          # 最佳解
y_best = objective3(x)                                                     # 最佳適合值
k = 0                                                                      # 第k次迭代
T = T_max                                                                  # 設溫度T
Iter = 10

while T > T_min:                                                           # 在T達到最小溫度之前
    for I in range(Iter):                                                  # 用for迴圈執行Iter_temp，在每個溫度I下產生三個試探解
        x_new = x + np.random.uniform(-R, R, size=D)                       # x(目前解) + -1 ~ 1之間 = 試探解
        if not restrict(x_new):                                            # 若x_new不符合限制式
            continue                                                       # 不更新

        delta = objective3(x_new) - objective3(x)                          # 變化量
        if delta < 0 or np.random.rand() < np.exp(-delta / T):             # 是否接受試探解
            x = x_new                                                      # 接受的話取代原本的目前解
            if objective3(x) > y_best:                                     # 新適合值較優，更新最佳解跟最佳適合值
                x_best = x.copy()
                y_best = objective3(x)
        k += 1
    T *= alpha                                                             # 降溫

print(f'best x = {x_best}')
print(f'Z* = {y_best:.4f}')
