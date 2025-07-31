import numpy as np

# 決策變數上下界、初始參數
T_max = 100
T_min = 1
alpha = 0.99
R = 1
D = 3
coef = np.array([4, 5, 3])

# 目標函數
def objective2(x):
    Z2 = np.dot(coef, x)
    return Z2

# 限制式
def restrict(x):
    return (
        x[0] + x[1] + 2 * x[2] >= 15 and
        5 * x[0] + 6 * x[1] - 5 * x[2] <= 60 and
        x[0] + 3 * x[1] + 5 * x[2] <= 40 and
        x[0] >= 0 and
        x[1] >= 0 and
        x[2] >= 0
    )

while True:                                                                   # 不斷生成x直到符合限制式
    x = np.round(np.random.uniform(0, 20, size=D)).astype(int)       # 在0~20間隨機產生維度D的數列，先四捨五入後轉換成整數型態
    if restrict(x):                                                           # 滿足限制式就跳出while迴圈
        break

x_best = x.copy()                                                             # 最佳解
y_best = objective2(x)                                                        # 最佳適合值
k = 0                                                                         # 第k次迭代
T = T_max                                                                     # 設溫度T
Iter = 3

while T > T_min:                                                              # 在T達到最小溫度之前
    for I in range(Iter):                                                     # 用for迴圈執行Iter_temp，在每個溫度I下產生三個試探解
        x_new = x + np.random.uniform(-R, R, size=D)                          # x(目前解) + -1 ~ 1之間 = 試探解
        x_new = np.round(x_new)                                               # 經退火後會有小數點，需進行四捨五入
        x_new = x_new.astype(int)                                             # 轉換成整數型態
        if not restrict(x_new):                                               # 如果x不符合限制式，不更新
            continue

        delta = objective2(x_new) - objective2(x)                             # 變化量
        if delta < 0 or np.random.rand() < np.exp(-delta / T):                # 是否接受試探解
            x = x_new                                                         # 接受的話取代原本的目前解
            if objective2(x) > y_best:                                        # 新適合值較優，更新最佳解跟最佳適合值
                x_best = x
                y_best = objective2(x)
        k += 1
    T *= alpha                                                                # 降溫

print(f'best x = {x_best}')
print(f'Z* = {y_best:.0f}')
