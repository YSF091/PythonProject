#退火求最小解
import numpy as np
import matplotlib.pyplot as plt

# 目標函數
def objective(x):
    return np.sin(x) + np.cos(2 * x)

# 決策變數上下界、初始參數
X_U = 6.28
X_L = 0
T_max = 100
T_min = 1
alpha = 0.95
R = 1

# 儲存過程
X_run = []                                                          # 模擬退火過程的X值[(x1,f(x1)), (x2,f(x2)),...]
T_run = []                                                          # 模擬退火過程的溫度[100, 99, ...]

x = np.random.uniform(X_L, X_U)                                     # 目前解
x_best = x                                                          # 最佳解
y_best = objective(x)                                               # 最佳適合值
k = 0                                                               # 第k次迭代
T = T_max                                                           # 設溫度T
Iter = 3

def Draw(x, y, k, x_best, y_best):
    plt.clf()                                                       # 清除目前畫面，畫下一張圖
    plt.title(f'Times={k}')                                         # 標題：迭代次數
    plt.xlim(X_L, X_U)                                        # x軸範圍
    a = np.linspace(0, 6.28, 100)                   # 在0 ~ 6.28之間切分成100個點，用來繪製函數曲線
    b = objective(a)                                                # 把切分的100個值帶入目標函數計算
    plt.plot(a, b, color='blue')                              # 用a、b畫線
    plt.scatter(x, y, color='red')                                  # 目前解
    plt.scatter(x_best, y_best, color='green')                      # 目前找到的最佳解
    plt.pause(0.01)                                                 # 每張影像暫停0.01秒

# 模擬退火過程（記錄軌跡和溫度）
plt.ion()                                                           # 開啟即時繪圖互動模式
while T > T_min:                                                    # 在T達到最小溫度之前
    for I in range(Iter):                                           # 用for迴圈執行Iter_temp，在每個溫度I下產生三個試探解
        x_new = x + np.random.uniform(-R, R)                        # x(目前解) + -1 ~ 1之間 = 試探解
        x_new = np.clip(x_new, X_L, X_U)                            # 設定x的值不超過上下界範圍
        delta = objective(x_new) - objective(x)                     # 變化量
        if delta < 0 or np.random.rand() < np.exp(-delta / T):      # 是否接受試探解
            x = x_new                                               # 接受的話取代原本的目前解
            if objective(x) < y_best:                               # 新適合值較優，更新最佳解跟最佳適合值
                x_best = x
                y_best = objective(x)
        Draw(x, objective(x), k, x_best, y_best)
        k += 1
    T *= alpha                                                       # 降溫

plt.ioff()                                                           # 關閉即時繪圖互動模式
plt.show()
print(f'best x = {x_best:.5f}')
print(f'best y = {y_best:.5f}')