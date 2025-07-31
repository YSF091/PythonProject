# 粒子群優化
import numpy as np
import matplotlib.pyplot as plt

# 目標函數
def objective (x):
    return np.sin(x) + np.cos(2 * x)

# 初始變數
X_U = 6.28
X_L = 0
m = 5                                                               # 總粒子數
w_init = 1                                                          # 初始權重
w_final = 0.4                                                       # 終止權重
c1 = 2                                                              # 認知參數
c2 = 2                                                              # 社群參數
v_init = 0                                                          # 初始速度
v_max = 1                                                           # 最大移動速度
v_min = -1                                                          # 最小移動速度
Iter = 100

x = np.random.uniform(X_L, X_U, m)                                  # m個粒子的目前解
p_best = x.copy()                                                   # 個人最佳解
g_best = p_best[objective(p_best).argmin()]                         # 群體最佳解，把p_best陣列中的每個x帶入目標函式並回傳代入函式結果的陣列，接著在此陣列找出最小值
k = 0                                                               # 第k次迭代
v = np.zeros(m)                                                     # 初始化速度矩陣

# 畫圖
def Draw(x, y, k):
    plt.clf()                                                       # 清除目前畫面，畫下一張圖
    plt.title(f'Times={k}')                                         # 標題：迭代次數
    plt.xlim(X_L, X_U)
    plt.plot(x, y, 'ro')                                      # m個粒子的位置
    xs = np.linspace(X_L, X_U, 200)
    plt.plot(xs, objective(xs), 'blue')  # 真實曲線
    plt.plot(g_best, objective(g_best), 'go', markersize=10)  # 綠色圓點代表全域最佳解
    plt.pause(0.01)                                                 # 每張影像暫停0.01秒

for k in range(Iter + 1):
    w = w_init - k / Iter * (w_init - w_final)                      # 內差法(線性權重遞減)
    w = np.clip(w, w_init, w_final)                                 # PSO常用的比較陣列與「純量」大小 -> 權重範圍限制

    v = w * v + c1 * np.random.rand(m) * (p_best - x) + c2 * np.random.rand(m) * (g_best - x)
    v = np.clip(v, v_min, v_max)                                    # PSO常用的比較「陣列」與純量大小 -> 速度範圍限制

    x = x + v                                                       # 移步公式
    x = np.clip(x, X_L, X_U)                                        # PSO常用的比較陣列與「純量」大小 -> 解範圍限制

    x_now = objective(x)                                            # 目前這個粒子的最佳解
    p_now = objective(p_best)                                       # 每個粒子的最佳解

    for a in range(m):                                              # 在m個粒子中
        if x_now[a] < p_now[a]:                                     # 如果粒子目前解(x_now[a]，第a個粒子)比個人目前最佳經驗解好
            p_best[a] = x[a]                                        # 取代掉

    p_new = objective(p_best)                                       # 個人最佳解
    b = np.argmin(p_new)                                            # 比較出個人最佳解中的最小值
    if p_new[b] < objective(g_best):                                # 目前個人最佳解較全域最佳解小
        g_best = p_best[b]                                          # 取代掉

    if k % 10 == 0:                                                 # 每迭代10次印一次x的位置
        print(f"k = {k}", np.round(x, 2))

    Draw(x, x_now, k)


print("p_best = ", p_best)
print("g_best = ", g_best)
print("final obj(g_best) = ", objective(g_best))
np.set_printoptions(linewidth=200)                                  # print出的V可以擺在同一行
print("V = ", v)
print("w = ", w)
plt.show()

# c1、c2越大可以增加探索能力