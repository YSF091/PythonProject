# 粒子群優化
import numpy as np
import matplotlib.pyplot as plt

# 目標函數
def objective (x):
    return np.sin(x) + np.cos(2 * x)

# 初始變數
X_U = 6.28
X_L = 0
m = 5                                       # 總粒子數
w_init = 1                                  # 初始權重
w_final = 0.4                                # 終止權重
c1 = 2                                      # 認知參數
c2 = 2                                      # 社群參數
v_init = 0                                  # 初始速度
v_max = 1                                   # 最大移動速度
v_min = -1                                  # 最小移動速度
Iter = 100

# 儲存過程
X_run = []
V_run = []
W_run = []

x = np.random.uniform(X_L, X_U, m)                                  # m個粒子的目前解
p_best = x.copy()                                             # 個人最佳解
g_best = p_best[objective(p_best).argmax()]                                            # 群體最佳解
k = 0                                                               # 第k次迭代
v = np.zeros(m)

# 畫圖
def Draw(x, y, k ):
    plt.clf()                             # 清除目前畫面，畫下一張圖
    plt.title(f'Times={k}')               # 標題：迭代次數
    plt.xlim(X_L, X_U)
    plt.plot(x, y, 'ro')  # 粒子點
    xs = np.linspace(X_L, X_U, 200)
    plt.plot(xs, objective(xs), 'blue')  # 真實曲線
    plt.plot(g_best, objective(g_best), 'go', markersize=10)  # 綠色圓點代表全域最佳解
    plt.pause(0.1)                                                 # 每張影像暫停0.01秒

for k in range(Iter + 1):
    w = w_init - k / Iter * (w_init - w_final)  # 內差法(線性權重遞減)
    w = np.clip(w, w_init, w_final)

    v = w * v + c1 * np.random.rand(m) * (p_best - x) + c2 * np.random.rand(m) * (g_best - x)
    v = np.clip(v, v_min, v_max)                                    # PSO常用的比較陣列大小 -> 速度範圍限制

    x = x + v
    x = np.clip(x, X_L, X_U)                                        # PSO常用的比較陣列大小 -> 解範圍限制

    x_now = objective(x)                                            # 目前這個粒子的最佳解
    p_now = objective(p_best)                                       # 每個粒子的最佳解

    for a in range(m):                                              # 在m個粒子中
        if x_now[a] > p_now[a]:                                     # 如果粒子目前解比個人最佳經驗解好
            p_best[a] = x[a]                                        # 取代掉

    p_new = objective(p_best)
    b = np.argmax(p_new)
    if p_new[b] > objective(g_best):
        g_best = p_best[b]


    Draw(x, x_now, k)



print("g_best = ", g_best)
print("final obj(g_best) = ", objective(g_best))
np.set_printoptions(linewidth=200)
print("V = ", v)
print("w = ", w)
plt.show()










