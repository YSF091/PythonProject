import numpy as np
import pandas as pd

# 匯入excel
List = pd.read_excel("排程最佳化.xlsx", sheet_name="20j5m", usecols=range(1, 6))
List = List.values
num_jobs = List.shape[0]                                                   # 工作表有幾列
print(List)

# PSO 參數設定
X_U = 100
X_L = 0
m = 30                                                                     # 粒子數
w_init = 1
w_final = 0.4
c1 = 2
c2 = 2
v_init = 0
v_max = 1
v_min = -1
Iter = 10

def bubble_sort(a):
    n = len(a)
    for i in range(n):
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:  # 如果前一個大於後一個，交換
                a[j], a[j + 1] = a[j + 1], a[j]
    return a

# 目標函式：根據p產生工單順序
def objective(p):
    p = np.clip(p, 0, 19)                                    # 設定p的範圍在0 ~ 19之間
    sorted_p = bubble_sort(p.copy())                                      # 用氣泡排序排序亂數值
    List_sorted = List[np.argsort(sorted_p)]                                           # 根據順序排序加工時間表

    R, C = List_sorted.shape
    Gantt = np.zeros([R, C * 2])  # 甘特圖
    Gantt[0, 0] = 0  # 甘特圖的1s為第0秒開始
    Gantt[0, 1] = Gantt[0, 0] + List_sorted[0, 0]  # 1e為1s那格+List第一格的時間

    for i in range(1, C):
        Gantt[0, 2 * i] = Gantt[0, (2 * i) - 1]  # s行：2 * 表示一次要完成兩個，下一個會等於前面那個的
        Gantt[0, (2 * i) + 1] = Gantt[0, 2 * i] + List_sorted[0, i]  # e行：s行 + 任務需要時間

    for i in range(1, R):  # R多少個任務
        Gantt[i, 0] = Gantt[i - 1, 1]  # 第一欄就是前一個任務R完成的時間
        Gantt[i, 1] = Gantt[i, 0] + List_sorted[i, 0]  #
        for j in range(1, C):
            if Gantt[i, 2 * j - 1] >= Gantt[i - 1, 2 * j + 1]:  # 若前一個工作站結束時間比較大
                Gantt[i, 2 * j] = Gantt[i, 2 * j - 1]  # 取代
            else:
                Gantt[i, 2 * j] = Gantt[i - 1, 2 * j + 1]  # 沒有比較大，原本執行完的時間就會等於下一個的時間
            Gantt[i, 2 * j + 1] = Gantt[i, 2 * j] + List_sorted[i, j]  # 比對完的s，加上工作時間，等於e

    return Gantt[-1, -1]




# 初始化粒子位置與速度
np.random.seed(0)
x = np.random.uniform(X_L, X_U, (m, num_jobs))                        # 30 個粒子，每個粒子 20 維（20 工單順序）

# 對每個粒子進行氣泡排序，使其成為有序工單順序
# for i in range(m):
#     x[i] = bubble_sort(x[i])

v = np.zeros_like(x)                                                       # v的零矩陣會跟x的陣列形狀、資料型態一樣
p_best = x.copy()

# 工單順序的候選解
p_best = x.copy()                                                   # 個人最佳解
p_best_fitness = np.array([objective(p) for p in p_best])
g_best = p_best[np.argmin(p_best_fitness)]
k = 0                                                               # 第k次迭代

# PSO
for k in range(Iter + 1):
    w = w_init - k / Iter * (w_init - w_final)
    w = np.clip(w, w_init, w_final)

    v = w * v + c1 * np.random.rand(m, num_jobs) * (p_best - x) + c2 * np.random.rand(m, num_jobs) * (g_best - x)
    v = np.clip(v, v_min, v_max)

    x = x + v
    x = np.clip(x, X_L, X_U)

    x_now = np.array([objective(xi) for xi in x])
    p_now = np.array([objective(p_best[i]) for i in range(m)])

    for a in range(m):
        if x_now[a] < p_now[a]:
            p_best[a] = x[a]

    p_new = np.array([objective(p_best[i]) for i in range(m)])
    b = np.argmin(p_new)  # 找到 p_best 中最小的目標值
    if p_new[b] < objective(g_best):  # 如果目前的個人最佳解比全體最佳解好
        g_best = p_best[b]

# 所有粒子的工件順序與對應 makespan 結果
result_matrix = []

for i in range(m):
    sorted_p = bubble_sort(x[i].copy())          # 用氣泡排序取得排序依據
    job_order = np.argsort(sorted_p)             # 得到實際工件順序
    makespan = objective(x[i])                   # 計算對應 makespan
    row = list(job_order) + [makespan]           # 一筆 = 順序 + makespan
    result_matrix.append(row)

# 印出全部結果（每一列一個粒子）
for row in result_matrix:
    print(np.round(row, 1))

# 最佳解
job_order = np.argsort(g_best)
makespan = objective(g_best)

print("\n最佳解：")
print("工件順序:", [int(i) for i in job_order])
print("總工時:", makespan)
