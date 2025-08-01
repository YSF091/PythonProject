import numpy as np
import pandas as pd

# 匯入excel
List = pd.read_excel("排程最佳化.xlsx", sheet_name="20j5m", usecols=range(1, 6))
List = List.values
num_jobs = List.shape[0]                                           # 工作表有幾列
print("List:", List)

# PSO 參數設定
X_U = 100
X_L = 0
m = 30             # 粒子數
w_init = 1
w_final = 0.4
c1 = 2
c2 = 2
v_init = 0
v_max = 1
v_min = -1
Iter = 10

# 初始化粒子位置與速度
np.random.seed(0)
x = np.random.uniform(X_L, X_U, (m, num_jobs))  # 30 個粒子，每個粒子 20 維（20 工單順序）
v = np.zeros_like(x)                                                       # v的零矩陣會跟x的陣列形狀、資料型態一樣
p_best = x.copy()


# 評估函數：輸入順序 -> 回傳甘特圖總完工時間（Makespan）
def gantt_eval(order):
    order = order.astype(int) % 20
    seen = set()
    job_seq = [j for j in order if not (j in seen or seen.add(j))]
    job_seq += [j for j in range(20) if j not in seen]  # 補滿20工單
    R, C = List.shape
    Gantt = np.zeros((R, C * 2))
    Gantt[0, 0] = 0
    Gantt[0, 1] = List[job_seq[0], 0]
    for i in range(1, C):
        Gantt[0, 2 * i] = Gantt[0, (2 * i) - 1]
        Gantt[0, (2 * i) + 1] = Gantt[0, 2 * i] + List[job_seq[0], i]
    for i in range(1, R):
        Gantt[i, 0] = Gantt[i - 1, 1]
        Gantt[i, 1] = Gantt[i, 0] + List[job_seq[i], 0]
        for j in range(1, C):
            Gantt[i, 2 * j] = max(Gantt[i, 2 * j - 1], Gantt[i - 1, 2 * j + 1])
            Gantt[i, 2 * j + 1] = Gantt[i, 2 * j] + List[job_seq[i], j]
    return Gantt[-1, -1]


# 初始評估
p_best_fit = np.array([gantt_eval(xi) for xi in p_best])
g_best = p_best[np.argmin(p_best_fit)]
g_best_fit = min(p_best_fit)

# PSO 主迴圈
for k in range(Iter):
    w = max(w_final, w_init - (w_init - w_final) * k / Iter)
    r1, r2 = np.random.rand(m, num_jobs), np.random.rand(m, 20)
    v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
    v = np.clip(v, v_min, v_max)
    x = x + v
    x = np.clip(x, X_L, X_U)

    x_fit = np.array([gantt_eval(xi) for xi in x])
    update_mask = x_fit < p_best_fit
    p_best[update_mask] = x[update_mask]
    p_best_fit[update_mask] = x_fit[update_mask]

    if min(p_best_fit) < g_best_fit:
        g_best = p_best[np.argmin(p_best_fit)]
        g_best_fit = min(p_best_fit)

# 轉換最佳順序為 job list
seen_g = set()
job_order = [j for j in (g_best.astype(int) % 20) if not (j in seen_g or seen_g.add(j))]
job_order += [j for j in range(20) if j not in job_order]

# 甘特圖產生（最佳順序）
R, C = List.shape
Gantt = np.zeros((R, C * 2))
Gantt[0, 0] = 0
Gantt[0, 1] = List[job_order[0], 0]
for i in range(1, C):
    Gantt[0, 2 * i] = Gantt[0, (2 * i) - 1]
    Gantt[0, (2 * i) + 1] = Gantt[0, 2 * i] + List[job_order[0], i]
for i in range(1, R):
    Gantt[i, 0] = Gantt[i - 1, 1]
    Gantt[i, 1] = Gantt[i, 0] + List[job_order[i], 0]
    for j in range(1, C):
        Gantt[i, 2 * j] = max(Gantt[i, 2 * j - 1], Gantt[i - 1, 2 * j + 1])
        Gantt[i, 2 * j + 1] = Gantt[i, 2 * j] + List[job_order[i], j]

# 顯示結果

# 若需顯示甘特圖表格
df_gantt = pd.DataFrame(Gantt, columns=[f"{'M' + str(i // 2 + 1)}_{'S' if i % 2 == 0 else 'E'}" for i in range(C * 2)])
print(df_gantt)
print("✅ 最佳工單順序（Job Order）:", [int(j) for j in job_order])
print("✅ 最小總完工時間（Makespan）:", Gantt[-1, -1])
