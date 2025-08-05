import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

folder = os.path.dirname( os.path.realpath( __file__ ) )                                          # 找出當前python腳本的位置

df = pd.read_excel("Homework5.xlsx", sheet_name="HW_5_MP_Linear")                              # 載入excel檔
print(df.columns)                                                                                 # 列出欄位名稱

X1 = df['X1'].values                                                                              # 以X1提取出欄位X1的數值
t = df['T'].values                                                                                # 以t提取出欄位T的數值
m = len(X1)                                                                                       # 樣本數
learning_rate = 0.01                                                                              # 學習率
k = 0                                                                                             # 計算第k次初始化為0
Max_iteration = 50                                                                                # 最多計算50個週期
c = 0                                                                                             # 目前計算第幾組
CCount_Max = 0

X = np.column_stack((np.ones(m), X1))                                                             # bias 項 = X0 = 1
T = np.array(t)
W = np.array([10.0, -1.0])                                                                        # 初始權重

weight_history = []                                                                               # 記錄每個Iteration的權重變化
W_best = []

for k in range(Max_iteration):
    print(f"\n===================== iteration{k+1} =====================")
    CCount = 0                                                                                    # 正確計算的數量

    # 逐筆資料檢查
    for i in range(m):
        y = 0
        for j in range(len(W)):                                                                   # 逐筆W去執行
            y += W[j] * X[i][j]                                                                   # y = W * X

        error = T[i] - y                                                                          # 計算誤差

        if error != 0:                                                                            # 如果誤差不等於0
            for j in range(len(W)):                                                               # 逐筆W去修正
                W[j] += learning_rate * error * X[i][j]                                           # 更新權重

        else:                                                                                     # 誤差等於0
            CCount += 1                                                                           # 正確計算數量+1

        if CCount > CCount_Max:                                                                   # 正確數量 > 目前正確數量
            CCount_Max = CCount                                                                   # 更新正確數量
            W_best = W.copy()

        k += 1

    weight_history.append(W.copy())                                                               # 記錄歷史權重
    W_best.append(W.copy())                                                                       # 紀錄目前最佳權重

    print(f"Updated Weights: {W}")

    if CCount == m:  # 若所有樣本皆正確分類，則結束訓練                                                # 如果正確分類次數 = 資料筆數
        print("\nCCount=m， W_best:")
        print(W_best)
        break

result = pd.DataFrame({                                                                           # 將最佳銓重結果儲存到result變數
    'Best_W1' : [W[0]],
    'Best_W2' : [W[1]]
})

output_path = os.path.join(folder, "Homework5_MPLinear_result.xlsx")                              # 將儲存到excel中
result.to_excel(output_path, index=False)
print(f"最佳權重已保存完成，檔案位置：{output_path}")


fig, ax = plt.subplots()                                                                          # 圖表格式設置
plt.xlabel("X1")
plt.ylabel("T")
plt.title("Adaline")

ax.scatter(X1, T, color='red', marker='o', label="Data Points")                                   # 畫出點
line, = ax.plot([], [], 'b-', label="Decision Boundary")                                    # 初始直線

x_vals = np.linspace(min(X1) - 1, max(X1) + 1, 100)                                          # 設定X軸範圍

def update(frame):
    W = weight_history[frame]                                                                     # 取得目前Iteration的權重
    y_vals = W[0] + W[1] * x_vals                                                                 # 計算直線方程式

    line.set_data(x_vals, y_vals)                                                                 # 更新直線
    ax.set_title(f"Iteration {frame+1}")                                                          # 更新標題
    return line,

# 建立動畫
ani = animation.FuncAnimation(fig, update, frames=len(weight_history), repeat=False)

# 顯示動畫
plt.legend()
plt.show()