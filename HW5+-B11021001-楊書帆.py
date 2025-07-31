import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

folder = os.path.dirname( os.path.realpath( __file__ ) )                                          #找出當前python腳本的位置
data = os.path.join( folder , "Homework5.xlsx" )                                                  #用folder找出的路徑結合EXCEL檔案，成為一個完整路徑，確保檔案可以被找到，且跨平台使用

df = pd.read_excel(data, sheet_name="HW_5_MP_un-Linear")                                             #讀取EXCEL中HW_5_CP工作表的內容
print(df.columns)                                                                                 #列出欄位名稱

X1 = df['X1'].values                                                                              #以X1提取出欄位X1的數值
t  = df['T'].values                                                                               #以t提取出欄位T的數值
m = len(X1)                                                                                       #樣本數
learning_rate = 0.01                                                                              #學習率
k = 0                                                                                             #計算第k次初始化為0
Max_iteration= 50                                                                                 #最多計算50個週期
c = 0                                                                                             #目前計算第幾組
updated = True

X = np.column_stack((np.ones(m), X1))                                                             #  bias 項 = X0 = 1
T = np.array(t)
W = np.array([10.0, -1.0])                                                                        #初始權重

weight_history = []                                                                               #記錄每個Iteration的權重變化

for k in range(Max_iteration):
    print(f"\n===================== iteration{k+1} =====================")
    updated = False                                                                               #新訓練迴圈開始設為False，有誤分類發生且更新權重設為True

    # 逐筆資料檢查
    for i in range(m):
        y = 0
        for j in range(len(W)):                                                                   #遍歷 W 和 X[i] 的每一個元素
            y += W[j] * X[i][j]                                                                   #y = W * X

        error = T[i] - y                                                                          #計算誤差

        for j in range(len(W)):
            W[j] += learning_rate * error * X[i][j]                                               #更新權重
            updated = True

    weight_history.append(W.copy())                                                               # 確保每個銓重有被記錄下來

    print(f"Updated Weights: {W}")


    if not updated:                                                                               #如果本次迭代沒有更新權重，則提早結束訓練
        print("Converged, stopping early.")
        break

result = pd.DataFrame({                                                                           #將最佳銓重結果儲存到result變數
    'Best_W1' : [W[0]],
    'Best_W2' : [W[1]]
})

output_path = os.path.join(folder, "Homework5+_MPunLinear_result.xlsx")                              #將儲存到excel中
result.to_excel(output_path, index=False)
print(f"最佳權重已保存完成，檔案位置：{output_path}")


fig, ax = plt.subplots()                                                                          #圖表格式設置
plt.xlabel("X1")
plt.ylabel("T")
plt.title("Adaline")

ax.scatter(X1, T, color='red', marker='o', label="Data Points")                                   #畫出點
line, = ax.plot([], [], 'b-', label="Decision Boundary")                                          #初始直線

x_vals = np.linspace(min(X1) - 1, max(X1) + 1, 100)                                          #設定X軸範圍



def update(frame):
    """根據不同的迭代權重，更新決策邊界"""
    W = weight_history[frame]  # 取得當前迭代的權重
    y_vals = W[0] + W[1] * x_vals  # 計算直線方程式

    line.set_data(x_vals, y_vals)  # 更新直線
    ax.set_title(f"Iteration {frame+1}")  # 更新標題
    return line,

# 建立動畫
ani = animation.FuncAnimation(fig, update, frames=len(weight_history), repeat=False)

# 顯示動畫
plt.legend()
plt.show()