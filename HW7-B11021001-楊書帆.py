import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # 引入 FuncAnimation

# 讀取 Excel 檔案
folder = os.path.dirname(os.path.realpath(__file__))  # 找出當前python腳本的位置
data = os.path.join(folder, "Homework7.xlsx")  # 用folder找出的路徑結合EXCEL檔案，成為一個完整路徑，確保檔案可以被找到，且跨平台使用

sht_1 = pd.read_excel(data, sheet_name=0, header=0).iloc[:, 1:].to_numpy()  # 讀取第一張資料表，將第一行隱藏，使用iloc從第二行開始讀取，跳過第一行引索，再將dataframe格式轉換成陣列
sht_2 = pd.read_excel(data, sheet_name=1, header=0).iloc[:, 1:].to_numpy()  # 讀取第二張資料表，將第一行隱藏，使用iloc從第二行開始讀取，跳過第一行引索，再將dataframe格式轉換成陣列

Iteration = 0
Max_Iteration = 100
hidden_learning_rate = 0.1  # 設定學習率
output_learning_rate = 0.001
momentum = 0.4  # 設定動量

def sigmoid(net):  # 定義sigmoid函數
    return 1 / (1 + np.exp(-net))

ones_row = np.ones((sht_1.shape[0], 1))  # 建立一列為1的矩陣
X = np.hstack([ones_row, sht_1[:, 0:1]])  # 合併1矩陣與sht_1提取出的X矩陣
T = sht_1[:, 1:2]  # sht_1提取出的T矩陣

Initial_w_input_hidden = np.array([[0.1, 0.2],
                                   [0.3, 0.4]])

Initial_w_hidden_output = np.array([0.1,
                                   0.2,
                                   0.3])

def forward_pass(x_sample, w_input_hidden, w_hidden_output):
    # 計算 h1
    net_h1 = 0
    for j in range(len(x_sample)):
        net_h1 += w_input_hidden[0][j] * x_sample[j]
    h1 = sigmoid(net_h1)

    # 計算 h2
    net_h2 = 0
    for j in range(len(x_sample)):
        net_h2 += w_input_hidden[1][j] * x_sample[j]
    h2 = sigmoid(net_h2)

    # 組合隱藏層輸出 (含 bias = 1)
    H = np.array([1, h1, h2])

    # 計算輸出層
    net_y = 0
    for j in range(len(H)):
        net_y += w_hidden_output[j] * H[j]

    return h1, h2, net_y, H

# 建立圖表
fig, ax = plt.subplots()
x_plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 200).reshape(-1, 1)
x_plot_with_bias = np.hstack([np.ones_like(x_plot), x_plot])  # 加上 bias

# 讀取第二張工作表的第三欄作為 target（紅線）
target_plot = sht_2[:, 1]

# 每一輪動畫更新函數
def update(frame):
    ax.clear()
    ax.set_title(f"Modeling  - Iteration {frame + 1}")
    ax.set_xlabel("x1")
    ax.set_ylabel("Output y")

    # 去掉格線
    ax.grid(False)  # 關閉格線

    # 在每次迭代中，更新權重並重新計算預測結果
    output_y_all = []  # 儲存每次迭代中每個樣本的 output_y
    for c in range(len(X)):  # 遍歷每個樣本進行訓練
        x_sample = X[c]
        t_sample = T[c]

        # 執行一次前向傳播
        h1, h2, output_y, H = forward_pass(x_sample, Initial_w_input_hidden, Initial_w_hidden_output)

        delta_y = t_sample[0] - output_y

        # 計算隱藏層的誤差項 delta_h1 和 delta_h2
        delta_h1 = delta_y * Initial_w_hidden_output[1] * h1 * (1 - h1)
        delta_h2 = delta_y * Initial_w_hidden_output[2] * h2 * (1 - h2)

        # 更新權重 - 輸出層權重變化 ΔWₖᴴ₂ₒ
        for i in range(len(Initial_w_hidden_output)):
            Initial_w_hidden_output[i] += output_learning_rate * delta_y * H[i]

        # 更新權重 - 隱藏層權重變化 ΔWₖᴵ²ᴴ
        for i in range(len(Initial_w_input_hidden.T)):
            Initial_w_input_hidden.T[i] += hidden_learning_rate * delta_h1 * x_sample[i]
            Initial_w_input_hidden.T[i] += hidden_learning_rate * delta_h2 * x_sample[i]

        # 儲存每次迭代中每個樣本的 output_y
        output_y_all.append(output_y)

        print(f"\nSample {c + 1}:")
        print("x_sample =", x_sample)
        print("h1 =", h1)
        print("h2 =", h2)
        print("Output_y =", output_y)
        print("Delta_y =", delta_y)
        print("Delta_h1 =", delta_h1)
        print("Delta_h2 =", delta_h2)
        print("Updated Initial_w_input_hidden:", Initial_w_input_hidden)
        print("Updated Initial_w_hidden_output:", Initial_w_hidden_output)
        print("============")
        # 最後權重
    W_input_hidden = Initial_w_input_hidden
    W_hidden_output = Initial_w_hidden_output
    print("Final W_input_hidden:", W_input_hidden)
    print("Final W_hidden_output:", W_hidden_output)

    # 預測連續輸入 x_plot 的輸出
    y_pred = []
    for x in x_plot_with_bias:
        _, _, out, _ = forward_pass(x, Initial_w_input_hidden, Initial_w_hidden_output)
        y_pred.append(out)

    # 畫目標點與模型曲線
    ax.scatter(X[:, 1], T, color='red', label='Training Data (Sheet1)', s=4)  # 紅線 = 第二張表的第三欄
    ax.plot(x_plot, y_pred, color='blue', label='Model Output')

    ax.legend()

# 建立動畫：每一輪畫一次
ani = FuncAnimation(fig, update, frames=Max_Iteration, interval=200, repeat=False)

# 顯示動畫
plt.show()

# 儲存最佳權重至Excel
W_input_hidden = Initial_w_input_hidden
W_hidden_output = Initial_w_hidden_output
W_input_hidden_flatten = W_input_hidden.flatten()
W_hidden_output_flatten = W_hidden_output.flatten()

# 創建 DataFrame 並將所有數據放入同一行
result = pd.DataFrame({
    'Final W_input_hidden': [W_input_hidden_flatten],  # W_input_hidden 展開後
    'Final W_hidden_output': [W_hidden_output_flatten]  # W_hidden_output
})

output_path = os.path.join(folder, "Homework7_result.xlsx")  # 儲存到Excel檔案
result.to_excel(output_path, index=False)
print(f"最佳權重已保存完成，檔案位置：{output_path}")