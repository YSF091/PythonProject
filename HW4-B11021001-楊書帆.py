import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 載入 Excel 檔
df = pd.read_excel("Homework4.xlsx", sheet_name="HW_4")

# 提取出 X1, X2, T 存到各變數中
X1 = df['X1'].values
X2 = df['X2'].values
t = df['T'].values
m = len(X1)                                                             # 樣本數 = 5
learning_rate = 1.0                                                     # 學習率
k = 0                                                                   # 訓練次數(time)
Max_time = 50                                                           # Iteration最多計算 50 次

X = np.column_stack((np.ones(m), X1, X2))  # bias 項 = X0 = 1
T = np.array(t)
W = np.array([0.0, 0.0, 1.1])

# 繪製動畫
def ani(W, title):
    plt.clf()  # 清除當前圖表

    # 找出不同類別的點
    class_1 = (T == 1)
    class_0 = (T == 0)

    # 繪製資料點
    plt.scatter(X1[class_1], X2[class_1], color='red', marker='o', label="T=1")
    plt.scatter(X1[class_0], X2[class_0], color='blue', marker='x', label="T=0")

    # 繪製決策邊界
    x_vals = np.linspace(-3, 3, 100)
    if W[2] != 0:
        y_vals = -(W[1] / W[2]) * x_vals - (W[0] / W[2])
        plt.plot(x_vals, y_vals, 'k-', label="Decision Boundary")

    # 圖表設定
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.grid()
    plt.pause(0.3)  # 暫停 0.3 秒，讓動畫可見

# 感知機訓練
plt.ion()
plt.figure(figsize=(6, 6))

CCount_Max = 0
stop = False                      # 控制不提早停止
for iteration in range(Max_time):
    print(f"\n===================== Iteration {iteration+1} =====================")
    #CCount = 0

    # 逐筆進行訓練
    for i in range(m):     
        net = 0
        for j in range(len(W)):                                         # 遍歷 W 和 X[i] 的每一個元素
            net = net + W[j] * X[i][j]                                  # net = W^T * X[i]**

        if T[i] == 1 and net <= 0:
                W = W + learning_rate * X[i]

        elif T[i] == 0 and net >= 0:
                W = W - learning_rate * X[i]

        #else:
        #    W = W
        #    CCount = CCount + 1

        #ani(W, f"Iteration {iteration + 1}, Data {i + 1}")
        # 印出目前的 W
        print(W)

        # 逐筆進行檢查
        CCount = 0
        for k in range(m):
            net_check = np.dot(W, X[k])
            if (T[k] == 1 and net_check > 0) or (T[k] == 0 and net_check < 0):
                CCount += 1

        # ✅ 若未達成100%，才繼續畫動畫
        if CCount < m:
            ani(W, f"Iteration {iteration + 1}, Data {i + 1}")
        else:
            stop = True
            break  # 全部分類正確，跳出內層

        # ✅ 更新最佳權重
        if CCount >= CCount_Max:
            CCount_Max = CCount
            W_best = W.copy()

    if stop:
        break  # 跳出外層迴圈

    print(f"CCount = {CCount}/{m}  W={W}")
    #print(f"Iteration {iteration + 1}: CCount = {CCount}/{m}  W={W}")



print("\n訓練完成!")
print(f"最終權重: W0={W[0]:.2f}, W1={W[1]:.2f}, W2={W[2]:.2f}")
plt.ioff()
ani(W, "Final Result")
plt.show()

# 輸出 Excel
output_file = r"C:\Users\shufa\OneDrive\桌面\大學\大四\人工智慧\Homework4_Results.xlsx"

def save_results(output_file, W):
    df_result = pd.DataFrame({
        "Bias (W0)": [W[0]],
        "Weight 1 (W1)": [W[1]],
        "Weight 2 (W2)": [W[2]]
    })
    df_result.to_excel(output_file, index=False)
    print(f"結果已保存至 {output_file}")


while True:  # 無限迴圈，直到輸入正確為止
    print(f"\n============================================")
    a = int(input("是否要繼續計算HW_4_2最佳權重(yes請輸入1/no請輸入0): "))
    if a == 1:
        # 載入 Excel 檔
        df = pd.read_excel("Homework4.xlsx", sheet_name="HW_4_2")

        # 提取出 X1, X2, T 存到各變數中
        X1 = df['X1'].values
        X2 = df['X2'].values
        t = df['T'].values
        m = len(X1)  # 樣本數 = 5
        learning_rate = 1.0  # 學習率
        k = 0  # 訓練次數(time)
        Max_time = 500  # Iteration最多計算 50 次

        X = np.column_stack((np.ones(m), X1, X2))  # bias 項 = X0 = 1
        T = np.array(t)
        W = np.array([0.0, 0.0, 1.1])

        # 感知機訓練
        CCount_Max = 0
        for iteration in range(Max_time):
            print(f"\n===================== Iteration {iteration + 1} =====================")
            CCount = 0

            # 逐筆資料檢查
            for i in range(m):
                net = 0
                for j in range(len(W)):  # 遍歷 W 和 X[i] 的每一個元素
                    net = net + W[j] * X[i][j]  # net = W^T * X[i]**

                if T[i] == 1 and net <= 0:
                    W = W + learning_rate * X[i]

                elif T[i] == 0 and net >= 0:
                    W = W - learning_rate * X[i]

                else:
                    W = W
                    CCount = CCount + 1

                # 印出目前的 W
                print(W)

            if CCount >= CCount_Max:  # 如果CCount超過歷史最高值，更新W_best
                CCount_Max = CCount
                W_best = W.copy()

            print(f"Iteration {iteration + 1}: CCount = {CCount}/{m}  W={W}")

            if CCount == m:
                break

        print("\n訓練完成!")
        print(f"最終權重: W0={W[0]:.2f}, W1={W[1]:.2f}, W2={W[2]:.2f}")

        class_1 = (T == 1)
        class_0 = (T == 0)

        plt.figure(figsize=(6, 6))
        plt.scatter(X1[class_1], X2[class_1], color='red', marker='o', label="T=1")
        plt.scatter(X1[class_0], X2[class_0], color='blue', marker='x', label="T=0")

        x_vals = np.linspace(-3, 3, 100)
        if W[2] != 0:
            y_vals = -(W[1] / W[2]) * x_vals - (W[0] / W[2])
            plt.plot(x_vals, y_vals, 'k-', label="Decision Boundary")

        plt.title("Final Perceptron Result")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.legend()
        plt.grid()
        plt.show()

    elif a == 0:  # 選擇不執行
        print("結束")
        break

    else:
        print("請重新輸入0或1")

while True:  # 無限迴圈，直到輸入正確為止
    print(f"\n============================================")
    a = int(input("是否要繼續計算HW_4_3最佳準確率(yes請輸入1/no請輸入0): "))
    if a == 1:
        # 載入 Excel 檔
        df = pd.read_excel("Homework4.xlsx", sheet_name="HW_4_3")

        # 提取出 X1, X2, T 存到各變數中
        X1 = df['X1'].values
        X2 = df['X2'].values
        t = df['T'].values
        m = len(X1)  # 樣本數 = 5
        learning_rate = 1.0  # 學習率
        k = 0  # 訓練次數(time)
        Max_time = 500  # Iteration最多計算 50 次

        X = np.column_stack((np.ones(m), X1, X2))  # bias 項 = X0 = 1
        T = np.array(t)
        W = np.array([0.0, 0.0, 1.1])
        W_best = W.copy()

        # 感知機訓練
        CCount_Max = 0
        for iteration in range(Max_time):
            CCount = 0

            # 逐筆資料檢查
            for i in range(m):
                net = 0
                for j in range(len(W)):  # 遍歷 W 和 X[i] 的每一個元素
                    net = net + W[j] * X[i][j]  # net = W^T * X[i]**

                if T[i] == 1 and net <= 0:
                    W = W + learning_rate * X[i]

                elif T[i] == 0 and net >= 0:
                    W = W - learning_rate * X[i]

                else:
                    #W = W
                    CCount = CCount + 1

            if CCount >= CCount_Max:  # 如果CCount超過歷史最高值，更新W_best
                    CCount_Max = CCount
                    W_best = W.copy()

            if CCount == m:
                    break

        max_accuracy = CCount_Max / m  # CCount_Max是最大正確分類數
        print(f"\nMax Accuracy: {max_accuracy}")

        break

    elif a == 0:  # 選擇不執行
        print("結束")
        break

    else:
        print("請重新輸入0或1")