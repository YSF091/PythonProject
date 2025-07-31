import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
0
file_path = "Homework4.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None, header=0)  # 讀取所有工作表
df = sheets["HW_4"]  # 選擇HW_4工作表
X1 = np.array(df["X1"])  # 讀取X1欄作為NumPy陣列
X2 = np.array(df["X2"])  # 讀取X2欄作為NumPy陣列
T = np.array(df["T"])  # 讀取目標標籤T(0、1)
m = len(X1)  # 樣本數量
X = np.column_stack((np.ones(len(X1)), X1, X2))  # 將所有樣本建立成[1 x1 x2]的矩陣
print("Input data: ")
print(X)
W = np.array([0, 0, 1.1]).reshape(-1, 1)  # 設定權重(轉矩陣是為了內積計算)
print("\nWeight: ")
print(W.T)

learning_rate = 1  # 學習率
k = 0  # 訓練次數(time)
Max_iteration = 50  # 最大訓練次數(time)
W_best = W  # 儲存最佳權重
CCount_Max = 0
W_store = []  # 存儲每次更新的權重

for iteration in range(Max_iteration):
    print(f"\n============== Iteration {iteration + 1} ===============")
    CCount = 0  # 每一輪正確分類的樣本數重置為0

    rows = X.shape[0]  # 總樣本數(5列)
    cols = X.shape[1]  # [x0 x1 x2](3行)
    # 逐一計算每一行(樣本)的內積
    for i in range(rows):  # X的每一行(5)
        net = 0  # 初始化
        for j in range(cols):  # X和W共同的維度(3)
            net += X[i][j] * W[j][0]  # 計算內積

        if T[i] == 1 and net <= 0:  # 錯誤分類
            W = W + learning_rate * X[i].reshape(-1, 1)  # 更新權重
            # print(net)
        elif T[i] == 0 and net >= 0:  # 錯誤分類
            W = W - learning_rate * X[i].reshape(-1, 1)  # 更新權重
            # print(net)
        else:  # 分類正確
            W = W
            CCount = CCount + 1  # 累加CCount
            # print(net)

            if CCount > CCount_Max:  # 如果CCount超過歷史最高值，更新W_best
                CCount_Max = CCount
                W_best = W

        k = k + 1
        print(f"k={k}  {W.T}")  # 輸出目前權重
        W_store.append(W.copy())  # 儲存每次的權重(不覆蓋)

    if CCount == m:  # 該輪所有樣本皆正確分類，則結束訓練
        print("\nCCount=m， W_best:")
        print(W_best.T)
        break

else:  # 達到最大訓練次數，則結束訓練
    W_best = W
    print("\nMax_iteration=50， W_best:")
    print(W_best.T)

# 儲存最佳權重
Best_W = W_store[-1].flatten()  # W_store的最後一個權重
W_store_df = pd.DataFrame([Best_W], columns=["W0", "W1", "W2"])  # 建立DataFrame
W_store_df["Bias"] = W_store_df["W0"]  # 新增bise欄位（內容與W0相同）

# 建立新的HW4_Result.xlsx，儲存W到result工作表
with pd.ExcelWriter("HW4_Result.xlsx", engine="openpyxl", mode="w") as writer:
    W_store_df.to_excel(writer, sheet_name="result", index=False)
    print("儲存完成")

# 動態繪圖
fig, ax = plt.subplots()  # 建立圖表和軸
ax.set_xlim(-3, 3)  # 設定x軸範圍
ax.set_ylim(-3, 3)  # 設定y軸範圍
ax.set_xlabel('X1')  # 設定x軸標籤
ax.set_ylabel('X2')  # 設定y軸標籤
ax.set_title('Perceptron')  # 設定標題

# 繪製輸入資料
for i in range(m):
    if T[i] == 1:  # 目標值為第1類
        ax.scatter(X1[i], X2[i], marker='x', color='red')  # 繪製紅色x
    else:  # 目標值為第0類
        ax.scatter(X1[i], X2[i], marker='o', color='blue')  # 繪製藍色o

# 繪製分類線
x1 = np.linspace(-3, 3, 100)  # 分類縣x軸範圍，從-3到3共100個點
line, = ax.plot([], [], 'k-')  # 建立初始分類線(黑色實心線條)


# 更新函數
def update(frame):
    W0, W1, W2 = W_store[frame].flatten()  # 從W_store取得目前權重值
    if W2 != 0:
        x2 = - (W0 / W2) - (W1 / W2) * x1  # 計算對應的x2(y)值
        line.set_data(x1, x2)  # 更新分類線
    return line,


# 設定動畫
ani = FuncAnimation(fig, update, frames=range(len(W_store)), interval=400, blit=False, repeat=False)

plt.show()  # 顯示動畫
print("繪製完成")

# ============================================HW_4_2============================================
while True:  # 無限迴圈，直到輸入正確為止
    print(f"\n============================================")
    a = int(input("是否要繼續計算HW_4_2最佳權重(yes請輸入1/no請輸入0): "))
    if a == 1:  # 選擇執行
        df = sheets["HW_4_2"]  # 選擇HW_4工作表
        X1 = np.array(df["X1"])  # 讀取X1欄作為NumPy陣列
        X2 = np.array(df["X2"])  # 讀取X2欄作為NumPy陣列
        T = np.array(df["T"])  # 讀取目標標籤T(0、1)
        m = len(X1)  # 樣本數量
        X = np.column_stack((np.ones(len(X1)), X1, X2))  # 將所有樣本建立成[1 x1 x2]的矩陣
        #print("Input data: ")
        #print(X)
        W = np.array([0, 0, 1.1]).reshape(-1, 1)  # 設定權重(轉矩陣是為了內積計算)
        #print("\nWeight: ")
        #print(W.T)
        learning_rate = 1  # 學習率
        k = 0  # 訓練次數(time)
        Max_iteration = 500  # 最大訓練次數(time)
        W_best = W  # 儲存最佳權重
        CCount_Max = 0
        W_store = []
        for iteration in range(Max_iteration):
            print(f"\n============== Iteration {iteration + 1} ===============")
            CCount = 0  # 每一輪正確分類的樣本數重置為0
            rows = X.shape[0]  # 總樣本數(5列)
            cols = X.shape[1]  # [x0 x1 x2](3行)
            # 逐一計算每一行(樣本)的內積
            for i in range(rows):  # X的每一行(5)
                net = 0  # 初始化
                for j in range(cols):  # X和W共同的維度(3)
                    net += X[i][j] * W[j][0]  # 計算內積
                if T[i] == 1 and net <= 0:  # 錯誤分類
                    W = W + learning_rate * X[i].reshape(-1, 1)  # 更新權重
                elif T[i] == 0 and net >= 0:  # 錯誤分類
                    W = W - learning_rate * X[i].reshape(-1, 1)  # 更新權重
                else:  # 分類正確
                    W = W
                    CCount = CCount + 1  # 累加CCount
                    if CCount > CCount_Max:  # 如果CCount超過歷史最高值，更新W_best
                        CCount_Max = CCount
                        W_best = W
                k = k + 1
                print(f"k={k}  {W.T}")  # 輸出目前權重
                W_store.append(W.copy())  # 儲存每次的權重(不覆蓋)
            if CCount == m:  # 該輪所有樣本皆正確分類，則結束訓練
                print("\nCCount=m， W_best:")
                print(W_best.T)
                break
        else:  # 達到最大訓練次數，則結束訓練
            W_best = W
            print("\nMax_iteration=500， W_best:")
            print(W_best.T)
        break
    elif a == 0:  # 選擇不執行
        print("結束")
        break
    else:
        print("請重新輸入0或1")

# ============================================HW_4_3============================================
while True:  # 無限迴圈，直到輸入正確為止
    print(f"\n============================================")
    a = int(input("是否要繼續計算HW_4_3最佳準確率(yes請輸入1/no請輸入0): "))
    if a == 1:  # 選擇執行
        df = sheets["HW_4_3"]  # 選擇HW_4工作表
        X1 = np.array(df["X1"])  # 讀取X1欄作為NumPy陣列
        X2 = np.array(df["X2"])  # 讀取X2欄作為NumPy陣列
        T = np.array(df["T"])  # 讀取目標標籤T(0、1)
        m = len(X1)  # 樣本數量
        X = np.column_stack((np.ones(len(X1)), X1, X2))  # 將所有樣本建立成[1 x1 x2]的矩陣
        W = np.array([0, 0, 1.1]).reshape(-1, 1)  # 設定權重(轉矩陣是為了內積計算)
        learning_rate = 1  # 學習率
        k = 0  # 訓練次數(time)
        Max_iteration = 500  # 最大訓練次數(time)
        W_best = W  # 儲存最佳權重
        CCount_Max = 0
        W_store = []
        for iteration in range(Max_iteration):
            #print(f"\n============== Iteration {iteration + 1} ===============")
            CCount = 0  # 每一輪正確分類的樣本數重置為0
            rows = X.shape[0]  # 總樣本數(5列)
            cols = X.shape[1]  # [x0 x1 x2](3行)
            # 逐一計算每一行(樣本)的內積
            for i in range(rows):  # X的每一行(5)
                net = 0  # 初始化
                for j in range(cols):  # X和W共同的維度(3)
                    net += X[i][j] * W[j][0]  # 計算內積
                if T[i] == 1 and net <= 0:  # 錯誤分類
                    W = W + learning_rate * X[i].reshape(-1, 1)  # 更新權重
                elif T[i] == 0 and net >= 0:  # 錯誤分類
                    W = W - learning_rate * X[i].reshape(-1, 1)  # 更新權重
                else:  # 分類正確
                    W = W
                    CCount = CCount + 1  # 累加CCount
                    if CCount > CCount_Max:  # 如果CCount超過歷史最高值，更新W_best
                        CCount_Max = CCount
                        W_best = W
                k = k + 1
                #print(f"k={k}  {W.T}")  # 輸出目前權重
                W_store.append(W.copy())  # 儲存每次的權重(不覆蓋)
            if CCount == m:  # 該輪所有樣本皆正確分類，則結束訓練
                #print("\nCCount=m， W_best:")
                #print(W_best.T)
                break
        else:  # 達到最大訓練次數，則結束訓練
            W_best = W
            #print("\nMax_iteration=500， W_best:")
            #print(W_best.T)
        max_accuracy = CCount_Max / m  # CCount_Max是最大正確分類數
        print(f"\nMax Accuracy: {max_accuracy}")
        break
    elif a == 0:  # 選擇不執行
        print("結束")
        break
    else:
        print("請重新輸入0或1")
