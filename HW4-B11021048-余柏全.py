import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

folder = os.path.dirname( os.path.realpath( __file__ ) ) #使用OS找出當前資料夾路徑
data = os.path.join( folder , "Homework4.xlsx" ) #用先前找出的路徑再加上EXCEL檔案名

M0 = pd.read_excel( data , sheet_name = 0 , header = 0 ).iloc[:, 1:].to_numpy() #讀取第一章資料表，將第一行隱藏，使用iloc從第二航開始讀取，跳過第一行引索，再將dataframe格式轉換成陣列
M1 = pd.read_excel( data , sheet_name = 1 , header = 0 ).iloc[:, 1:].to_numpy() #讀取第二章資料表，將第一行隱藏，使用iloc從第二航開始讀取，跳過第一行引索，再將dataframe格式轉換成陣列
M2 = pd.read_excel( data , sheet_name = 2 , header = 0 ).iloc[:, 1:].to_numpy() #讀取第三章資料表，將第一行隱藏，使用iloc從第二航開始讀取，跳過第一行引索，再將dataframe格式轉換成陣列

data = [M0, M1]
def Perceptron(M1,learning_rate,Weight,Max_iteration):
    M1_x = M1[:, :2]                                                                       # 僅讀取X1,X2(在陣列中僅讀取前兩個)
    M1_y = M1[:, 2:]                                                                       # 僅讀取y(在陣列中讀取最後一個目標值)

    row = M1.shape[0]                                                                      # 抓取資料筆數以驗證正確率
    count = 0                                                                              #紀錄計算次數
    weight_history = []                                                                    #紀錄每次的權重參數 後續matplotlib使用

    while(count < Max_iteration):                                                          #當目前count小於計算次數時重複計算
        correct = 0                                                                        #每次計算時重製為0
        for j in range( M1_x.shape[0] ):                                                   #依序計算陣列內資料
            if count >= Max_iteration:
                break                                                                      #為避免在數據無法整除最大計算次數時，會有多算的問題，所以在每次計算前增加一個判斷式
            count += 1                                                                     #計算次屬累加
            num = Weight[0] * 1 +Weight[1] * M1_x[j,0] + Weight[2] * M1_x[j,1]             #決策函數
            if num >= 0 and M1_y[j,0] == 0:                                                #當決策值大於等於0時且目標值為0時，需要向下調整權重

                Weight[0] -= learning_rate * 1
                Weight[1] -= learning_rate * M1_x[j,0]
                Weight[2] -= learning_rate * M1_x[j,1]

            elif num <= 0 and M1_y[j,0] == 1:                                              #當決策值小於等於0時且目標值為1時，需要向上調整權重

                Weight[0] += learning_rate * 1
                Weight[1] += learning_rate * M1_x[j, 0]
                Weight[2] += learning_rate * M1_x[j, 1]

            else:                                                                          #其他情況都算正確，所以CORRECT累加
                correct += 1
            weight_history.append(list(Weight.copy()))                                            #將每次的權重都保存，以便應用在變化圖
            print("========================================================")
            print(f"Iteration {count}: correct = {correct}/{row}")
            print(f"Weight {Weight}\n")

        if correct == row:
            print("Final Weights:", Weight)
            break
    result = pd.DataFrame({
        'Best_bias': [Weight[0]],
        'Best_W1': [Weight[1]],
        'Best_W2': [Weight[2]],
    })

    result.to_excel("Homework4_liner_result.xlsx")
    print("權重已保存完成")
    weight_history = np.array(weight_history)

    fig, ax = plt.subplots()                                                               #這邊在建立圖表的樣式
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Perceptron")

    for i in range(row):                                                                   #以目標值0、1判斷資料應該以藍色圓點或紅色叉叉區分
        if M1_y[i] <= 0:
            ax.scatter(M1_x[i, 0], M1_x[i, 1], color='blue', marker='o')
        else:
            ax.scatter(M1_x[i, 0], M1_x[i, 1], color='red', marker='x')

    decision_boundary, = ax.plot([], [], 'r-', lw=2)                                       #初始化決策邊界的函數，線條寬度為2 顏色為紅

    def update(frame):                                                                     #更新線條的自定義函數
        W = weight_history[frame]
        x_vals = np.array(ax.get_xlim())
        if W[2] != 0:                                                                      # 計算決策邊界的 Y 值
            y_vals = -(W[0] + W[1] * x_vals) / W[2]
        else:
            y_vals = np.array([ax.get_ylim()[0], ax.get_ylim()[1]])                        # 避免除以 0 的錯誤

        decision_boundary.set_data(x_vals, y_vals)                                         #更新決策邊界的X和Y值。
        return decision_boundary,

    ani = animation.FuncAnimation(fig, update, frames=len(weight_history), interval=200, repeat=False) #更新線條，頻率為每100毫秒

    plt.show()


def Perceptron_nonliner(M1,learning_rate,Weight,Max_iteration):
    M1_x = M1[:, :2]                                                                       # 僅讀取X1,X2(在陣列中僅讀取前兩個)
    M1_y = M1[:, 2:]                                                                       # 僅讀取y(在陣列中讀取最後一個目標值)

    row = M1.shape[0]                                                                      # 抓取資料筆數以驗證正確率
    count = 0                                                                              #紀錄計算次數
    weight_history = []                                                                    #紀錄每次的權重參數 後續matplotlib使用

    Weight = np.array(Weight, dtype=float)                                                 # 轉為 NumPy 陣列
    best_weight = Weight.copy()                                                            # 記錄最佳權重
    best_correct = 0                                                                       # 記錄最佳正確數

    while(count < Max_iteration):                                                          #當目前count小於計算次數時重複計算
        correct = 0                                                                        #每次計算時重製為0
        for j in range( M1_x.shape[0] ):                                                   #依序計算陣列內資料
            if count >= Max_iteration:
                break                                                                      #為避免在數據無法整除最大計算次數時，會有多算的問題，所以在每次計算前增加一個判斷式
            count += 1                                                                     #計算次屬累加
            num = Weight[0] * 1 +Weight[1] * M1_x[j,0] + Weight[2] * M1_x[j,1]             #決策函數
            if num >= 0 and M1_y[j,0] == 0:                                                #當決策值大於等於0時且目標值為0時，需要向下調整權重

                Weight[0] -= learning_rate * 1
                Weight[1] -= learning_rate * M1_x[j,0]
                Weight[2] -= learning_rate * M1_x[j,1]

            elif num <= 0 and M1_y[j,0] == 1:                                              #當決策值小於等於0時且目標值為1時，需要向上調整權重

                Weight[0] += learning_rate * 1
                Weight[1] += learning_rate * M1_x[j, 0]
                Weight[2] += learning_rate * M1_x[j, 1]

            else:                                                                          #其他情況都算正確，所以CORRECT累加
                correct += 1  # 若分類正確則 +1

            weight_history.append(Weight.copy())  # 記錄權重變化

                                                                                           # 記錄最佳權重與正確數
        if correct > best_correct:
            best_correct = correct
            best_weight = Weight.copy()

        print("========================================================")
        print(f"Iteration {count}: correct = {correct}/{row}")
        print(f"Current Weights: {Weight}")
        print(f"Best Correct so far: {best_correct}/{row}")

    print("Best Weights:", best_weight)
    print("Accuracy :", best_correct / row)

    result = pd.DataFrame({
        'Best_bias' : [best_weight[0]],
        'Best_W1' : [best_weight[1]],
        'Best_W2' : [best_weight[2]],
        'Accuracy': [best_correct / row],
    })

    result.to_excel("Homework4_Nonliner_result.xlsx")
    print("權重已保存完成")

    weight_history = np.array(weight_history)

    fig, ax = plt.subplots()                                                               #這邊在建立圖表的樣式
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Perceptron")

    for i in range(row):                                                                   #以目標值0、1判斷資料應該以藍色圓點或紅色叉叉區分
        if M1_y[i] <= 0:
            ax.scatter(M1_x[i, 0], M1_x[i, 1], color='blue', marker='o')
        else:
            ax.scatter(M1_x[i, 0], M1_x[i, 1], color='red', marker='x')

    decision_boundary, = ax.plot([], [], 'r-', lw=2)                                       #初始化決策邊界的函數，線條寬度為2 顏色為紅

    def update(frame):                                                                     #更新線條的自定義函數
        W = weight_history[frame]
        x_vals = np.array(ax.get_xlim())
        if W[2] != 0:                                                                      # 計算決策邊界的 Y 值
            y_vals = -(W[0] + W[1] * x_vals) / W[2]
        else:
            y_vals = np.array([ax.get_ylim()[0], ax.get_ylim()[1]])                        # 避免除以 0 的錯誤

        decision_boundary.set_data(x_vals, y_vals)                                         #更新決策邊界的X和Y值。
        return decision_boundary,

    ani = animation.FuncAnimation(fig, update, frames=len(weight_history), interval=10, repeat=False) #更新線條，頻率為每100毫秒

    plt.show()



#-------------------Parameter-------------------
learning_rate = 1
Weight = [ 0 , 0 , 1.1 ]
Max_iteration = 500

while(True): #簡易的防呆判斷
    try:
        number = int(input("Enter the worksheet number to view the results (0~2) :"))
        if number < 0 or number > 2:
            print("Please enter the correct range")
        elif number == 0:
            Max_iteration = 50
            break
        else:
            break

    except ValueError:
        print("Please enter only numbers")

if number == 2 : #使用非線性計算函數找出最佳權重
    Perceptron_nonliner(M2, learning_rate, Weight, Max_iteration)
else: #使用線性計算函數
    Perceptron(data[number], learning_rate, Weight, Max_iteration)

