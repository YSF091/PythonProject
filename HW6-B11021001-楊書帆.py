import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 Excel 檔案
folder = os.path.dirname( os.path.realpath( __file__ ) )                                          #找出當前python腳本的位置
data = os.path.join( folder , "Homework6.xlsx" )                                                  #用folder找出的路徑結合EXCEL檔案，成為一個完整路徑，確保檔案可以被找到，且跨平台使用

w_input_hidden = pd.read_excel( data , sheet_name = 0 , header = 0 ).iloc[:, 1:].to_numpy()       #讀取第一張資料表，將第一行隱藏，使用iloc從第二行開始讀取，跳過第一行引索，再將dataframe格式轉換成陣列
w_hidden_output = pd.read_excel( data , sheet_name = 1 , header = 0 ).iloc[:, 1:].to_numpy()      #讀取第二張資料表，將第一行隱藏，使用iloc從第二行開始讀取，跳過第一行引索，再將dataframe格式轉換成陣列

print("\nWeight_In_Hid:\n", w_input_hidden)                                                       #印出w_input_hidden
print("\nWeight_Hid_Out:\n", w_hidden_output)                                                     #印出w_hidden_output
w_input_hidden = w_input_hidden.T                                                                 #print出的權重方向相反，將矩陣轉置

def sigmoid(net):                                                                                 #定義sigmoid函數
    return 1 / (1 + np.exp(-net))

x1 = float(input("\nInput Data x1: "))                                                            #使用者輸入X1
x_input = np.array([1, x1])

net_h1 = 0  # 初始化 net_h1
for i in range(len(w_input_hidden[0])):
    net_h1 += w_input_hidden[0][i] * x_input[i]
h1 = sigmoid(net_h1)
print("\nh1:", h1)

net_h2 = 0  # 初始化 net_h2
for i in range(len(w_input_hidden[1])):
    net_h2 += w_input_hidden[1][i] * x_input[i]
h2 = sigmoid(net_h2)
print("\nh2:", h2)

H = np.array([1, h1, h2])
print("\nH:", H)

net_y = 0
for i in range(len(H)):
    net_y += w_hidden_output[i] * H[i]
print("\nOutput_y:",net_y)


# 繪製 Sigmoid 函數圖像
x_vals = np.linspace(-10, 10, 100)
y_vals = sigmoid(x_vals)

plt.figure(figsize=(6, 6))

# h1
h1_plot = sigmoid(x_vals * w_input_hidden[0][1] + w_input_hidden[0][0])
plt.subplot(3, 1, 1)
plt.plot(x_vals, h1_plot, label="h1 Activation")
plt.axhline(y=h1, color='r', linestyle='--', label=f'h1={h1:.4f}')                              #以紅色水平虛線顯示所輸入x1所對應的h1值
plt.xlabel('x')
plt.ylabel('h1')
plt.legend()

# h2
h2_plot = sigmoid(x_vals * w_input_hidden[1][1] + w_input_hidden[1][0])
plt.subplot(3, 1, 2)
plt.plot(x_vals, h2_plot, label="h2 Activation")
plt.axhline(y=h2, color='r', linestyle='--', label=f'h2={h2:.4f}')  # 水平線顯示 h2 值
plt.xlabel('x')
plt.ylabel('h2')
plt.legend()

# y
y_plot = w_hidden_output[0] * H[0] + w_hidden_output[1] * h1_plot + w_hidden_output[2] * h2_plot
plt.subplot(3, 1, 3)
plt.plot(x_vals, y_plot, label="y Activation")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
