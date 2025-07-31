import numpy as np
import matplotlib.pyplot as plt
import pandas as pd     #匯入需要用到的套件


file_path = "Homework6.xlsx"  
df_weight_in_hid = pd.read_excel(file_path, sheet_name="Weight_In_Hid", header=None)
df_weight_hid_out = pd.read_excel(file_path, sheet_name="Weight_Hid_Out", header=None) # 讀取 Excel 檔案


weight_in_hid_clean = df_weight_in_hid.iloc[1:, 1:]  # 清除第一行和第一列
weight_in_hid_clean = np.array(weight_in_hid_clean, dtype=float)  # 將數據轉換為浮點數np陣列型態


weight_hid_out_clean = df_weight_hid_out.iloc[1:, 1:]
weight_hid_out_clean = np.array(weight_hid_out_clean, dtype=float)  

# 輸出清理後的數據
print(f"輸入至隱藏層權重:\n{weight_in_hid_clean}")
print(f"隱藏至輸出層權重:\n{ weight_hid_out_clean}")

def sigmoid(x):         # Sigmoid 函數
    return 1 / (1 + np.exp(-x))

k = float(input("\n請輸入x值："))     # 輸入x
hidden_outputs = []        # 儲存隱藏層的輸出結果

for i in range(weight_in_hid_clean.shape[1]): # 隱藏層神經元數量
    b = weight_in_hid_clean[0,i]    # 第一行是bias  
    w = weight_in_hid_clean[1,i]        # 第二行是weight
    out = sigmoid(b + w * k)    # 計算net
    hidden_outputs.append(out)  # 將每個隱藏層的輸出結果放入hidden_outputs裡面

b_out = weight_hid_out_clean[0,0]  # 輸出層部分 第一行是bias
w_out = weight_hid_out_clean[1:,0] # 第二行是weight
final_output = b_out + np.dot(w_out, hidden_outputs)    # 內積+bias結果

print(f"\nBPN 輸出結果：{final_output:.4f}")  # 輸出結果






#繪圖
x_input = np.linspace(-10,10,100)  # 產生從-10到10的100個點
output_list = []    # 儲存輸出結果
hidden_outputs_all = []     # 儲存隱藏層輸出結果

for x in x_input:                                   
    hidden_outputs = []                             #前
    for i in range(weight_in_hid_clean.shape[1]):   #面
        bias = weight_in_hid_clean[0,i]             #複
        weight = weight_in_hid_clean[1,i]           #製
        net = bias + weight * x                     #貼                                                     
        out = sigmoid(net)                          #上
        hidden_outputs.append(out)                  

    hidden_outputs_all.append(hidden_outputs)       # 轉到hidden_outputs_all

    bias_out = weight_hid_out_clean[0,0]           #輸出層部分 第一行是bias
    weights_out = weight_hid_out_clean[1:,0]       #第二行是weight
    final_out = bias_out + np.dot(weights_out,hidden_outputs)  # 內積+bias結果
    output_list.append(final_out)                   # 將每個輸出結果放入output_list裡面


hidden_outputs_all = np.array(hidden_outputs_all).T  # 轉置後畫圖才不會出問題，因為要對應到整個 x_input

# 子圖數量（2+1）
num_hidden = hidden_outputs_all.shape[0] # 隱藏層數量(2)
fig, axs = plt.subplots(num_hidden+1,1,figsize=(10,7))  #3列1行的圖,每個圖的大小為10*7


for i in range(num_hidden):         # 隱藏層參數
    axs[i].plot(x_input, hidden_outputs_all[i], label=f"Hidden Layer {i+1}", color='green')
    axs[i].set_title(f"Hidden Layer {i+1}")
    axs[i].set_xlabel("x")
    axs[i].set_ylabel(f"h{i+1}")
    axs[i].grid(True)
    axs[i].legend()


axs[-1].plot(x_input, output_list, label="BPN Output", color='red')  # 輸出層參數
axs[-1].set_title(" BPN ")
axs[-1].set_xlabel("x")
axs[-1].set_ylabel("y")
axs[-1].grid(True)
axs[-1].legend()

plt.tight_layout()
plt.show()
