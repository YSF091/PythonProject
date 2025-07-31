import numpy as np
import pandas as pd
# 匯入excel
List = pd.read_excel("排程最佳化.xlsx", sheet_name="5j5m", usecols=range(1, 6))
List = List.values
print(List)

R, C = np.shape(List)
Gantt = np.zeros([R, C * 2])                                        # 甘特圖
Gantt[0, 0] = 0                                                     # 甘特圖的1s為第0秒開始
Gantt[0, 1] = Gantt[0, 0] + List[0, 0]                              # 1e為1s那格+List第一格的時間

for i in range(1, C):
    Gantt[0, 2 * i] = Gantt[0, (2 * i) - 1]                         # s行：2 * 表示一次要完成兩個，下一個會等於前面那個的
    Gantt[0, (2 * i) + 1] = Gantt[0, 2 * i] + List[0, i]            # e行：s行 + 任務需要時間

for i in range(1, R):                                               # R多少個任務
    Gantt[i, 0] = Gantt[i - 1, 1]                                   # 第一欄就是前一個任務R完成的時間
    Gantt[i, 1] = Gantt[i, 0] + List[i, 0]                          #
    for j in range(1, C):
        if Gantt[i, 2 * j - 1] >= Gantt[i - 1, 2 * j + 1]:          # 若前一個工作站結束時間比較大
            Gantt[i, 2 * j] = Gantt[i, 2 * j - 1]                   # 取代
        else:
            Gantt[i, 2 * j] = Gantt[i - 1, 2 * j + 1]               # 沒有比較大，原本執行完的時間就會等於下一個的時間
        Gantt[i, 2 * j + 1] = Gantt[i, 2 * j] + List[i, j]          # 比對完的s，加上工作時間，等於e

print(Gantt)
print("Total Work Time：", Gantt[-1, -1])
