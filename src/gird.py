import pandas as pd
para1min = 15
para1max = 35
para2min = 15
para2max = 35
para3min = 10
para3max = 20
file_path = r'E:\code\scenarioagentcnn\data0917\base\\13.csv'
fileWriter = open(file_path, 'w+')
t = 0
for h in range(para1min,para1max,1):
    for i in range(para2min,para2max,1):
        # print(i)
        for j in range(para3min,para3max,1):
        # if k_stop1 ==1 and w == su1x[k2] and h == su1y[k2] :
        #     k_stop1 = DrawBox(w,h,su1x,su1y,k_stop1,r_num,k2,filepath)
        # elif k_stop2==1 and w == su2x[k3] and h == su2y[k3]:
        #     k_stop2 = DrawBox(w,h,su2x,su2y,k_stop2,r_num,k3,filepath)
            print(h,",",i,",",j,",",file=fileWriter)
            t+=1

df = pd.read_csv(file_path,header=None)
total_row = len(df.iloc[:, 0])

x = [0]*3
for i in range(0,total_row-1):
    x[0] = df.iloc[i,0]
    x[1] = df.iloc[i,1]
    x[2] = df.iloc[i,2]
print(x)
print(total_row,t)