import pandas as pd
import numpy as np
path = r"C:\Users\abhis\OneDrive\Desktop\ML experiments\data.csv"
data = pd.read_csv(path)
d = data.iloc[:,:5]
target_d = data.iloc[:,6]
print(list(target_d))
def training(temp_d,target):
    for i,value in enumerate(target):
        if value == "yes":
            s_hypo = temp_d[i].copy()
            break
    for i,val in enumerate(temp_d):
        if target == "yes":
             for x in range(len(s_hypo)):
                 if val[x]!= s_hypo[x]:
                     s_hypo[x] = "~"
                 else:pass
        return s_hypo

testing = training(data,target_d)
print(testing)
