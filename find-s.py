import pandas as pd
import numpy as np
path = r"C:\Users\abhis\OneDrive\Desktop\ML experiments\data.csv"
data = pd.read_csv(path)
d = np.array(data)[:,:-1]
target_d = np.array(data)[:,-1]

def train(c,t):
    for i,val in enumerate(t):
        if val == "yes":
            s_hypo = c[i].copy()
            break
    for i,val in enumerate(c):
        if t[i] == "yes":
            for x in range(len(s_hypo)):
                if val[x]!= s_hypo[x]:
                    s_hypo[x] = "*?*"
    return s_hypo

test = train(d,target_d)
print(test)

                
            
