import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''last = -1
counter = -1
capture = 1
df = pd.read_csv('input.csv', sep=',')
for i in range(0,df.shape[0]):
    #print("last: "+str(last)+" | counter: "+str(counter))
    if (df['capture'][i] == capture and df['measure'][i] == last):  # and df['measure'][i]==0):
        # print(df['measure'][i],end=", ")
        counter += 1
    else:
        # if (df['measure'][i]==50):
        if df['measure'][i] != last and df['capture'][i] == capture:
            if (last != -1 and counter != -1):
                print("df['capture'] = "+str(capture)+" | df['measure']: " + str(last) + " = " + str(counter))
            last = df['measure'][i]
            counter = 1

print("df['capture'] = "+str(capture)+" | df['measure']: " + str(last) + " = " + str(counter))'''

counter0 = 0
counter1 = 0

df = pd.read_csv('../inputs/input.csv', sep=',')
for i in range(0,df.shape[0]):
    if df['stalling_event'][i] == 0:
        counter0+=1
    else:
        counter1+=1

print("counter 0: "+str(counter0))
print("counter 1: "+str(counter1))