import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import os

#print(os.getcwd())

df_general = pd.DataFrame()
delay = {0,25,50,75,100,125}
os.chdir("../inputs/xlsx")
for k in range(0,len(delay)):#for # para cada retardo
#for k in range(0,2):#for # para cada retardo
    os.chdir("../../inputs/xlsx")
    xlsxFiles = os.listdir()
    print(xlsxFiles)
    sheetNames = pd.ExcelFile(str(xlsxFiles[k])).sheet_names
    #print(xlsxFiles)
    #print(sheetNames)
    #os.chdir('C:/Users/Usuario/PycharmProjects/StallingEvents/inputs/csv/'+str(xlsxFiles[k][:-5]))
    csvs = [1,2,3,4,5,6,7,8,9,10]
    for j in range(0,len(sheetNames)):#for para cada simulación (1-10) dentro de cada retardo
    #for j in range(3,6):#for para cada simulación (1-10) dentro de cada retardo
        os.chdir('../xlsx')
        print(sheetNames)
        aux = xlsxFiles[k]
        print(str(xlsxFiles[k])+" | "+str(sheetNames[j])+" | ", end="")
        ts = np.array(pd.read_excel(aux,sheet_name=sheetNames[j],usecols='C', header=None)) #Modificar Sheet_name
        ts2 = np.append(ts[2:],ts[0:1]).tolist()
        ts3 = []
        st_ev = []
        for i in range(0,len(ts2)):
            ts3.append((datetime.combine(date.min,ts2[i])-datetime.min).total_seconds())
            if i%2 == 0 and i>0:
                st_ev.append([ts3[i-1],ts3[i]])
        st_ev = np.array(st_ev)
        #print(ts3)
        #print(st_ev)


        prebuf = ts3[0]
        endS = ts3[-1]
        #print(prebuf)
        #print(endS)

        os.chdir('../csv')
        csvsFile = str(xlsxFiles[k][:-5])+'/'+str(csvs[j])+".csv"
        print(csvsFile)
        df = pd.read_csv(csvsFile, sep=',',error_bad_lines=False)

        df['tcp_prev_NC'] = df['tcp_prev_NC'].fillna(0)
        df['tcp_ack_dup'] = df['tcp_ack_dup'].fillna(0)
        df['tcp_fast_ret'] = df['tcp_fast_ret'].fillna(0)
        df['tcp_pkt_ooo'] = df['tcp_pkt_ooo'].fillna(0)
        #print(df.columns)

        indexValues = df[(df['Source'] == '00:00:00_00:00:08') | (df['Destination'] == '00:00:00_00:00:08') |
                         (df['Destination'] == 'Broadcast') | (df['Source'] == 'CompalIn_cb:c3:39') |
                         (df['Source'] == 'fe80::1935:4432:c2e0:8ebb') | (df['Destination'] == 'CompalIn_cb:c3:39') |
                         (df['Destination'] == '239.255.255.250') | (df['Destination'] == '200.100.0.255')].index

        #print(indexValues.values)
        df['out_of_band'] = 0
        df['out_of_band'] = np.where((df['Source'] == '00:00:00_00:00:08') | (df['Destination'] == '00:00:00_00:00:08') |
                         (df['Destination'] == 'Broadcast') | (df['Source'] == 'CompalIn_cb:c3:39') |
                         (df['Source'] == 'fe80::1935:4432:c2e0:8ebb') | (df['Destination'] == 'CompalIn_cb:c3:39') |
                         (df['Destination'] == '239.255.255.250') | (df['Destination'] == '200.100.0.255'),1,0)

        df['Traffic'] = np.where((df['Source'] == '200.100.0.3') & (df['Destination'].values == '200.100.1.3'),1,0)

        df[['tcp_seq_num','tcp_win_si2','ip_len']] = df[['tcp_seq_num','tcp_win_si2','ip_len']].fillna(value=-1)
        df[['tcp_hea_len','tcp_flag_ack','tcp_flag_urg','tcp_flag_psh','tcp_flag_rst','tcp_flag_syn','tcp_flag_fin','tcp_win_siz']] = \
            df[['tcp_hea_len','tcp_flag_ack','tcp_flag_urg','tcp_flag_psh','tcp_flag_rst','tcp_flag_syn','tcp_flag_fin','tcp_win_siz']].fillna(value=-1)


        deltaT = np.array([(df.Time[m + 1] - df.Time[m]) for m in range(len(df)-1)])
        deltaT = np.around(deltaT,decimals=4)
        deltaT = np.concatenate((np.array([0]), deltaT))

        #df['Time'] = np.around(df['Time'],decimals=4)
        df.insert(2, 'At', deltaT)
        #df['Protocol'] = df['Protocol'].map(proto)
        df['delay'] = xlsxFiles[k][4:-5]
        df['capture'] = csvs[j]
        df['prebuffering'] = 0
        df['stalling_event'] = 0

        t = np.array(df['Time'].values.tolist())
        df['prebuffering'] = np.where(t<prebuf,1,0).tolist()
        #print(len(st_ev))
        #print(st_ev)
        for i in range(0,len(st_ev)):
            df['stalling_event'] = np.where(np.logical_and(t>st_ev[i][0],t<st_ev[i][1]),1,df['stalling_event'])
        #print(df['stalling_event'].tolist().count(1))
        #print(df)
        '''print(df.columns)'''
        #print(df['Protocol'].unique())

        #print('../output/'+str(xlsxFiles[k][:-5])+'/'+str(csvs[j])+'.csv')
        #print(df)

        #df = df.drop(indexValues)
        df = df.drop(labels=['Source', 'Destination', 'No.','Time'], axis=1)
        df_general = pd.concat([df_general,df], ignore_index=True)
        print(df[pd.isnull(df).any(axis=1)].to_string())
        df = df.loc[:,df.columns!='At'].astype(int)
        df.to_csv('../../output/'+str(xlsxFiles[k][:-5])+'/'+str(csvs[j])+'.csv',sep=',',header=True,index=False,encoding='utf-8')
        print(df_general.shape)
'''
indexValues = df_general[(df_general['Source'] == '00:00:00_00:00:08') | (df_general['Destination'] == '00:00:00_00:00:08') |
                 (df_general['Destination'] == 'Broadcast') | (df_general['Source'] == 'CompalIn_cb:c3:39') |
                 (df_general['Destination'] == 'CompalIn_cb:c3:39')].index

df_general = df_general.drop(indexValues.tolist(), inplace=True)
df_general = df_general.drop(labels=['Source', 'Destination', 'No.'], axis=1)'''

print(df_general)
df_general.to_csv('../../output/df_general.csv', sep=',', header=True, index=False, encoding='utf-8')
