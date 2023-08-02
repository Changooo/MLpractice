import csv
import pandas as pd

# 저수위	방수로	저수량	저수율	유입량	공용량	총 방류량	발전	여수로	기타	취수량
# ["level", "1", "contain", "2", "income", "3", "outcome", "4", "5", "6", "7"]

paldang_level = pd.DataFrame()

for i in range(23):
    data = pd.read_csv("./dataset/p{}.csv".format(i+1), header=None)
    data.columns = ["level", "1", "contain", "2", "income", "3", "outcome", "4", "5", "6", "7"]
    paldang_level = pd.concat([paldang_level, data[:-1]], axis=0)
    print(len(data))

jamsoo_level = pd.read_csv("./dataset/jamsoo_level.csv", header=None)
jamsoo_level.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
songjeong_rain = pd.read_csv("./dataset/songjeong_rain.csv", header=None)
songjeong_rain.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"] 
print(paldang_level)
print(songjeong_rain)
print(jamsoo_level)




