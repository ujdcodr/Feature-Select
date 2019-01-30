import pandas as pd
import numpy as np
import time
np.set_printoptions(suppress=True)

df = pd.read_csv('kddcup.data.corrected')
print df.columns

df.flag = pd.Categorical(df.flag)
df['flag'] = df.flag.cat.codes

df.protocol_type = pd.Categorical(df.protocol_type)
df['protocol_type'] = df.protocol_type.cat.codes

df.service = pd.Categorical(df.service)
df['service'] = df.service.cat.codes
df.attack = pd.Categorical(df.attack)
df['attack'] = df.attack.cat.codes

target = np.array(df['attack'])

sel = [1,2,3,4,5,11,22 ,23 ,24, 25, 28, 31, 32, 33, 35, 37, 38]
new_df = pd.DataFrame()
features = list()
for s in sel:
	features.append(df.columns[s])

new_df = df[features].copy()
new_df = new_df.assign(attack=target)
new_df.to_csv('fullreducedL3.csv',index=False)

