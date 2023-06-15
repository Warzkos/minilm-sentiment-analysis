import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output_L12.csv', header=None)
df=df.transpose()

new_l = []
for i in range(len(df.columns)):
    new_l.extend(df[i][:])

df = pd.DataFrame(new_l)
print(df.shape)
print(df)

df = df.iloc[0::200]

print(df)
print(df.shape)

plt.figure(); df.plot()
plt.savefig('loss_L12.png')