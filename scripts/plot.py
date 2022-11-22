#%%
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("temp.csv", ',')

plt.plot(df.values[:,0], df.values[:,1])
plt.plot(df.values[:,0], df.values[:,2])
plt.plot(df.values[:,0], df.values[:,3])
plt.grid(True)
plt.legend(('1', '2', '3'))
plt.show()
