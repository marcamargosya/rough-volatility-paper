import pandas as pd
import matplotlib.pyplot as plt

spx = pd.read_csv("spx.csv", skiprows=[1,2], index_col=0, parse_dates=True)
vix = pd.read_csv("vix.csv", skiprows=[1,2], index_col=0, parse_dates=True)

fig, ax1 = plt.subplots()
ax1.plot(spx.index, spx["Close"], color="tab:blue")
ax1.set_ylabel("SPX", color="tab:blue")

ax2 = ax1.twinx()
ax2.plot(vix.index, vix["Close"], color="tab:red")
ax2.set_ylabel("VIX", color="tab:red")

plt.title("SPX vs VIX")
plt.show()