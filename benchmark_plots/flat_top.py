import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py


data = pd.read_csv("../benchmarks_flat_top.txt")
times = data.iloc[1:-1:2].to_numpy()[:, 0]

for i in range(len(times)):
    times[i] = float(times[i][6:])

f = h5py.File("../data_flat top.h5")
data = f["Flat top"]
print(data.keys())
losses = data["losses"][:]
psi = data["potential"][:]
f.close()

plt.imshow(psi, cmap='RdBu')
plt.colorbar()
plt.show()
fig, ax1 = plt.subplots()
ax1.plot(losses, label='Loss', color='black')
ax1.set_xlabel("Iteration", fontsize='16')
ax1.set_ylabel("Loss", fontsize='16')
ax1.set_xlim(0, len(times))
ax1.tick_params(axis='both', labelsize=12)

ax2 = ax1.twinx()
ax2.plot(times, color='gray', linestyle=' ', marker='.', label='Time')
ax2.set_ylim(0, 200)
ax2.set_ylabel("Time [s]", fontsize='16')
ax2.tick_params(axis='both', labelsize=12)
fig.legend(loc=(0.17, 0.80), fontsize='14')
plt.tight_layout()
plt.show()

print(np.mean(times))
