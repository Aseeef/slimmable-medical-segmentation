import numpy as np
from matplotlib import pyplot as plt


# Interpolate missing values to create smooth connecting lines
def interpolate_missing(x, y):
    x_arr = np.array(x)
    y_arr = np.array([v if v is not None else np.nan for v in y])
    mask = ~np.isnan(y_arr)
    return x_arr[mask], y_arr[mask]

# Data
channels = [3, 6, 8, 10, 13, 17, 20, 23, 25, 27, 30, 34]
ours = [0.4497, None, 0.9350885153, None, None, 0.9462072253, None, None, None, None, None, 0.9582130909]
paper = [None, None, None, None, None, 0.9343, None, None, None, None, None, 0.9502]
slim_nets = [None, None, 0.8287, None, None, 0.9226, None, None, 0.937, None, None, 0.9386]
kd_1_slim_nets = [None, None, 0.8811, None, None, 0.9298, None, None, 0.935, None, None, 0.9259]
kd_3_slim_nets = [None, None, 0.9068, None, None, 0.9282, None, None, 0.9326, None, None, 0.9312]
adam_slim_nets = [None, None, 0.9146, None, None, 0.9325, None, None, 0.9324, None, None, 0.9327]
us_nets = [0.9092, 0.7634, None, 0.8772, 0.9043, 0.91, 0.9056, 0.9288, None, 0.9267, 0.9287, 0.933]

x_ours, y_ours = interpolate_missing(channels, ours)
x_paper, y_paper = interpolate_missing(channels, paper)
#x_snn, y_snn = interpolate_missing(channels, slim_nets)
#x_kd1, y_kd1 = interpolate_missing(channels, kd_1_slim_nets)
#x_kd3, y_kd3 = interpolate_missing(channels, kd_3_slim_nets)
x_adam, y_adam = interpolate_missing(channels, adam_slim_nets)
x_us_net, y_us_net = interpolate_missing(channels, us_nets)

# Plot with connected lines
plt.figure(figsize=(10, 6))
plt.plot(x_ours, y_ours, marker='o', label='Individually Trained (Us)')
plt.plot(x_paper, y_paper, marker='s', label='Individually Trained (Paper)')
#plt.plot(x_snn, y_snn, marker='^', label='Slimmable Neural Network')
#plt.plot(x_kd1, y_kd1, marker='v', label='Slim Nets w/ KD Training (t=1.0)')
#plt.plot(x_kd3, y_kd3, marker='v', label='Slim Nets w/ KD Training (t=1.0)')
plt.plot(x_adam, y_adam, marker='v', label='Slim Nets w/ KD Training (t=3.0) + Adam')
plt.plot(x_us_net, y_us_net, marker='v', label='US-Nets w/ KD Training (t=3.0) + Adam')
plt.xlabel('Channels', fontsize=14)
plt.ylabel('DICE Score', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0.4, 1.0)
plt.legend(fontsize=16)
plt.title('Individually Trained vs the Slim Nets')
plt.grid(True)

# Single legend call: increase text size, and optionally tweak handle length
leg = plt.legend(
    fontsize=16,         # label font size
    handlelength=2.5,    # length of the legend lines
    borderpad=0.8,
    labelspacing=0.5
)

# If you want the legend lines themselves even thicker than your plotted lines:
for legline in leg.get_lines():
    legline.set_linewidth(4)

plt.tight_layout()
plt.show()
