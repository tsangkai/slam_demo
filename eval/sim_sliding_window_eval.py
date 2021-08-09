import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import visualization


fig_width = 5.84
fig_height = 4.38

algorithms = ['opt', 'em', 'boem']
algorithms_display = ['opt.', 'EM', 'BOEM']

# error plot
num_realizations = int(sys.argv[1])
gt_data = pd.read_csv("result/sim/sliding_window/gt.csv")
p_error_array = np.zeros((len(algorithms), len(gt_data['p_x']), num_realizations))
q_error_array = np.zeros_like(p_error_array)


for l in range(len(algorithms)):
	for k in range(num_realizations):
		data = pd.read_csv("result/sim/sliding_window/%s_%s.csv" % (algorithms[l],k))
		for i in range(len(gt_data['p_x'])):
			p_error_array[l, i, k] = math.sqrt( (gt_data['p_x'][i]-data['p_x'][i])**2 + (gt_data['p_y'][i]-data['p_y'][i])**2
												 + (gt_data['p_z'][i]-data['p_z'][i])**2)
			q_error_array[l, i, k] = visualization.quat_diff([data['q_w'][i], data['q_x'][i], data['q_y'][i], data['q_z'][i]],
											   [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])


fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)
line_width = 1.2
for n in range(len(algorithms)):
	error_q_m =  np.mean(q_error_array[n], axis=1)
	error_q_s = np.std(q_error_array[n], axis=1)

	ax1.fill_between(gt_data['timestamp'], error_q_m - error_q_s/2,
					 error_q_m + error_q_s/2, color=visualization.color[algorithms[n]], alpha=0.5, linewidth=0.0)

	error_p_m =  np.mean(p_error_array[n], axis=1)
	error_p_s = np.std(p_error_array[n], axis=1)
	ax2.fill_between(gt_data['timestamp'], error_p_m - error_p_s/2,
					 error_p_m + error_p_s/2, color=visualization.color[algorithms[n]], alpha=0.5, linewidth=0.0)

	ax1.plot(gt_data['timestamp'], error_q_m, color=visualization.color[algorithms[n]], linewidth=line_width, label=algorithms_display[n])
	ax2.plot(gt_data['timestamp'], error_p_m, color=visualization.color[algorithms[n]], linewidth=line_width, label=algorithms_display[n])


ax1.set(ylabel='rotation RMSE [deg]')
ax1.set_ylim(visualization.add_boarder((0, 3)))
ax1.legend(loc = 1)

ax2.set(ylabel='position RMSE [m]')
ax2.set(xlabel='time [s]')
ax2.set_ylim(visualization.add_boarder((0, 0.4)))

plt.show()

