import sys
import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import visualization

fig_width = 5.84
fig_height = 4.00 #4.38



# file should be named like: name_num_win_num_realization
# there will be 6 realizations 20 s apart
num_realizations = int(sys.argv[1])
time_diff = int(sys.argv[2])
num_win = int(sys.argv[3])

algorithms = ['opt', 'em', 'boem']
algorithms_display = ['opt.', 'EM', 'BOEM']

# read the csv files and write to separate csv files
raw_lists = [[] for i in range(len(algorithms))]
for m in range(len(algorithms)):
	for tw in range(time_diff, time_diff*num_win+1, time_diff):
		skip_len = 0
		for nr in range(1,num_realizations+1):
			raw_lists[m] = pd.read_csv("result/sim/exp_window/%s_%s.csv" % (algorithms[m], tw), skiprows=skip_len, nrows=tw*10)
			raw_lists[m].to_csv("result/sim/exp_window/%s_%s_%s.csv" % (algorithms[m], tw, nr), index=False, float_format='%.6f')
			skip_len+=(tw*10+1)
# 		os.remove("result/sim_exp_window/%s_%s.csv" % (algorithms[m], tw))

# get data from the txt files
result_lists = [[] for i in range(len(algorithms))]
time_lists = [[] for i in range(len(algorithms))]
sd_lists = [[] for i in range(len(algorithms))]

for str_index, str in enumerate(algorithms):
	with open("result/sim/exp_window/perf_%s.txt" % (str,)) as textFile:
		for line in textFile.readlines():
			index1 = line.find('task')
			if index1!= -1:
				line_2_phrase1 = line[:index1]
				index2 = line.find('( ')
				line_2_phrase2 = line[index2+1:index2+10]
				result_lists[str_index].append(line_2_phrase1+line_2_phrase2 )

	textFile.close()



for j in range(len(algorithms)):
	for tnw in range(num_win):
		findex = result_lists[j][tnw].find('+')
		time = result_lists[j][tnw][:findex]
		std = result_lists[j][tnw][findex+2:]
		time_lists[j].append(float(time.strip().replace(",", ""))/1000)
		sd_lists[j].append(float(std.strip())/100*float(time.strip().replace(",", ""))/1000)

# find RMSE error
gt_data = pd.read_csv("result/sim/exp_window/gt.csv")
error_array = np.zeros((len(algorithms), num_win, num_realizations))
for l in range(len(algorithms)):
	for h in range(num_win):
		for k in range(num_realizations):
			data = pd.read_csv("result/sim/exp_window/%s_%s_%s.csv" % (algorithms[l], (h+1)*time_diff, (k+1)))
			for i in range(len(data['p_x'])):
				error_array[l, h, k] += (gt_data['p_x'][i]-data['p_x'][i])**2 + (gt_data['p_y'][i]-data['p_y'][i])**2\
									   + (gt_data['p_z'][i]-data['p_z'][i])**2
			error_array[l, h, k] = np.sqrt(error_array[l, h, k]/len(data['p_x']))

# plot average over number of realizations with one half standard deviation
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)
line_width = 1.2
time_data = range(time_diff, time_diff*num_win+1, time_diff)

for n in range(len(algorithms)):
	ax1.fill_between(time_data, np.mean(error_array[n], axis=1) + np.std(error_array[n],axis=1)/2,
					            np.mean(error_array[n], axis=1) - np.std(error_array[n],axis=1)/2, 
					            color = visualization.color[algorithms[n]], alpha = 0.5, linewidth=0.0)
	ax2.fill_between(time_data, np.array(time_lists[n]) + np.array(sd_lists[n])/2, 
		                        np.array(time_lists[n]) - np.array(sd_lists[n])/2,
					            color = visualization.color[algorithms[n]], alpha = 0.5, linewidth=0.0)

for n in range(len(algorithms)):
	ax1.plot(time_data, np.mean(error_array[n], axis=1), '-x', 
		color = visualization.color[algorithms[n]], 
		linewidth=line_width, label=algorithms_display[n])
	ax2.plot(time_data, time_lists[n], '-x', 
		color = visualization.color[algorithms[n]], 
		linewidth=line_width)



ax1.set(ylabel='position RMSE [m]')
ax1.legend()
ax1.set_ylim(visualization.add_boarder((0, 0.04)))

ax2.semilogy()
ax2.set(ylabel='processing time [log(s)]')
ax2.set(xlabel='time [s]')
plt.show()

fig.savefig('result/sim/exp_window/sim_exp_win.png')






