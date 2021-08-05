import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def add_boarder(lim):
	coeff = 0.02
	diff = lim[1] - lim[0]
	return (lim[0]-coeff*diff, lim[1]+coeff*diff)

def quat_diff(q1, q2):
	[w1, x1, y1, z1] = q1
	[w2, x2, y2, z2] = q2

	del_q = w1*w2 + x1*x2 + y1*y2 + z1*z2
	del_x = w1*x2 - x1*w2 + y1*z2 - z1*y2
	del_y = w1*y2 - y1*w2 - x1*z2 + z1*x2
	del_z = w1*z2 - w2*z1 + x1*y2 - y1*x2
	del_xyz = [del_x, del_y, del_z]
	theta = 2*np.arctan2(np.linalg.norm(del_xyz), np.abs(del_q))
	theta = (180*theta/np.pi)

	return theta


plot_color = {
	'gt': [0, 0, 0],
	'dr': [0.0000, 0.5490, 0.3765],
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],
	'lmk': [0.1, 0.1490, 0.3765]
}

fig_width = 5.84
fig_height = 4.38

# keyframe has larger time range than ground truth

gt_data = pd.read_csv("result/sim/easy/gt.csv")
dr_data = pd.read_csv("result/sim/easy/dr.csv")
opt_data = pd.read_csv("result/sim/easy/opt.csv")
em_data = pd.read_csv("result/sim/easy/em.csv")
boem_data = pd.read_csv("result/sim/easy/boem.csv")


dr_rot_error = np.zeros_like(gt_data['p_x'])
opt_rot_error = np.zeros_like(gt_data['p_x'])
em_rot_error = np.zeros_like(gt_data['p_x'])
boem_rot_error = np.zeros_like(gt_data['p_x'])

dr_pos_error = np.zeros_like(gt_data['p_x'])
opt_pos_error = np.zeros_like(gt_data['p_x'])
em_pos_error = np.zeros_like(gt_data['p_x'])
boem_pos_error = np.zeros_like(gt_data['p_x'])



for i in range(len(gt_data['p_x'])):

	dr_rot_error[i] = quat_diff([dr_data['q_w'][i], dr_data['q_x'][i], dr_data['q_y'][i], dr_data['q_z'][i]],
		                         [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
	opt_rot_error[i] = quat_diff([opt_data['q_w'][i], opt_data['q_x'][i], opt_data['q_y'][i], opt_data['q_z'][i]],
		                         [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
	em_rot_error[i] = quat_diff([em_data['q_w'][i], em_data['q_x'][i], em_data['q_y'][i], em_data['q_z'][i]],
		                         [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
	boem_rot_error[i] = quat_diff([boem_data['q_w'][i], boem_data['q_x'][i], boem_data['q_y'][i], boem_data['q_z'][i]],
		                         [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])


	dr_pos_error[i] = math.sqrt( (gt_data['p_x'][i]-dr_data['p_x'][i])**2 + (gt_data['p_y'][i]-dr_data['p_y'][i])**2 + (gt_data['p_z'][i]-dr_data['p_z'][i])**2)
	opt_pos_error[i] = math.sqrt( (gt_data['p_x'][i]-opt_data['p_x'][i])**2 + (gt_data['p_y'][i]-opt_data['p_y'][i])**2 + (gt_data['p_z'][i]-opt_data['p_z'][i])**2)
	em_pos_error[i] = math.sqrt( (gt_data['p_x'][i]-em_data['p_x'][i])**2 + (gt_data['p_y'][i]-em_data['p_y'][i])**2 + (gt_data['p_z'][i]-em_data['p_z'][i])**2)
	boem_pos_error[i] = math.sqrt( (gt_data['p_x'][i]-boem_data['p_x'][i])**2 + (gt_data['p_y'][i]-boem_data['p_y'][i])**2 + (gt_data['p_z'][i]-boem_data['p_z'][i])**2)



fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)

line_width = 1.2
ax1.plot(dr_data['timestamp'], dr_rot_error, color = plot_color['dr'], label='dead reckoning')
ax1.plot(opt_data['timestamp'], opt_rot_error, color = plot_color['opt'], label='opt.')
ax1.plot(em_data['timestamp'], em_rot_error, color = plot_color['em'], label='EM')
ax1.plot(boem_data['timestamp'], boem_rot_error, color = plot_color['boem'], label='BOEM')
ax1.set(ylabel='rotation RMSE [deg]')
ax1.set_ylim(add_boarder((0, 2)))
ax1.legend(loc = 1)


ax2.plot(dr_data['timestamp'], dr_pos_error, color = plot_color['dr'])
ax2.plot(opt_data['timestamp'], opt_pos_error, color = plot_color['opt'])
ax2.plot(em_data['timestamp'], em_pos_error, color = plot_color['em'])
ax2.plot(boem_data['timestamp'], boem_pos_error, color = plot_color['boem'])
ax2.set(ylabel='position RMSE [m]')
ax2.set(xlabel='time [s]')
ax2.set_ylim(add_boarder((0, 0.1)))

plt.show()