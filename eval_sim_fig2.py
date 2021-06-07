import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

plot_color = {
	'gt': [0, 0, 0],
	'dr': [0.0000, 0.5490, 0.3765],
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],
	'lmk': [0.1, 0.1490, 0.3765],
}

fig_width = 5.84
fig_height = 4.38



# error plot
gt_data = pd.read_csv("result/sim/long_traj/gt.csv")

dr_error_p = np.zeros_like(gt_data['p_x'])
est_opt_error_p = np.zeros_like(gt_data['p_x'])
est_em_error_p = np.zeros_like(gt_data['p_x'])
est_boem_error_p = np.zeros_like(gt_data['p_x'])

dr_error_q = np.zeros_like(gt_data['p_x'])
est_opt_error_q = np.zeros_like(gt_data['p_x'])
est_em_error_q = np.zeros_like(gt_data['p_x'])
est_boem_error_q = np.zeros_like(gt_data['p_x'])

def quat_rot_angle(q2, x2, y2, z2, i):
	q1, x1, y1, z1 = gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]
	del_q = q1*q2 + x1*x2 + y1*y2 + z1*z2
	del_x = q1*x2 - x1*q2 + y1*z2 - z1*y2
	del_y = q1*y2 - y1*q2 - x1*z2 + z1*x2
	del_z = q1*z2 - q2*z1 + x1*y2 - y1*x2
	del_xyz = [del_x, del_y, del_z]
	theta = 2*np.arctan2(np.linalg.norm(del_xyz), np.abs(del_q))
	theta = (180*theta/np.pi)
	return theta*theta

for i in range(len(gt_data['p_x'])):
	dr_error_list_p = []
	est_opt_error_list_p = []
	est_em_error_list_p = []
	est_boem_error_list_p = []

	dr_error_list_q = []
	est_opt_error_list_q = []
	est_em_error_list_q = []
	est_boem_error_list_q = []

	for k in range(0,50):
		dr_data = pd.read_csv("result/sim/long_traj/dr_%s.csv" %k)
		est_opt_data = pd.read_csv("result/sim/long_traj/opt_%s.csv" %k)
		est_em_data = pd.read_csv("result/sim/long_traj_bug_fix/em_%s.csv" %k)
		est_boem_data = pd.read_csv("result/sim/long_traj_bug_fix/boem_%s.csv" %k)

		dr_error_list_p.extend([(gt_data['p_x'][i]-dr_data['p_x'][i])**2, (gt_data['p_y'][i]-dr_data['p_y'][i])**2, (gt_data['p_z'][i]-dr_data['p_z'][i])**2])
		est_opt_error_list_p.extend([(gt_data['p_x'][i]-est_opt_data['p_x'][i])**2, (gt_data['p_y'][i]-est_opt_data['p_y'][i])**2, (gt_data['p_z'][i]-est_opt_data['p_z'][i])**2])
		est_em_error_list_p.extend([(gt_data['p_x'][i]-est_em_data['p_x'][i])**2, (gt_data['p_y'][i]-est_em_data['p_y'][i])**2, (gt_data['p_z'][i]-est_em_data['p_z'][i])**2])
		est_boem_error_list_p.extend([(gt_data['p_x'][i]-est_boem_data['p_x'][i])**2, (gt_data['p_y'][i]-est_boem_data['p_y'][i])**2, (gt_data['p_z'][i]-est_boem_data['p_z'][i])**2])

		dr_error_list_q.append(quat_rot_angle(dr_data['q_w'][i], dr_data['q_x'][i], dr_data['q_y'][i], dr_data['q_z'][i], i))
		est_opt_error_list_q.append(quat_rot_angle(est_opt_data['q_w'][i], est_opt_data['q_x'][i], est_opt_data['q_y'][i], est_opt_data['q_z'][i], i))
		est_em_error_list_q.append(quat_rot_angle(est_em_data['q_w'][i], est_em_data['q_x'][i], est_em_data['q_y'][i], est_em_data['q_z'][i], i))
		est_boem_error_list_q.append(quat_rot_angle(est_boem_data['q_w'][i], est_boem_data['q_x'][i], est_boem_data['q_y'][i], est_boem_data['q_z'][i], i))


	dr_error_p[i] = math.sqrt(sum(dr_error_list_p)/len(dr_error_list_p))
	est_opt_error_p[i] = math.sqrt(sum(est_opt_error_list_p)/len(est_opt_error_list_p))
	est_em_error_p[i] = math.sqrt(sum(est_em_error_list_p)/len(est_em_error_list_p))
	est_boem_error_p[i] = math.sqrt(sum(est_boem_error_list_p)/len(est_boem_error_list_p))

	dr_error_q[i] = math.sqrt(sum(dr_error_list_q)/len(dr_error_list_q))
	est_opt_error_q[i] = math.sqrt(sum(est_opt_error_list_q)/len(est_opt_error_list_q))
	est_em_error_q[i] = math.sqrt(sum(est_em_error_list_q)/len(est_em_error_list_q))
	est_boem_error_q[i] = math.sqrt(sum(est_boem_error_list_q)/len(est_boem_error_list_q))

fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)
line_width = 1.2
ax1.plot(gt_data['timestamp'], dr_error_q, color = plot_color['dr'], linewidth=line_width, label='dead reckoning')
ax1.plot(gt_data['timestamp'], est_opt_error_q, color = plot_color['opt'], linewidth=line_width, label='opt.')
ax1.plot(gt_data['timestamp'], est_em_error_q, color = plot_color['em'], linewidth=line_width, label='EM')
ax1.plot(gt_data['timestamp'], est_boem_error_q, color = plot_color['boem'], linewidth=line_width, label='BOEM')
ax1.set(ylabel='rotation RMSE [deg]')
ax1.set_ylim(-0.2, 12.1)
ax1.legend(loc = 1)


ax2.plot(gt_data['timestamp'], dr_error_p, color = plot_color['dr'], linewidth=line_width, label='dr')
ax2.plot(gt_data['timestamp'], est_opt_error_p, color = plot_color['opt'], linewidth=line_width, label='opt.')
ax2.plot(gt_data['timestamp'], est_em_error_p, color = plot_color['em'], linewidth=line_width, label='EM')
ax2.plot(gt_data['timestamp'], est_boem_error_p, color = plot_color['boem'], linewidth=line_width, label='BOEM')
ax2.set(ylabel='position RMSE [m]')
ax2.set(xlabel='time [s]')
ax2.set_ylim(-0.01, 0.7)
plt.show()




