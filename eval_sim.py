import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

plot_color = {
	'gt': [0, 0, 0],
	'vo': [0.0000, 0.5490, 0.3765], # spruce
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],  # navy
}


fig_width = 5.84
fig_height = 4.38

# keyframe has larger time range than ground truth


gt_data = pd.read_csv("result/sim/gt.csv")
dr_data = pd.read_csv("result/sim/dr.csv")
est_opt_data = pd.read_csv("result/sim/opt.csv")
#est_em_data = pd.read_csv("result/" + dataset + "/traj_em_x.csv")
#est_boem_data = pd.read_csv("result/" + dataset + "/traj_boem_x.csv")






fig = plt.figure(1)
fig.set_size_inches(fig_width, fig_height)

#ax = fig.gca(projection='3d')
ax = Axes3D(fig, rect=(0.0, 0.05, 1, 0.9))


# ax = fig.add_axes((0.1, 0.1, 0.1, 0.1))


line_width = 1.2

ax.plot(gt_data['p_x'], gt_data['p_y'], gt_data['p_z'], color = plot_color['gt'], linewidth=line_width, label='ground truth')
ax.plot(dr_data['p_x'], dr_data['p_y'], dr_data['p_z'], color = plot_color['vo'], linewidth=line_width, label='deadreckoning')
ax.plot(est_opt_data['p_x'], est_opt_data['p_y'], est_opt_data['p_z'], color = plot_color['opt'], linewidth=line_width, label='opt.')
#ax.plot(est_em_data['p_x'], est_em_data['p_y'], est_em_data['p_z'], color = plot_color['em'], linewidth=line_width, label='EM')
#ax.plot(est_boem_data['p_x'], est_boem_data['p_y'], est_boem_data['p_z'], color = plot_color['boem'], linewidth=line_width, label='BOEM')




ax.view_init(39, 3)


# trajectory only
'''
ax.set_xlim(-9.5,0.5)
ax.set_ylim(-0.5,9.5)
ax.set_zlim(-1.7,0.9)
'''

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

ax.legend()

#plt.savefig("result/" + dataset + "/trajectory.pdf")

plt.show()






# error plot
'''
fig = plt.figure(2)
fig.set_size_inches(fig_width, fig_height)

line_width = 1.5


vo_error = np.zeros_like(gt_data['p_x']);
est_opt_error = np.zeros_like(gt_data['p_x']);
est_em_error = np.zeros_like(gt_data['p_x']);
est_boem_error = np.zeros_like(gt_data['p_x']);

for i in range(len(gt_data['p_x'])):
	vo_error[i] = math.sqrt( (gt_data['p_x'][i]-vo_data['p_x'][i])**2 + (gt_data['p_y'][i]-vo_data['p_y'][i])**2 + (gt_data['p_z'][i]-vo_data['p_z'][i])**2)
	est_opt_error[i]  = math.sqrt( (gt_data['p_x'][i]-est_opt_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_opt_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_opt_data['p_z'][i])**2)
	est_em_error[i]   = math.sqrt( (gt_data['p_x'][i]-est_em_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_em_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_em_data['p_z'][i])**2)
	est_boem_error[i] = math.sqrt( (gt_data['p_x'][i]-est_boem_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_boem_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_boem_data['p_z'][i])**2)


plt.plot(vo_data['timestamp'], vo_error, color = plot_color['vo'], linewidth=line_width, label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_error, color = plot_color['opt'], linewidth=line_width, label='opt.')
plt.plot(est_em_data['timestamp'], est_em_error, color = plot_color['em'], linewidth=line_width, label='EM')
plt.plot(est_boem_data['timestamp'], est_boem_error, color = plot_color['boem'], linewidth=line_width, label='BOEM')


plt.legend()

plt.xlabel('time [s]')
plt.ylabel('error [m]')
plt.ylim([0,1.31])

plt.savefig("result/" + dataset + "/error.pdf")

plt.show()

print("duration:\t" + str(gt_data['timestamp'].iat[-1] - gt_data['timestamp'][0]))

print("VO:\t" + str(np.mean(vo_error)))
print("opt.:\t" + str(np.mean(est_opt_error)))
print("EM:\t" + str(np.mean(est_em_error)))
print("BOEM:\t" + str(np.mean(est_boem_error)))
'''