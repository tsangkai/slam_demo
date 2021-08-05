import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

plot_color = {
	'gt': [0.1, 0.1, 0.1],
	'vo': [0.0000, 0.5490, 0.3765], # spruce
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],  # navy
}


fig_width = 5.84
fig_height = 4.00 #4.38

# keyframe has larger time range than ground truth

dataset = sys.argv[1]

gt_data = pd.read_csv("result/" + dataset + "/traj_gt.csv")
vo_data = pd.read_csv("result/" + dataset + "/traj_vo_x.csv")
est_opt_data = pd.read_csv("result/" + dataset + "/traj_opt_x.csv")
est_em_data = pd.read_csv("result/" + dataset + "/traj_em_x.csv")
est_boem_data = pd.read_csv("result/" + dataset + "/traj_boem_x.csv")



# offset initial time
init_time = gt_data['timestamp'][0]
gt_data['timestamp'] = gt_data['timestamp'] - init_time
vo_data['timestamp'] = vo_data['timestamp'] - init_time
est_opt_data['timestamp'] = est_opt_data['timestamp'] - init_time
est_em_data['timestamp'] = est_em_data['timestamp'] - init_time
est_boem_data['timestamp'] = est_boem_data['timestamp'] - init_time





fig = plt.figure(1)
fig.set_size_inches(fig_width, fig_height)

#ax = fig.gca(projection='3d')
ax = Axes3D(fig, rect=(0.0, 0.05, 1, 0.9))


# ax = fig.add_axes((0.1, 0.1, 0.1, 0.1))


line_width = 1.2

ax.plot(gt_data['p_x'], gt_data['p_y'], gt_data['p_z'], color = plot_color['gt'], linewidth=line_width, label='ground truth')
ax.plot(vo_data['p_x'], vo_data['p_y'], vo_data['p_z'], color = plot_color['vo'], linewidth=line_width, label='VIO')
ax.plot(est_opt_data['p_x'], est_opt_data['p_y'], est_opt_data['p_z'], color = plot_color['opt'], linewidth=line_width, label='opt.')
ax.plot(est_em_data['p_x'], est_em_data['p_y'], est_em_data['p_z'], color = plot_color['em'], linewidth=line_width, label='EM')
ax.plot(est_boem_data['p_x'], est_boem_data['p_y'], est_boem_data['p_z'], color = plot_color['boem'], linewidth=line_width, label='BOEM')




ax.view_init(34, -106)


# trajectory only
'''
ax.set_xlim(-9.5,0.5)
ax.set_ylim(-0.5,9.5)
ax.set_zlim(-1.7,0.9)
'''

print(ax.get_xlim())
print(ax.get_ylim())
print(ax.get_zlim())

x_len = ax.get_xlim()[1] - ax.get_xlim()[0]
y_len = ax.get_ylim()[1] - ax.get_ylim()[0]
z_len = ax.get_zlim()[1] - ax.get_zlim()[0]


ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

ax.legend()


# inset
'''
ax2 = Axes3D(fig, rect=(0.03, 0.03, 0.33, 0.33))

ax2.plot(gt_data['p_x'], gt_data['p_y'], gt_data['p_z'], color = plot_color['gt'], linewidth=line_width, label='ground truth')
ax2.plot(vo_data['p_x'], vo_data['p_y'], vo_data['p_z'], color = plot_color['vo'], linewidth=line_width, label='VIO')
ax2.plot(est_opt_data['p_x'], est_opt_data['p_y'], est_opt_data['p_z'], color = plot_color['opt'], linewidth=line_width, label='opt.')
ax2.plot(est_em_data['p_x'], est_em_data['p_y'], est_em_data['p_z'], color = plot_color['em'], linewidth=line_width, label='EM')
ax2.plot(est_boem_data['p_x'], est_boem_data['p_y'], est_boem_data['p_z'], color = plot_color['boem'], linewidth=line_width, label='BOEM')


ax2.set_xlim(2,2 + 0.25*x_len)
ax2.set_ylim(-2.5,-2.5 + 0.25*y_len)
ax2.set_zlim(-0.2,-0.2+0.25*z_len)

ax2.view_init(34, -106)
'''

plt.savefig("result/" + dataset + "/trajectory.pdf")

plt.show()


def add_boarder(lim):
	coeff = 0.02
	diff = lim[1] - lim[0]
	return (lim[0]-coeff*diff, lim[1]+coeff*diff)



# error plot
fig = plt.figure(2)
fig.set_size_inches(fig_width, fig_height)

line_width = 1.2


vo_error = np.zeros_like(gt_data['p_x'])
est_opt_error = np.zeros_like(gt_data['p_x'])
est_em_error = np.zeros_like(gt_data['p_x'])
est_boem_error = np.zeros_like(gt_data['p_x'])

vo_error_2_sum = 0
est_opt_error_2_sum = 0
est_em_error_2_sum = 0
est_boem_error_2_sum = 0

for i in range(len(gt_data['p_x'])):
	vo_error[i] = math.sqrt( (gt_data['p_x'][i]-vo_data['p_x'][i])**2 + (gt_data['p_y'][i]-vo_data['p_y'][i])**2 + (gt_data['p_z'][i]-vo_data['p_z'][i])**2)
	est_opt_error[i]  = math.sqrt( (gt_data['p_x'][i]-est_opt_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_opt_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_opt_data['p_z'][i])**2)
	est_em_error[i]   = math.sqrt( (gt_data['p_x'][i]-est_em_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_em_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_em_data['p_z'][i])**2)
	est_boem_error[i] = math.sqrt( (gt_data['p_x'][i]-est_boem_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_boem_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_boem_data['p_z'][i])**2)

	vo_error_2_sum += (gt_data['p_x'][i]-vo_data['p_x'][i])**2 + (gt_data['p_y'][i]-vo_data['p_y'][i])**2 + (gt_data['p_z'][i]-vo_data['p_z'][i])**2
	est_opt_error_2_sum += (gt_data['p_x'][i]-est_opt_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_opt_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_opt_data['p_z'][i])**2
	est_em_error_2_sum += (gt_data['p_x'][i]-est_em_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_em_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_em_data['p_z'][i])**2 
	est_boem_error_2_sum += (gt_data['p_x'][i]-est_boem_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_boem_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_boem_data['p_z'][i])**2


plt.plot(vo_data['timestamp'], vo_error, color = plot_color['vo'], linewidth=line_width, label='VIO')
plt.plot(est_opt_data['timestamp'], est_opt_error, color = plot_color['opt'], linewidth=line_width, label='opt.')
plt.plot(est_em_data['timestamp'], est_em_error, color = plot_color['em'], linewidth=line_width, label='EM')
plt.plot(est_boem_data['timestamp'], est_boem_error, color = plot_color['boem'], linewidth=line_width, label='BOEM')


plt.legend(loc='upper right', ncol=2)

plt.xlabel('time [s]')
plt.ylabel('error [m]')
plt.ylim(add_boarder((0,1.45)))

plt.savefig("result/" + dataset + "/error.pdf")

plt.show()

print("\ntime:\t" + str(gt_data['timestamp'].iat[-1] - gt_data['timestamp'][0]))

print("VO:\t" + str(math.sqrt(vo_error_2_sum / len(gt_data['p_x']))))
print("opt.:\t" + str(math.sqrt(est_opt_error_2_sum / len(gt_data['p_x']))))
print("EM:\t" + str(math.sqrt(est_em_error_2_sum / len(gt_data['p_x']))))
print("BOEM:\t" + str(math.sqrt(est_boem_error_2_sum / len(gt_data['p_x']))))