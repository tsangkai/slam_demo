import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

plot_color = {
	'gt': [0, 0, 0],
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],  # navy
	'vo': [0.0000, 0.5490, 0.3765], # spruce
}


fig_width = 11.326
fig_height = 7.0

# keyframe has larger time range than ground truth

dataset = sys.argv[1]

gt_data = pd.read_csv("data/" + dataset + "/gt.csv")
vo_data = pd.read_csv("data/" + dataset + "/traj_vo.csv")
#est_opt_data = pd.read_csv("data/" + dataset + "/traj_opt.csv")

# landmark_data = pd.read_csv("landmark.csv")


# offset initial time
init_time = gt_data['timestamp'][0]
gt_data['timestamp'] = gt_data['timestamp'] - init_time
vo_data['timestamp'] = vo_data['timestamp'] - init_time





fig = plt.figure(1)
fig.set_size_inches(fig_width, fig_height)

ax = fig.gca(projection='3d')


# ax.scatter(landmark_data['p_x'], landmark_data['p_y'], landmark_data['p_z'])

ax.plot(gt_data['p_x'], gt_data['p_y'], gt_data['p_z'], color = plot_color['gt'], label='gt')
ax.plot(vo_data['p_x'], vo_data['p_y'], vo_data['p_z'], color = plot_color['vo'], label='VO')
#ax.plot(est_opt_data['p_x'], est_opt_data['p_y'], est_opt_data['p_z'], color = plot_color['opt'], label='est. opt.')


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

# plt.savefig('result/result_2.pdf')  

plt.show()

