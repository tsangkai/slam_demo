import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

gt_data = pd.read_csv("result/sim/gt.csv")
dr_data = pd.read_csv("result/sim/dr.csv")

dr_err_data = pd.read_csv("result/sim/dr_error.csv")



fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
ax1.set_title('rotation')

line_width = 1.2
ax1.plot(gt_data['timestamp'], gt_data['q_w'], color = plot_color['gt'], label='groud truth')
ax1.plot(dr_data['timestamp'], dr_data['q_w'], color = plot_color['dr'], label='dead reckoning')
ax1.set_ylabel('q_w')
ax1.legend(loc = 1)


ax2.plot(gt_data['timestamp'], gt_data['q_x'], color = plot_color['gt'], label='groud truth')
ax2.plot(dr_data['timestamp'], dr_data['q_x'], color = plot_color['dr'], label='dead reckoning')
ax2.set_ylabel('q_x')

ax3.plot(gt_data['timestamp'], gt_data['q_y'], color = plot_color['gt'], label='groud truth')
ax3.plot(dr_data['timestamp'], dr_data['q_y'], color = plot_color['dr'], label='dead reckoning')
ax3.set_ylabel('q_y')

ax4.plot(gt_data['timestamp'], gt_data['q_z'], color = plot_color['gt'], label='groud truth')
ax4.plot(dr_data['timestamp'], dr_data['q_z'], color = plot_color['dr'], label='dead reckoning')
ax4.set_ylabel('q_z')

ax5.plot(dr_data['timestamp'], dr_err_data['d_q'], color = plot_color['dr'], label='dead reckoning')
ax5.set_ylabel('q error')

plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.set_title('velocity')

line_width = 1.2
ax1.plot(gt_data['timestamp'], gt_data['v_x'], color = plot_color['gt'], label='groud truth')
ax1.plot(dr_data['timestamp'], dr_data['v_x'], color = plot_color['dr'], label='dead reckoning')
ax1.set_ylabel('v_x')
ax1.legend(loc = 1)



ax2.plot(gt_data['timestamp'], gt_data['v_y'], color = plot_color['gt'], label='groud truth')
ax2.plot(dr_data['timestamp'], dr_data['v_y'], color = plot_color['dr'], label='dead reckoning')
ax2.set_ylabel('v_y')

ax3.plot(gt_data['timestamp'], gt_data['v_z'], color = plot_color['gt'], label='groud truth')
ax3.plot(dr_data['timestamp'], dr_data['v_z'], color = plot_color['dr'], label='dead reckoning')
ax3.set_ylabel('v_z')


plt.show()


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.set_title('position')

line_width = 1.2
ax1.plot(gt_data['timestamp'], gt_data['p_x'], color = plot_color['gt'], label='groud truth')
ax1.plot(dr_data['timestamp'], dr_data['p_x'], color = plot_color['dr'], label='dead reckoning')
ax1.set_ylabel('p_x')
ax1.legend(loc = 1)



ax2.plot(gt_data['timestamp'], gt_data['p_y'], color = plot_color['gt'], label='groud truth')
ax2.plot(dr_data['timestamp'], dr_data['p_y'], color = plot_color['dr'], label='dead reckoning')
ax2.set_ylabel('p_y')

ax3.plot(gt_data['timestamp'], gt_data['p_z'], color = plot_color['gt'], label='groud truth')
ax3.plot(dr_data['timestamp'], dr_data['p_z'], color = plot_color['dr'], label='dead reckoning')
ax3.set_ylabel('p_z')



ax4.plot(dr_data['timestamp'], dr_err_data['d_p'], color = plot_color['dr'], label='dead reckoning')
ax4.set_ylabel('p error')
ax4.set_ylim(top=2+0.01, bottom=0-0.01)




plt.show()