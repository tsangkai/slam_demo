import numpy as np


# error calculation
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


# plotting


color = {
	'gt': [0, 0, 0],
	'dr': [0.0000, 0.5490, 0.3765],
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],
	'lmk': [0.1, 0.1490, 0.3765],
}


def add_boarder(lim):
	coeff = 0.02
	diff = lim[1] - lim[0]

	return (lim[0]-coeff*diff, lim[1]+coeff*diff)