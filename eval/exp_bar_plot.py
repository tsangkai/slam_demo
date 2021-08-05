import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig_width = 5.84
fig_height = 4.00 #4.38

plot_color = {
	'gt': [0, 0, 0],
	'vo': [0.0000, 0.5490, 0.3765], # spruce
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],  # navy
}

def lighten_color(color):
	scale = 0.85
	return [1 - scale*(1 - color[0]), 1 - scale*(1 - color[1]), 1 - scale*(1 - color[2])]



labels = ['MH 01', 'MH 02', 'MH 03', 'MH 04', 'MH 05']


vo   = [0.496, 0.899, 0.313, 0.489, 0.517]
opt  = [0.467, 0.609, 0.292, 0.437, 0.375]
em   = [0.438, 0.807, 0.286, 0.432, 0.495]
boem = [0.452, 0.830, 0.290, 0.430, 0.497]



x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars
scale = 0.95

fig = plt.figure(1)
fig.set_size_inches(fig_width, fig_height)
ax = fig.add_subplot(111)


rects_vo   = ax.bar(x - 1.5*width, vo, scale*width, color = lighten_color(plot_color['vo']), label='VIO')
rects_opt  = ax.bar(x - 0.5*width, opt, scale*width, color = lighten_color(plot_color['opt']), label='opt.')
rects_em   = ax.bar(x + 0.5*width, em, scale*width, color = lighten_color(plot_color['em']), label='EM')
rects_boem = ax.bar(x + 1.5*width, boem, scale*width, color = lighten_color(plot_color['boem']), label='BOEM')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('trajectory error [m]')
# ax.set_ylim([0, 0.905])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

fig.savefig("result/bar_error.pdf")



########################################################




opt_mean =  [205.282174059, 23.280404758, 44.098048405, 89.614287361, 90.325791128]
em_mean =   [93.351977320, 84.340373156, 75.233596509, 31.204921718, 41.469785103]
boem_mean = [31.697512873, 23.845210073, 20.769196960, 10.689120848, 14.613471780]

opt_std =  [205.282174059*0.0794, 23.280404758*0.0466, 44.098048405*0.1195, 89.614287361*0.0056, 90.325791128*0.0098]
em_std =   [ 93.351977320*0.0112, 84.340373156*0.0069, 75.233596509*0.0011, 31.204921718*0.0011, 41.469785103*0.0014]
boem_std = [ 31.697512873*0.0294, 23.845210073*0.0014, 20.769196960*0.0011, 10.689120848*0.002, 14.613471780*0.0011]


x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars
scale = 0.95

fig = plt.figure(2)
fig.set_size_inches(fig_width, fig_height)
ax = fig.add_subplot(111)

rects_opt  = ax.bar(x - 1*width, opt_mean, scale*width, yerr=opt_std, ecolor=[0.2, 0.2, 0.2], capsize=2, color = lighten_color(plot_color['opt']), label='opt.')
rects_em   = ax.bar(x + 0.0*width, em_mean, scale*width, yerr=em_std, ecolor=[0.2, 0.2, 0.2], capsize=2, color = lighten_color(plot_color['em']), label='EM')
rects_boem = ax.bar(x + 1*width, boem_mean, scale*width, yerr=boem_std, ecolor=[0.2, 0.2, 0.2], capsize=2, color = lighten_color(plot_color['boem']), label='BOEM')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('processing time [sec]')
#ax.set_ylim([0, 0.55])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

fig.savefig("result/bar_time.pdf")

#########################################

