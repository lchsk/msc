import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np

pl.xlabel('Matrix size')
pl.ylabel('GFLOPS')
pl.ylim(0, 1000)
pl.grid(True)

# i5, xeonbest, xeon, phi, my
EXPERIMENT = 'my'


if EXPERIMENT == 'i5':
    d = pd.read_csv("i5.txt")

    pl.title('MM (double precision) on i5 (4 threads)\n(IKJ_VECT_2D, MKL)')
    l1 = mpatches.Patch(color='red', label='Test (no warm up)')
    l2 = mpatches.Patch(color='green', label='Test (warm up)')
    l3 = mpatches.Patch(color='blue', label='MKL (no warm up)')
    l4 = mpatches.Patch(color='black', label='MKL (warm up)')
    pl.legend(handles=[l1, l2, l3, l4], bbox_to_anchor=(0.5, 1))
    pl.ylim(0, 60)

    pl.plot(d['size'], d['test-no-warm-up'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['test-warm-up'], marker='o', color='green', lw=2)
    pl.plot(d['size'], d['mkl-no-warm-up'], marker='o', color='blue', lw=2)
    pl.plot(d['size'], d['mkl-warm-up'], marker='o', color='black', lw=2)

    pl.show()

elif EXPERIMENT == 'xeonbest':
    d = pd.read_csv("xeon.txt")

    pl.title('MM (double precision) on Xeon (40 threads) / Xeon + 3xPhi (40 + 3x244)\n(IKJ_VECT_2D, MKL)')
    l1 = mpatches.Patch(color='red', label='Test (no warm up)')
    l2 = mpatches.Patch(color='green', label='MKL (no warm up, no auto_offload)')
    l3 = mpatches.Patch(color='blue', label='MKL (no warm up, auto_offload)')
    l4 = mpatches.Patch(color='black', label='Test (warm up)')
    l5 = mpatches.Patch(color='yellow', label='MKL (warm up, no auto_offload)')
    l6 = mpatches.Patch(color='pink', label='MKL (warm up, auto_offload)')
    pl.legend(handles=[l1, l2, l3, l4, l5, l6], prop={'size':10}, bbox_to_anchor=(0.5, 1))
    pl.ylim(0, 1000)

    pl.plot(d['size'], d['test-no-warm-up'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['mkl-no-warm-up-no-ao'], marker='o', color='green', lw=2)
    pl.plot(d['size'], d['mkl-no-warm-up-ao'], marker='o', color='blue', lw=2)
    pl.plot(d['size'], d['test-warm-up'], marker='o', color='black', lw=2)
    pl.plot(d['size'], d['mkl-warm-up-no-ao'], marker='o', color='yellow', lw=2)
    pl.plot(d['size'], d['mkl-warm-up-ao'], marker='o', color='pink', lw=2)

    pl.show()

elif EXPERIMENT == 'xeon':
    d = pd.read_csv("xeon.txt")

    pl.title('MM (double precision) on Xeon (40 threads)\n(IKJ_VECT_2D, MKL)')
    l1 = mpatches.Patch(color='red', label='Test (no warm up)')
    l2 = mpatches.Patch(color='green', label='MKL (no warm up, no auto_offload)')
    l3 = mpatches.Patch(color='black', label='Test (warm up)')
    l4 = mpatches.Patch(color='yellow', label='MKL (warm up, no auto_offload)')
    pl.legend(handles=[l1, l2, l3, l4], prop={'size':10}, bbox_to_anchor=(0.5, 1))
    pl.ylim(0, 300)

    pl.plot(d['size'], d['test-no-warm-up'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['mkl-no-warm-up-no-ao'], marker='o', color='green', lw=2)
    pl.plot(d['size'], d['test-warm-up'], marker='o', color='black', lw=2)
    pl.plot(d['size'], d['mkl-warm-up-no-ao'], marker='o', color='yellow', lw=2)

    pl.show()

elif EXPERIMENT == 'phi':
    d = pd.read_csv("phi.txt")

    pl.title('MM (double precision) on Xeon Phi (native, 244 threads)\n(IKJ_VECT_2D, MKL)')
    l1 = mpatches.Patch(color='red', label='Test (no warm up)')
    l2 = mpatches.Patch(color='green', label='Test (warm up)')
    l3 = mpatches.Patch(color='blue', label='MKL (no warm up)')
    l4 = mpatches.Patch(color='black', label='MKL (warm up)')
    pl.legend(handles=[l1, l2, l3, l4], prop={'size':10}, bbox_to_anchor=(0.5, 1))
    pl.ylim(0, 500)

    pl.plot(d['size'], d['test-no-warm-up'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['test-warm-up'], marker='o', color='green', lw=2)
    pl.plot(d['size'], d['mkl-no-warm-up'], marker='o', color='blue', lw=2)
    pl.plot(d['size'], d['mkl-warm-up'], marker='o', color='black', lw=2)

    pl.show()

elif EXPERIMENT == 'my':
    i5 = pd.read_csv("i5.txt")
    xeon = pd.read_csv("xeon.txt")
    phi = pd.read_csv("phi.txt")

    pl.title('MM (double precision)\nIKJ_VECT_2D')
    l1 = mpatches.Patch(color='red', label='i5, cold')
    l2 = mpatches.Patch(color='green', label='i5, warm')
    l3 = mpatches.Patch(color='blue', label='Xeon, cold')
    l4 = mpatches.Patch(color='black', label='Xeon, warm')
    l5 = mpatches.Patch(color='yellow', label='Xeon Phi, cold')
    l6 = mpatches.Patch(color='purple', label='Xeon Phi, warm')
    pl.legend(handles=[l1, l2, l3, l4, l5, l6], prop={'size':12}, bbox_to_anchor=(1, 1))
    pl.ylim(0, 150)

    x = np.arange(0, 10000, 100);

    pl.plot(i5['size'], i5['test-no-warm-up'], marker='o', color='red', lw=2, linestyle='--')
    pl.plot(i5['size'], i5['test-warm-up'], marker='o', color='green', lw=2, linestyle='--')
    pl.plot(xeon['size'], xeon['test-no-warm-up'], marker='o', color='blue', lw=2)
    pl.plot(xeon['size'], xeon['test-warm-up'], marker='o', color='black', lw=2)
    pl.plot(phi['size'], phi['test-no-warm-up'], marker='o', color='yellow', lw=2)
    pl.plot(phi['size'], phi['test-warm-up'], marker='o', color='purple', lw=2)

    pl.show()

# pl.plot(d['size'], d['xeon'], 'r', color='r', lw=2)
# pl.plot(d['size'], d['phi'], 'r', color='r', lw=2)
# pl.plot(d.index, d['xeon'], 'ro', color='r')
# pl.plot(d.index, d['phi'], 'ro', color='r')
# pl.plot(pl0['distP'], pl0['countP'], 'go', color='g')
# pl.plot(pl0['x'], pl0['y'], 'go', markersize=3)
#



# x = np.arange(0, 5, 0.1);
# y = np.sin(x)
# pl.plot(x, y)
# pl.show()

# pl.plot(test['distM'], test['pred'], 'ro',  markersize=5)
# # pl.plot(test.index.values, test['height'], 'ro', color='g')
