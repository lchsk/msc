import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np

pl.xlabel('Matrix size')
pl.ylabel('GFLOPS')
pl.ylim(0, 1000)
pl.grid(True)
# size,cuda,cublas,xIKJ_VECT_2D,xIKJ_VECT_2D_TILED,xTEST,xMKL,pIKJ_VECT_2D,pIKJ_VECT_2D_TILED,pTEST,pMKL
# all, my, my2
EXPERIMENT = 'my2'


if EXPERIMENT == 'all':
    d = pd.read_csv("data.csv")

    pl.title('MM (double precision) on Xeon E5 2680, Xeon Phi 7120P, Tesla M2070\nSizes are powers of 2')
    l1 = mpatches.Patch(color='red', label='CUDA')
    l2 = mpatches.Patch(color='green', label='CUBLAS')

    l3 = mpatches.Patch(color='black', label='IKJ_VECT_2D (Xeon)')
    # l4 = mpatches.Patch(color='blue', label='IKJ_VECT_2D_TILED (Xeon)')
    l5 = mpatches.Patch(color='yellow', label='TEST (Xeon)')
    l6 = mpatches.Patch(color='pink', label='MKL (Xeon)')

    l7 = mpatches.Patch(color='black', label='IKJ_VECT_2D (Phi)', linestyle='dashed', fill=False, lw=2)
    # l8 = mpatches.Patch(color='blue', label='IKJ_VECT_2D_TILED (Phi)', linestyle='dashed', lw=2)
    l9 = mpatches.Patch(color='yellow', label='TEST (Phi)', linestyle='dashed', fill=False, lw=2)
    l10 = mpatches.Patch(color='pink', label='MKL (Phi)', linestyle='dashed', fill=False, lw=2)

    pl.legend(handles=[l1, l2, l3, l5, l6, l7, l9, l10], prop={'size':10}, bbox_to_anchor=(0.3, 0.8))

    # Example data
    # a = np.arange(0,3, .02)
    # b = np.arange(0,3, .02)
    # c = np.exp(a)
    # fig, ax = pl.subplots()
    # ax.plot(d['size'], d['cuda'], 'k--', label='Model length')
    # ax.plot(a, c, 'k:', label='Data length')
    # ax.plot(a, c, 'k', label='Total message length')

    # legend = ax.legend(loc='upper center', shadow=True)

    pl.ylim(-5, 340)


    pl.plot(d['size'], d['cuda'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['cublas'], marker='o', color='green', lw=2)

    pl.plot(d['size'], d['xIKJ_VECT_2D'], marker='o', color='black', lw=2)
    # pl.plot(d['size'], d['xIKJ_VECT_2D_TILED'], marker='o', color='blue', lw=2)
    pl.plot(d['size'], d['xTEST'], marker='o', color='yellow', lw=2)
    pl.plot(d['size'], d['xMKL'], marker='o', color='pink', lw=2)

    pl.plot(d['size'], d['pIKJ_VECT_2D'], marker='o', color='black', lw=3, linestyle='--')
    # pl.plot(d['size'], d['pIKJ_VECT_2D_TILED'], marker='o', color='blue', lw=3, linestyle='--')
    pl.plot(d['size'], d['pTEST'], marker='o', color='yellow', lw=3, linestyle='--')
    pl.plot(d['size'], d['pMKL'], marker='o', color='pink', lw=3, linestyle='--')

    pl.show()


elif EXPERIMENT == 'my':
    d = pd.read_csv("data.csv")

    pl.title('MM (double precision) on Xeon E5 2680, Xeon Phi 7120P, Tesla M2070\nSizes are powers of 2')
    l3 = mpatches.Patch(color='red', label='IKJ_VECT_2D (Xeon)')
    # l4 = mpatches.Patch(color='blue', label='IKJ_VECT_2D_TILED (Xeon)')
    l5 = mpatches.Patch(color='purple', label='TEST (Xeon)')
    l6 = mpatches.Patch(color='green', label='MKL (Xeon)')

    l7 = mpatches.Patch(color='red', label='IKJ_VECT_2D (Phi)', linestyle='dashed', fill=False, lw=2)
    # l8 = mpatches.Patch(color='blue', label='IKJ_VECT_2D_TILED (Phi)', linestyle='dashed', lw=2)
    l9 = mpatches.Patch(color='purple', label='TEST (Phi)', linestyle='dashed', fill=False, lw=2)
    l10 = mpatches.Patch(color='green', label='MKL (Phi)', linestyle='dashed', fill=False, lw=2)

    pl.legend(handles=[l3, l5, l6, l7, l9, l10], prop={'size':12}, loc=0)

    pl.ylim(-5, 340)


    pl.plot(d['size'], d['xIKJ_VECT_2D'], marker='o', color='red', lw=2)
    # pl.plot(d['size'], d['xIKJ_VECT_2D_TILED'], marker='o', color='blue', lw=2)
    pl.plot(d['size'], d['xTEST'], marker='o', color='purple', lw=2)
    pl.plot(d['size'], d['xMKL'], marker='o', color='green', lw=2)

    pl.plot(d['size'], d['pIKJ_VECT_2D'], marker='o', color='red', lw=3, linestyle='--')
    # pl.plot(d['size'], d['pIKJ_VECT_2D_TILED'], marker='o', color='blue', lw=3, linestyle='--')
    pl.plot(d['size'], d['pTEST'], marker='o', color='purple', lw=3, linestyle='--')
    pl.plot(d['size'], d['pMKL'], marker='o', color='green', lw=3, linestyle='--')

    pl.show()

elif EXPERIMENT == 'my2':
    d = pd.read_csv("data.csv")

    pl.title('MM (double precision) on Xeon E5 2680, Xeon Phi 7120P, Tesla M2070\nSizes are powers of 2')
    l3 = mpatches.Patch(color='red', label='IKJ_VECT_2D (Xeon)')
    l5 = mpatches.Patch(color='purple', label='TEST (Xeon)')

    l7 = mpatches.Patch(color='red', label='IKJ_VECT_2D (Phi)', linestyle='dashed', fill=False, lw=2)
    l9 = mpatches.Patch(color='purple', label='TEST (Phi)', linestyle='dashed', fill=False, lw=2)

    pl.legend(handles=[l3, l5, l7, l9], prop={'size':12}, loc=0)

    pl.ylim(-5, 100)


    pl.plot(d['size'], d['xIKJ_VECT_2D'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['xTEST'], marker='o', color='purple', lw=2)

    pl.plot(d['size'], d['pIKJ_VECT_2D'], marker='o', color='red', lw=3, linestyle='--')
    pl.plot(d['size'], d['pTEST'], marker='o', color='purple', lw=3, linestyle='--')

    pl.show()