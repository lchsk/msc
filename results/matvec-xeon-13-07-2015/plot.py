import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np

pl.xlabel('Size')
pl.ylabel('GFLOPS')
pl.ylim(0, 1000)
pl.grid(True)

EXPERIMENT = 'all'


if EXPERIMENT == 'all':
    d = pd.read_csv("data.csv")

    pl.title('Matrix x vector (double precision) on Xeon E5 2680, Xeon Phi 7120P')
    l1 = mpatches.Patch(color='blue', label='Xeon')
    l2 = mpatches.Patch(color='green', label='Xeon Phi')
    pl.legend(handles=[l1, l2], prop={'size': 12}, loc=0)

    pl.ylim(0, 40)


    pl.plot(d['size'], d['xeon'], marker='o', color='blue', lw=2)
    pl.plot(d['size'], d['phi'], marker='o', color='green', lw=2)

    pl.show()


