import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np

pl.xlabel('File size')
pl.ylabel('Time [ms]')
pl.ylim(0, 1000)
pl.grid(True)

# i5, xeon, phi, ver2, simple, ver2_simple, load
EXPERIMENT = 'load'


if EXPERIMENT == 'i5':
    d = pd.read_csv("i5.txt")

    pl.title('Character Count on i5 (4 threads / 26 threads)')
    l1 = mpatches.Patch(color='red', label='Simple')
    l2 = mpatches.Patch(color='green', label='Ver2')
    pl.legend(handles=[l1, l2], bbox_to_anchor=(0.5, 1))
    pl.ylim(-100, 3500)
    pl.xlim(0, 5)

    pl.plot(d['size'], d['simple'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['ver2'], marker='o', color='green', lw=2)

    pl.xticks([1, 2, 3, 4], ['1KB', '1MB', '100MB', '1GB'])

    pl.show()

elif EXPERIMENT == 'xeon':
    d = pd.read_csv("xeon.txt")

    pl.title('Character Count on Intel Xeon (80 threads / 78 threads)')
    l1 = mpatches.Patch(color='red', label='Simple')
    l2 = mpatches.Patch(color='green', label='Ver2')
    pl.legend(handles=[l1, l2], bbox_to_anchor=(0.5, 1))
    pl.ylim(-100, 600)
    pl.xlim(0, 5)

    pl.plot(d['size'], d['simple'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['ver2'], marker='o', color='green', lw=2)

    pl.xticks([1, 2, 3, 4], ['1KB', '1MB', '100MB', '1GB'])

    pl.show()

elif EXPERIMENT == 'phi':
    d = pd.read_csv("phi.txt")

    pl.title('Character Count on Intel Xeon Phi (240 threads / 234 threads)')
    l1 = mpatches.Patch(color='red', label='Simple')
    l2 = mpatches.Patch(color='green', label='Ver2')
    pl.legend(handles=[l1, l2], bbox_to_anchor=(0.5, 1))
    pl.ylim(-100, 600)
    pl.xlim(0, 5)

    pl.plot(d['size'], d['simple'], marker='o', color='red', lw=2)
    pl.plot(d['size'], d['ver2'], marker='o', color='green', lw=2)

    pl.xticks([1, 2, 3, 4], ['1KB', '1MB', '100MB', '1GB'])

    pl.show()

elif EXPERIMENT == 'ver2':
    i5 = pd.read_csv("i5.txt")
    xeon = pd.read_csv("xeon.txt")
    phi = pd.read_csv("phi.txt")

    pl.title('Character Count using Ver2')
    l1 = mpatches.Patch(color='red', label='i5')
    l2 = mpatches.Patch(color='yellow', label='Xeon')
    l3 = mpatches.Patch(color='purple', label='Xeon Phi')
    pl.legend(handles=[l1, l2, l3], prop={'size':12}, bbox_to_anchor=(0.5, 1))
    pl.ylim(-100, 3500)
    pl.xlim(0.5, 4.5)

    pl.xticks([1, 2, 3, 4], ['1KB', '1MB', '100MB', '1GB'])

    pl.plot(i5['size'], i5['ver2'], marker='o', color='red', lw=2)
    pl.plot(xeon['size'], xeon['ver2'], marker='o', color='yellow', lw=2)
    pl.plot(phi['size'], phi['ver2'], marker='o', color='purple', lw=2)

    pl.show()

elif EXPERIMENT == 'simple':
    i5 = pd.read_csv("i5.txt")
    xeon = pd.read_csv("xeon.txt")
    phi = pd.read_csv("phi.txt")

    pl.title('Character Count using naive algorithm')
    l1 = mpatches.Patch(color='red', label='i5')
    l2 = mpatches.Patch(color='yellow', label='Xeon')
    l3 = mpatches.Patch(color='purple', label='Xeon Phi')
    pl.legend(handles=[l1, l2, l3], prop={'size':12}, bbox_to_anchor=(0.5, 1))
    pl.ylim(-100, 500)
    pl.xlim(0.5, 4.5)

    pl.xticks([1, 2, 3, 4], ['1KB', '1MB', '100MB', '1GB'])

    pl.plot(i5['size'], i5['simple'], marker='o', color='red', lw=2)
    pl.plot(xeon['size'], xeon['simple'], marker='o', color='yellow', lw=2)
    pl.plot(phi['size'], phi['simple'], marker='o', color='purple', lw=2)

    pl.show()

elif EXPERIMENT == 'ver2_simple':
    i5 = pd.read_csv("i5.txt")
    xeon = pd.read_csv("xeon.txt")
    phi = pd.read_csv("phi.txt")

    pl.title('Character Count (naive / ver2)')
    l1 = mpatches.Patch(color='red', label='Naive (i5)')
    l2 = mpatches.Patch(color='yellow', label='Naive (Xeon)')
    l3 = mpatches.Patch(color='purple', label='Naive (Xeon Phi)')
    l4 = mpatches.Patch(color='green', label='Ver2 (i5)')
    l5 = mpatches.Patch(color='pink', label='Ver2 (Xeon)')
    l6 = mpatches.Patch(color='blue', label='Ver2 (Xeon Phi)')
    pl.legend(handles=[l1, l2, l3, l4, l5, l6], prop={'size':12}, bbox_to_anchor=(0.5, 1))
    pl.ylim(-100, 500)
    pl.xlim(0.5, 4.5)

    pl.xticks([1, 2, 3, 4], ['1KB', '1MB', '100MB', '1GB'])

    pl.plot(i5['size'], i5['simple'], marker='o', color='red', lw=2)
    pl.plot(xeon['size'], xeon['simple'], marker='o', color='yellow', lw=2)
    pl.plot(phi['size'], phi['simple'], marker='o', color='purple', lw=2)

    pl.plot(i5['size'], i5['ver2'], marker='o', color='green', lw=2)
    pl.plot(xeon['size'], xeon['ver2'], marker='o', color='pink', lw=2)
    pl.plot(phi['size'], phi['ver2'], marker='o', color='blue', lw=2)

    pl.show()

elif EXPERIMENT == 'load':
    i5 = pd.read_csv("i5.txt")
    xeon = pd.read_csv("xeon.txt")
    phi = pd.read_csv("phi.txt")

    pl.title('Loading text files')
    l1 = mpatches.Patch(color='green', label='i5')
    l2 = mpatches.Patch(color='yellow', label='Xeon')
    l3 = mpatches.Patch(color='purple', label='Xeon Phi')
    pl.legend(handles=[l1, l2, l3], prop={'size':12}, bbox_to_anchor=(0.5, 1))
    pl.ylim(-100, 3300)
    pl.xlim(0.5, 4.5)

    pl.xticks([1, 2, 3, 4], ['1KB', '1MB', '100MB', '1GB'])

    pl.plot(i5['size'], i5['load'], marker='o', color='green', lw=2)
    pl.plot(xeon['size'], xeon['load'], marker='o', color='yellow', lw=2)
    pl.plot(phi['size'], phi['load'], marker='o', color='purple', lw=2)

    pl.show()

