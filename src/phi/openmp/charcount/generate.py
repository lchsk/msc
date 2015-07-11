import os, sys, random


def run():

    filename = 'default.txt'
    bytes = 1048576

    alphabet = []

    for i in xrange(0, 26):
        alphabet.append(chr(97 + i))

    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    if len (sys.argv) >= 3:
        bytes = int(sys.argv[2])

    fh = open (filename, 'w')

    for i in xrange(0, bytes):
        fh.write(random.choice(alphabet))

    fh.close()



if __name__ == '__main__':

    run()
