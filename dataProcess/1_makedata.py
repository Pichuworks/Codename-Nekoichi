import numpy as np
import glob
import re


def getsequenceandstructure(filename, headersize):
    data = np.loadtxt(filename, skiprows = headersize, dtype='str')

    sequence = data[0]
    pattern = re.compile('.{1,1}')
    sequence = ' '.join(pattern.findall(sequence))

    structure = data[1]
    structure = ' '.join(pattern.findall(structure))
    return sequence, structure


def writedatafile(paths, outfile, headersize):
    f = open(outfile, 'w')
    for path in paths:
        sequence, structure = getsequenceandstructure(path, headersize)
        f.write(path + '\n')
        f.write(sequence + ' \n')
        f.write(structure + ' \n')
        f.write('\n')

    f.close()
    return

if __name__ == '__main__':
    
    # CHANGE THESE IF YOU'RE USING YOUR OWN DATA
    outfile = 'fucksta.txt'  # output file to write to
    headersize = 2  # number of lines in the .ct file before the sequence begins
    
    
    # get all filepaths
    fuck = '*.sta'
    paths = glob.glob(fuck, recursive = True)
    
    writedatafile(paths, outfile, headersize)
    