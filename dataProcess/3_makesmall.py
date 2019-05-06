'''
fuck rna
fuck deep learning
process the data
'''
# -*- coding: utf-8 -*-

with open('fucksta.txt') as f:
    lines = f.readlines()

with open('sequence_new_150_np.txt', 'w') as f:
    for i, line in enumerate(lines):
        if i%4 == 1:
            if(len(line.split(' ')) <= 150 and ('[' not in lines[i+1])):
                f.write(line)      

with open('structure_new_150_np.txt', 'w') as f:
    for i, line in enumerate(lines):
        if i%4 == 2:
            if(len(line.split(' ')) <= 150 and ('[' not in line)):
                f.write(line)