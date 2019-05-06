'''
fuck rna
fuck deep learning
process the data
'''
# -*- coding: utf-8 -*-

with open('fucksta.txt') as f:
    lines = f.readlines()

with open('sequence_new.txt', 'w') as f:
    for i, line in enumerate(lines):
        if i%4 == 1:
            f.write(line)      

with open('structure_new.txt', 'w') as f:
    for i, line in enumerate(lines):
        if i%4 == 2:
            f.write(line)