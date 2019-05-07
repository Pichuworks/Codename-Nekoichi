# -*- coding: utf-8 -*-

str_src = list()
str_dist = list()

with open('test_ans.txt') as f:
    for line in f.readlines():
        line = line.strip() 
        str_src.append(str(line))

with open('test_res.txt') as f:
    for line in f.readlines():
        line = line.strip() 
        str_dist.append(str(line))

list_src = list()
list_dist = list()

for word in str_src:
    list_src.append(word.split(' '))

for word in str_dist:
    list_dist.append(word.split(' '))

all_count = 0
all_count_crrt = 0
all_only_crrt = 0
average_acc = 0
cmp_len = len(str_dist)

with open('compare_log.txt', 'w') as f:
    for i in range(cmp_len):
        print('compare epoch ' + str(i))
        f.write('compare epoch ' + str(i) + '\n')
        print('=====================')
        f.write('=====================' + '\n')
        line_count = 0
        line_count_crrt = 0
        line_only_crrt = 0
        for j in range(min(len(list_dist[i]), len(list_src[i]))):
            all_count = all_count + 1
            line_count = line_count + 1
            if list_src[i][j] == list_dist[i][j]:
                all_count_crrt = all_count_crrt + 1
                all_only_crrt = all_count_crrt + 1
                line_count_crrt = line_count_crrt + 1
                line_only_crrt = line_only_crrt + 1
            elif (list_src[i][j] != '.' and list_dist[i][j] != '.'):
                all_only_crrt = all_count_crrt + 1
                line_only_crrt = line_only_crrt + 1
        line_acc = line_count_crrt / line_count
        line_only_acc = line_only_crrt / line_count
        average_acc = average_acc + line_only_acc

        print('line count: ' + str(line_count))
        f.write('line count: ' + str(line_count))
        f.write('\n')
        print('corrent line count: ' + str(line_count_crrt))
        f.write('corrent line count: ' + str(line_count_crrt))
        f.write('\n')
        print('only acc = ' + str(line_only_acc))
        f.write('only acc = ' + str(line_only_acc))
        f.write('\n')
        print('line acc = ' + str(line_acc))
        f.write('line acc = ' + str(line_acc))
        f.write('\n')
        print(' ')
        f.write('\n')

    all_acc = all_count_crrt / all_count
    all_only_acc = all_only_crrt / all_count
    average_acc = average_acc / cmp_len
    print('compare result')
    f.write('compare result')
    f.write('\n')
    print('=====================')
    f.write('=====================')
    f.write('\n')
    print('all count: ' + str(all_count))
    f.write('all count: ' + str(all_count))
    f.write('\n')
    print('corrent all count: ' + str(all_count_crrt))
    f.write('corrent all count: ' + str(all_count_crrt))
    f.write('\n')
    print('all only acc = ' + str(all_only_acc))
    f.write('all only acc = ' + str(all_only_acc))
    f.write('\n')
    print('all acc = ' + str(all_acc))
    f.write('all acc = ' + str(all_acc))
    f.write('\n')
    print('average acc = ' + str(average_acc))
    f.write('average acc = ' + str(average_acc))