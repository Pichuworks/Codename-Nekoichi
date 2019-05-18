'''
Fuck RNN
'''

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

list_src_pair = list()
list_dist_pair = list()

all_real_pair = 0
all_pred_pair = 0
all_err_pair = 0    # -1 in predict data
list_real_pair = list()
list_pred_pair = list()
list_err_pair = list()

for i, sentence in enumerate(list_src):
    stack = []
    tmp_src_pair = [0 for i in range(len(sentence))]
    sent_pair = 0
    for j, char in enumerate(sentence):
        if char == '(':
            stack.append(['(', j])
        elif char == ')':
            stack_pop = stack.pop()
            tmp_src_pair[j] = stack_pop[1] + 1
            tmp_src_pair[stack_pop[1]] = j + 1
            # pair
            all_real_pair = all_real_pair + 1
            sent_pair = sent_pair + 1

    if len(stack) != 0:
        for i in range(len(stack)):
            stack_pop = stack.pop()
            tmp_src_pair[j] = -1
    list_src_pair.append(tmp_src_pair)
    list_real_pair.append(sent_pair)


for i, sentence in enumerate(list_dist):
    stack = []
    tmp_dist_pair = [0 for i in range(len(sentence))]
    sent_pair = 0
    err_pair = 0
    for j, char in enumerate(sentence):
        if char == '(':
            stack.append(['(', j])
        elif char == ')':
            if len(stack) != 0:
                stack_pop = stack.pop()
                tmp_dist_pair[j] = stack_pop[1] + 1
                tmp_dist_pair[stack_pop[1]] = j + 1
                # pair
                all_pred_pair = all_pred_pair + 1
                sent_pair = sent_pair + 1
            else:
                stack.append([')', j])
    if len(stack) != 0:
        for i in range(len(stack)):
            stack_pop = stack.pop()
            tmp_dist_pair[j] = -1
            all_err_pair = all_err_pair + 1
            err_pair = err_pair + 1
    list_dist_pair.append(tmp_dist_pair)
    list_pred_pair.append(sent_pair)
    list_err_pair.append(err_pair)

print('[OK] Stack processing...')
print(' ')

all_TP = 0
all_FN = 0
all_FP = 0 

avg_sen = 0         # sen = tp/(tp+fn)
avg_spec = 0        # spec = tp/(tp+fp)

cmp_len = len(list_dist_pair)

with open('compare_log.txt', 'w') as f:
    for i in range(cmp_len):
        print('compare epoch ' + str(i))
        f.write('compare epoch ' + str(i) + '\n')
        print('=====================')
        f.write('=====================' + '\n')

        line_TP = 0
        line_FN = 0
        line_FP = 0
        line_sen = 0
        line_spec = 0

        line_flag = [0 for j in range(min(len(list_src_pair[i]), len(list_dist_pair[i])))]
        for j in range(min(len(list_src_pair[i]), len(list_dist_pair[i]))):
            # TP
            if (list_src_pair[i][j] == list_dist_pair[i][j] and line_flag[j] == 0 and line_flag[list_src_pair[i][j]-1] == 0):
                all_TP = all_TP + 1
                line_TP = line_TP + 1
                line_flag[j] = 1
                line_flag[list_src_pair[i][j]-1] = 1
            # FN
            elif (list_src_pair[i][j] > 0  and list_dist_pair[i][j] == 0):
                all_FN = all_FN + 1
                line_FN = line_FN + 1
            # FP
            elif (list_src_pair[i][j] == 0  and list_dist_pair[i][j] > 0):
                all_FP = all_FP + 1
                line_FP = line_FP + 1

        line_sen = line_TP / (line_TP + line_FN)
        line_spec = line_TP / (line_TP + line_FP)

        avg_sen = avg_sen + line_sen
        avg_spec = avg_spec + line_spec

        print('line actual pairs: ' + str(list_real_pair[i]))
        f.write('line actual pairs: ' + str(list_real_pair[i]))
        f.write('\n')

        print('line predict pairs: ' + str(list_pred_pair[i]))
        f.write('line predict pairs: ' + str(list_pred_pair[i]))
        f.write('\n')

        print('line error pairs: ' + str(list_err_pair[i]))
        f.write('line error pairs: ' + str(list_err_pair[i]))
        f.write('\n')

        print('line true positive: ' + str(line_TP))
        f.write('line true positive: ' + str(line_TP))
        f.write('\n')

        print('line false negative: ' + str(line_FN))
        f.write('line false negative: ' + str(line_FN))
        f.write('\n')

        print('line false positive: ' + str(line_FP))
        f.write('line false positive: ' + str(line_FP))
        f.write('\n')

        print('line sensitivity: ' + str(line_sen))
        f.write('line sensitivity: ' + str(line_sen))
        f.write('\n')

        print('line specificity: ' + str(line_spec))
        f.write('line specificity: ' + str(line_spec))
        f.write('\n')

        print(' ')
        f.write('\n')

    all_sen = all_TP / (all_TP + all_FN)
    all_spec = all_TP / (all_TP + all_FP)
    avg_sen = avg_sen / cmp_len
    avg_spec = avg_spec / cmp_len

    print('compare result')
    f.write('compare result')
    f.write('\n')

    print('=====================')
    f.write('=====================')
    f.write('\n')
        
    print('all actual pairs: ' + str(all_real_pair))
    f.write('all actual pairs: ' + str(all_real_pair))
    f.write('\n')

    print('all predict pairs: ' + str(all_pred_pair))
    f.write('all predict pairs: ' + str(all_pred_pair))
    f.write('\n')

    print('all error pairs: ' + str(all_err_pair))
    f.write('all error pairs: ' + str(all_err_pair))
    f.write('\n')

    print('all true positive: ' + str(all_TP))
    f.write('all true positive: ' + str(all_TP))
    f.write('\n')

    print('all false negative: ' + str(all_FN))
    f.write('all false negative: ' + str(all_FN))
    f.write('\n')

    print('all false positive: ' + str(all_FP))
    f.write('all false positive: ' + str(all_FP))
    f.write('\n')

    print('all sensitivity: ' + str(all_sen))
    f.write('all sensitivity: ' + str(all_sen))
    f.write('\n')

    print('all specificity: ' + str(all_spec))
    f.write('all specificity: ' + str(all_spec))
    f.write('\n')

    print('average sensitivity: ' + str(avg_sen))
    f.write('average sensitivity: ' + str(avg_sen))
    f.write('\n')

    print('average specificity: ' + str(avg_sen))
    f.write('average specificity: ' + str(avg_sen))