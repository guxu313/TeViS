import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List
import scipy.stats as stats
from numpy import *


def sortArrayByCollationArray(originalArray: List[int], collationArray: List[List[int]]) -> List[int]:
    result = []
    temporary = [sorted(x)[::-1] for x in collationArray]
    for oa in originalArray:
        flag = True
        for i, ca in enumerate(collationArray):
            if oa in ca:
                result.append(temporary[i][-1])
                temporary[i].pop()
                flag = False
                break
        if flag:
            result.append(oa)
    return result

# human annotation about similarity & talk
sim_pairs_path = 'data/human_sim_gt.txt'
talk_gt_path = 'data/human_talk_gt.txt'

try:
    model_test_txt_path = sys.argv[1]
    with open(model_test_txt_path, 'r') as test:
        pass
except:
    print('Error, please check argv.')
    exit(-1)

try:
    pred_list=''
    with open(model_test_txt_path, 'r') as model_test_f:
        for line in model_test_f:
            pred_list+=line
    pred_list = eval(pred_list)
except:
    pred_list=[]
    with open(model_test_txt_path, 'r') as f02:
        for line02 in f02:
            pred_list.append(list(eval(line02.strip('\n'))))

sim_pairs_list=[]
with open(sim_pairs_path,'r') as sim_gt_f:
    for l1 in sim_gt_f:
        sim_pairs_list.append(eval(l1.strip('\n')))

talk_gt_list=[]
with open(talk_gt_path,'r') as talk_gt_f:
    for l2 in talk_gt_f:
        talk_gt_list.append(eval(l2.strip('\n')))

kdlls_d_t={"all":[],"2":[],"3_5":[],"6_max":[]}

for i,p in enumerate(pred_list):
    t = list(range(1, len(p)+1))
    if sim_pairs_list[i] != []:
        p_d = sortArrayByCollationArray(p, sim_pairs_list[i])
    else:
        p_d = p

    if talk_gt_list[i] == []:
        kdll_d_t, _ = stats.kendalltau(t, p_d)
    else:
        kdll_d_t, _ = stats.kendalltau(list(range(1, len(p)+1)), p_d)
        for j in talk_gt_list[i]:
            kdll_d_t_tmp, _ = stats.kendalltau(j, p_d)
            kdll_d_t = max(kdll_d_t_tmp, kdll_d_t)
    
    kdlls_d_t["all"].append(kdll_d_t)
    if len(p_d) >= 3 and len(p_d) <= 5:
        kdlls_d_t["3_5"].append(kdll_d_t)
    elif len(p_d) >= 6 :
        kdlls_d_t["6_max"].append(kdll_d_t)

print('#\tkdll tau')
print('all\t',format(mean(kdlls_d_t['all']), '.3f'))
print('3_5\t',format(mean(kdlls_d_t['3_5']), '.3f'))
print('6_max\t',format(mean(kdlls_d_t['6_max']), '.3f'))


