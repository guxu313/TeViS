import sys
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

if __name__ == '__main__':
    try:
        recall_number = sys.argv[1]
        if recall_number not in ['20', '30']:
            exit(-1)
    except:
        print("Error.")
        exit(-1)

    test_types = ['clip','clip_s','clip_g','our']
    for test_type in test_types:
        sim_pairs_path = 'res/human_sim_gt_{}.txt'.format(recall_number, recall_number)
        talk_gt_path = 'res/human_talk_gt_{}.txt'.format(recall_number, recall_number)

        model_test_txt_path = 'res/{}_{}.txt'.format(recall_number, test_type)

        pred_list=''
        with open(model_test_txt_path,'r') as model_test_f:
            for line in model_test_f:
                pred_list+=line
        pred_list = eval(pred_list)

        sim_pairs_list=[]
        with open(sim_pairs_path,'r') as sim_gt_f:
            for l1 in sim_gt_f:
                sim_pairs_list.append(eval(l1.strip('\n')))

        talk_gt_list=[]
        with open(talk_gt_path,'r') as talk_gt_f:
            for l2 in talk_gt_f:
                talk_gt_list.append(eval(l2.strip('\n')))

        kdlls_d_t={"all":[]}

        for i,p in enumerate(pred_list):
            t = list(range(1, len(p)+1))
            
            if sim_pairs_list[i] != []:
                p_d = sortArrayByCollationArray(p, sim_pairs_list[i])
            else:
                p_d = p
            
            if talk_gt_list[i] == []:
                kdll_d_t, _ = stats.kendalltau(t, p_d)
            elif talk_gt_list[i] != []:
                kdll_d_t, _ = stats.kendalltau(list(range(1, len(p)+1)), p_d)
                for j in talk_gt_list[i]:
                    kdll_d_t_tmp, _ = stats.kendalltau(j, p_d)
                    kdll_d_t = max(kdll_d_t_tmp, kdll_d_t)

            kdlls_d_t["all"].append(kdll_d_t)

        print(test_type)
        print('kdll_n:\t kdll')
        print('all:\t', format(mean(kdlls_d_t['all']), '.3f'))
