import os
import numpy as np
import eval_metrics as em
import matplotlib.pyplot as plt
import argparse

def init():
    parser = argparse.ArgumentParser("eval model scores")
    parser.add_argument("-t", "--task", type=str, help="which dataset you would liek to score on",
                        required=False, default='codecfake', choices=["19eval","ITW","codecfake"])
    args = parser.parse_args()
    return args


def compute_eer_and_tdcf(cm_score_file,task):

    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_sources = cm_data[:, 0]
    cm_scores = cm_data[:, 1].astype(np.float)

    other_cm_scores = -cm_scores

    if task == '19eval' or task == 'ITW':
        cm_keys = cm_data[:, 2]
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_keys == 'spoof']
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_keys == 'spoof'])[0]
        print(cm_score_file)
        print('   EER            = {:7.3f} % (Equal error rate for countermeasure)'.format(min(eer_cm, other_eer_cm) * 100))

    if task == 'codecfake':
        cm_keys = cm_data[:, -1]
        bona_cm = cm_scores[cm_keys == 'real']
        spoof_cm = cm_scores[cm_keys == 'fake']
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'fake'], other_cm_scores[cm_keys == 'real'])[0]
        print(cm_score_file)
        print('   EER            = {:7.3f} % (Equal error rate for countermeasure)'.format(min(eer_cm, other_eer_cm) * 100))



    return min(eer_cm, other_eer_cm)


if __name__ == "__main__":
    args = init()
    if args.task == '19eval':
        cm_score_file = './result/19LA_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
    if args.task == 'ITW':
        cm_score_file = './result/ITW_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
    if args.task == 'codecfake':
        cm_score_file = './result/C1_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/C2_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/C3_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/C4_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/C5_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/C6_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/C7_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/L1_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/L2_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)
        cm_score_file = './result/L3_result.txt'
        compute_eer_and_tdcf(cm_score_file,args.task)