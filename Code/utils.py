import os
import torch
import random
import numpy as np

cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag, acc_taw_b, acc_tag_b, forg_taw_b, forg_tag_b):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg', 'TAw macro', 'TAg macro', 'TAw macro Forg', 'TAg macro Forg'],
                            [acc_taw, acc_tag, forg_taw, forg_tag, acc_taw_b, acc_tag_b, forg_taw_b, forg_tag_b]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:.4f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:.4f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:.4f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)


def get_class_order(mode):
    if mode == 'm2l':
        class_order = [36, 22, 21, 35, 20, 24,  6,  0, 12, 15, 17, 19, 26,  8, 30, 29, 31,
                       16, 33, 10,  5, 27,  1,  2,  3, 14, 25, 18, 13, 11, 32, 34,  9,  4,
                       7, 23, 28]
    elif mode == 'l2m':
        class_order = [28, 23,  7,  4,  9, 32, 13, 11, 34, 18, 25, 14,  3,  2,  1, 33, 10,
                       27,  5, 31, 16, 29, 30,  8, 26, 19, 17, 15, 12,  0,  6, 24, 20, 21,
                       35, 22, 36]
    elif mode == 'random1':
        class_order = [15, 34,  8,  1, 26,  6, 14, 19, 29, 10, 33, 36, 20,  3, 12, 35, 31,
                       28, 23, 30, 32,  0, 21, 25, 13,  4, 11, 22,  7, 18,  5, 27, 17,  9,
                       16, 24,  2]
    elif mode == 'random2':
        class_order = [20, 36, 10, 33, 16, 31,  9, 35, 15, 22, 12, 34, 17, 27, 14, 24,  1,
                       25, 26, 19,  4, 18,  6, 23, 11, 29, 13,  3, 28,  7, 21,  0,  8, 32,
                       2, 30,  5]
    elif mode == 'random3':
        class_order = [29, 31, 27, 15, 19, 21, 18, 20,  4,  3, 26, 23, 14,  2, 34, 10,  6,
                       5,  0, 36, 25, 35,  1, 30, 24, 32, 33, 16, 22,  9,  7,  8, 12, 28,
                       11, 17, 13]
    elif mode == 'no':
        class_order = None

    return class_order



def get_class_order_ps(mode):
    if mode == 'm2l':
        class_order = [0, 1, 2, 3, 5, 4, 6, 7, 9, 11, 10, 8, 13, 12, 14, 16, 15,
                       17, 18, 19, 20, 22, 24, 21, 25, 26, 23, 27, 29, 28, 31, 30, 35, 33,
                       34, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 45, 48, 51, 49,
                       52, 53, 50]
    elif mode == 'l2m':
        class_order = [50, 53, 52, 49, 48, 51, 45, 47, 46, 44, 43, 42, 40, 41, 38, 39, 36,
                       37, 32, 33, 34, 35, 30, 31, 28, 29, 23, 27, 26, 25, 21, 24, 22, 20,
                       19, 18, 17, 15, 16, 14, 12, 13, 8, 10, 11, 7, 9, 6, 4, 5, 3,
                       2, 1, 0]
    elif mode == 'random1':
        class_order = [6, 49, 9, 8, 14, 26, 3, 39, 15, 5, 47, 4, 42, 33, 36, 31, 25, 37,
                        22, 23, 41, 52, 40, 0, 45, 1, 43, 19, 27, 18, 50, 46, 28, 35, 21, 16,
                        38, 29, 32, 11, 2, 51, 44, 10, 12, 7, 53, 34, 13, 24, 20, 30, 17, 48]
    elif mode == 'random2':
        class_order = [1, 51,  4, 13, 24, 40,  7,  6, 36, 38, 47, 29,  5, 21, 25, 37, 23,
                       50, 26,  0, 20, 42, 16, 35, 22, 34, 46, 48, 10,  9, 43, 52, 49, 33,
                       41,  8, 11, 28, 45, 17, 32, 31, 39,  2,  3, 15, 30, 12, 14, 53, 18,
                       19, 44, 27]
    elif mode == 'random3':
        class_order = [9, 52,  7, 46, 47, 27, 37, 41, 23, 51, 32, 33, 12, 14, 44, 50, 36,
                       6, 10, 30, 25,  2, 35, 29, 40, 43, 24, 13, 20, 18,  1, 48,  5, 11,
                       16,  0, 31, 17, 28, 45,  3, 21,  8, 53, 22, 19, 26, 15, 42, 38, 49,
                       34,  4, 39]
    elif mode == 'random4':
        class_order = [19, 18, 40, 33,  4, 39, 17,  8, 34,  5, 13, 21, 25, 48, 37,  2, 20,
                       22, 52, 27, 16,  6, 29, 45, 51, 44, 50, 26, 43, 30,  7, 35, 11, 41,
                       10, 24, 31, 32, 23, 47,  3, 14, 42, 38,  9,  0, 28, 46, 15,  1, 53,
                       36, 49, 12]
    elif mode == 'no':
        class_order = None

    return class_order
