import os
import json
import h5py
import numpy as np


# r_list = []
# for i in range(1024):
#     r_list.append(os.path.exists(
#         '/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_1/frame_0/subject_1_regression_model_i_4917_o_128_latent_{}.pth'.format(
#             i)))
#
# print(sum(r_list))
def read_test_loss():
    all = []
    for start, end in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 128)]:
        with open("testlosslog/test_loss_{}_{}_wd_0.0013.json".format(start, end), 'r') as fp:
            lossf = json.load(fp)
            loss_list = lossf['loss']['1']
            for loss in loss_list:
                all.append(loss[-1])
    return np.array(all)


def generate_reciprocal_weight(all):
    weightArr = np.empty((128,))
    # all = np.array(all)
    sortedIdx = np.argsort(all)
    for i, j in zip(sortedIdx, sortedIdx[::-1]):
        weightArr[j] = 1 / all[i]
    # print(all)
    return weightArr


def save_weight_to_file(weightArr):
    with open("testlosslog/weight.json", 'w') as fp:
        json.dump({'weight': list(weightArr)}, fp)

    print('end')


def generate_reduced_weight(all):
    weightArr = np.where(all < 0.0005, 1, 0)
    weightArr = [int(i) for i in list(weightArr)]
    return weightArr


if __name__ == '__main__':
    all_loss = read_test_loss()
    weightArr = generate_reduced_weight(all_loss)
    save_weight_to_file(weightArr=weightArr)
