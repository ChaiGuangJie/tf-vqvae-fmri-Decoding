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
    for start, end in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90),
                       (90, 100), (100, 110), (110, 120), (120, 128)]:
        # [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 128)]
        with open("testlosslog/test_loss_{}_{}_wd_0.05.json".format(start, end), 'r') as fp:
            lossf = json.load(fp)
            loss_list = lossf['loss']['0']
            for loss in loss_list:
                all.append(loss[-1])
    return np.array(all)


def read_test_loss_of_slice():
    all_loss = []
    for start, end in [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90), (90, 105), (105, 120), (120, 128)]:
        # [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 128)]
        # [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90),(90, 100), (100, 110), (110, 120), (120, 128)]
        with open("testlosslog/test_loss_{}_{}_wd_0.05_2.json".format(start, end), 'r') as fp:
            lossf = json.load(fp)
            loss_list = lossf['loss']['0']
            for fmap_loss in loss_list:
                fmap_loss_list = []
                for slice_loss in fmap_loss:
                    fmap_loss_list.append(slice_loss[-1])
                all_loss.append(fmap_loss_list)
    return np.array(all_loss)


def generate_reciprocal_weight(all):
    weightArr = np.empty((128,))
    # all = np.array(all)
    sortedIdx = np.argsort(all)
    for i, j in zip(sortedIdx, sortedIdx[::-1]):
        weightArr[j] = 1 / all[i]
    # print(all)
    return weightArr


def save_weight_to_file(filename, weightArr):
    with open("testlosslog/{}".format(filename), 'w') as fp:
        json.dump({'weight': list(weightArr)}, fp)

    print('end')


def generate_reduced_weight(all, threshold):
    weightArr = np.where(all < threshold, 1, 0)
    return weightArr.tolist()


# def generate_weight_matrix():
#     with open('/data1/home/guangjie/Project/python/tf-vqvae/testlosslog/slice_weight.json') as fp1:
#         w1 = json.load(fp1)['weight']
#         with open('/data1/home/guangjie/Project/python/tf-vqvae/testlosslog/slice_weight_2.json') as fp2:
#             w2 = json.load(fp2)['weight']
#             assert len(w1) == len(w2)
#             for w
def read_eval_loss():
    with open('testlosslog/eval_loss/feature_map_loss.json', 'r') as fp:
        lossfile = json.load(fp)
        # train_loss = lossfile['train']
        test_loss = lossfile['test']
        return test_loss
        # for i, (train, test) in enumerate(zip(train_loss, test_loss)):
        #     print('fmap_{} :'.format(i), train, test)


# def mse_solution():
#     with h5py.File(
#             "/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_map/subject_1/rt/frame_0/subject_1_frame_0_ze_fmap_all.hdf5",
#             'r') as zf:
#         predict_latent = zf['latent']  # shape = (6000,32,32,128)
#         with h5py.File()
#         for i in range(1024):
#             latent


if __name__ == '__main__':
    # all_loss = read_test_loss()
    # all_loss = read_test_loss_of_slice()
    # print(all_loss)
    all_loss = read_eval_loss()
    weightArr = generate_reduced_weight(np.array(all_loss), 0.00038)
    save_weight_to_file(filename='eval_loss/weight.json', weightArr=weightArr)
    print('end')
