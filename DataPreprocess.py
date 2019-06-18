import h5py
import numpy as np
import json
from PIL import Image, ImageSequence
import cv2
import time
import scipy.io as sio
import torch
import os
import tensorflow as tf
from MyDataset import vim1_fmri_transpose_dataset, vim2_fmri_transpose_dataset
import itertools
from sklearn import preprocessing


def filter_voxel_of_v1234(subject):
    with h5py.File(
            "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_rv_rva0.hdf5".format(
                subject), 'r') as rtf:
        rt = rtf['rt'][:]
        # std = []
        # for item in rt:
        #     std.append(np.std(item))
        # std = np.array(std)
        # effectiveIdx = np.where(std != 0)
        with open("selectVoxel_subject_{}.json".format(subject)) as fp:
            effectiveIdx = json.load(fp)
            effectiveIdx = np.array(effectiveIdx)
        effectiveRt = rt[effectiveIdx]  # rt[63:]
        with h5py.File(
                "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_train.hdf5".format(
                    subject), 'w') as trainf:
            trainf.create_dataset('rt', data=effectiveRt[:, :7000])
        with h5py.File(
                "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_test.hdf5".format(
                    subject), 'w') as testf:
            testf.create_dataset('rt', data=effectiveRt[:, 7000:])
    print('end')


def filter_voxel_of_v1234_uniform_sample(subject):
    with h5py.File(
            "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_ips_rt_rv_rva0.hdf5".format(
                subject), 'r') as rtf:
        rt = rtf['rt'][:]
        # std = []
        # for item in rt:
        #     std.append(np.std(item))
        # std = np.array(std)
        # effectiveIdx = np.where(std != 0)
        # with open("selectVoxel_subject_{}.json".format(subject)) as fp:
        #     effectiveIdx = json.load(fp)
        #     effectiveIdx = np.array(effectiveIdx)
        effectiveRt = rt[83:]  # rt[effectiveIdx]  #
        effAllIdx = set(range(7200))
        effTestIdx = set(range(0, 7200, 36))
        effTrainIdx = effAllIdx - effTestIdx

        effTestIdx = np.sort(np.array(list(effTestIdx)))  # 必须sort，不然会乱序
        effTrainIdx = np.sort(np.array(list(effTrainIdx)))

        with h5py.File(
                "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_ips_rt_train.hdf5".format(
                    subject), 'w') as trainf:
            trainf.create_dataset('rt', data=effectiveRt[:, effTrainIdx])
        with h5py.File(
                "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_ips_rt_test.hdf5".format(
                    subject), 'w') as testf:
            testf.create_dataset('rt', data=effectiveRt[:, effTestIdx])
    print(effectiveRt.shape)
    print('end')


def transform_jpg_to_hdf5_of_purdue_dataset(dt_key, key, n_seg, frame_idx=0):
    with h5py.File(
            "/data1/home/guangjie/Data/purdue/exprimentData/Stimuli/Stimuli_{}_frame_{}.hdf5".format(key, frame_idx),
            'w') as sf:
        dataset = sf.create_dataset(dt_key, shape=(240 * n_seg, 128, 128, 3))  # 14400
        for seg in np.arange(1, n_seg + 1):
            videoCap = cv2.VideoCapture(
                "/data1/home/guangjie/Data/purdue/Stimuli/video_fmri_dataset/stimuli/{}{}.mp4".format(
                    'seg' if key == 'train' else 'test', seg))
            current_item = 0
            for idx in range(14400):
                # Capture frame-by-frame
                ret, frame = videoCap.read()
                if idx % 240 == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 非常重要
                    frame = cv2.resize(frame, (128, 128))
                    dataset[current_item] = frame
                    current_item += 1
                # cv2.imshow('frame', frame)
                # cv2.waitKey(33)
                # print(frame)
                # print(frame)
            videoCap.release()
            print(seg)


def get_top_k_idx_of_purdue_data(k):
    zmf = sio.loadmat("/data1/home/guangjie/Data/purdue/exprimentData/fMRI/Zmat.mat")
    zmat = zmf['Zmat']
    meanZmat = np.abs(np.mean(zmat, axis=1))  # todo abs 需要?
    topk = np.argsort(meanZmat)[-k:]
    return np.sort(topk)


def concatenate_purdue_train_fmri(topk, subject=1):
    matf = sio.loadmat(
        "/data1/home/guangjie/Data/purdue/video_fmri_dataset/subject{}/fmri/training_fmri.mat".format(subject))
    dt1 = matf['fmri']['data1'][0][0]
    dt2 = matf['fmri']['data2'][0][0]
    topk_idx = get_top_k_idx_of_purdue_data(topk)
    # rt = (dt1 + dt2) / 2
    rt = np.mean([dt1, dt2], axis=0)
    rt = rt[topk_idx].reshape(len(topk_idx), rt.shape[1] * rt.shape[2])
    rt = np.nan_to_num(rt)
    with h5py.File("/data1/home/guangjie/Data/purdue/exprimentData/fMRI/subject{}_rt_fmri.hdf5".format(subject),
                   'w') as trainf:
        trainf.create_dataset('rt', data=rt)


def concatenate_purdue_test_fmri(topk, repeat, subject=1):
    assert repeat <= 10
    assert topk <= 59412
    with h5py.File(
            "/data1/home/guangjie/Data/purdue/video_fmri_dataset/subject{}/fmri/testing_fmri.mat".format(subject),
            'r') as matf:
        fmris = []
        for i in range(5):
            path = "fmritest/test{}".format(i + 1)
            fmris.append(matf[path][:].transpose((2, 1, 0)))  # shape = (10,240,59412)
        rv = np.concatenate(fmris, axis=1)
        # rv = rv.transpose((2, 1, 0))
        # rv = np.mean(fmris, axis=0)  # todo 有问题
        rv = np.mean(rv[:, :, :repeat], axis=2)
        topk_idx = get_top_k_idx_of_purdue_data(topk)
        rv = np.nan_to_num(rv[topk_idx])
        with h5py.File("/data1/home/guangjie/Data/purdue/exprimentData/fMRI/subject{}_rv_fmri.hdf5".format(subject),
                       'w') as testf:
            testf.create_dataset('rv', data=rv)


def extract_vim2_voxel():
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/orig/VoxelResponses_subject1.mat", 'r') as f:
        rt = f['rt']
        rt_train = np.nan_to_num(rt[:, :6000])
        rt_test = np.nan_to_num(rt[:, 6000:])
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/rt_train.hdf5", 'w') as trainf:
            trainf.create_dataset('rt', data=rt_train)
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/rt_test.hdf5", 'w') as testf:
            testf.create_dataset('rt', data=rt_test)

    print('end')


def get_vim1_roi_idx(roi_key, voxel_file):
    with h5py.File(voxel_file, 'r') as vrf:
        roi = vrf[roi_key][:].flatten()
        idxAll = []
        for i in [1, 2, 3, 4, 5, 6]:
            idx = np.where(roi == i)[0]
            idxAll = np.concatenate([idxAll, idx])
        idxAll = np.sort(idxAll).astype(np.uint64)
        return idxAll


def vim1_stimuli_gaussian_blur():
    from MyDataset import vim1_stimuli_dataset
    # with h5py.File("/data1/home/guangjie/Data/vim-1/Stimuli.mat", 'r') as rf:
    # rf = sio.loadmat("/data1/home/guangjie/Data/vim-1/Stimuli.mat")
    oriTrnData = vim1_stimuli_dataset("/data1/home/guangjie/Data/vim-1/Stimuli.mat", 'stimTrn', 1)  # rf['stimTrn']
    oriValData = vim1_stimuli_dataset("/data1/home/guangjie/Data/vim-1/Stimuli.mat", 'stimVal', 1)  # rf['stimVal']
    with h5py.File("/data1/home/guangjie/Data/vim-1/BlurStimuli.hdf5", 'w') as wf:
        trnData = wf.create_dataset('stimTrn', shape=(len(oriTrnData), 128, 128))
        for i in range(len(oriTrnData)):
            # img = (oriTrnData[i] * 255).astype(np.uint8)
            img = oriTrnData[i]
            # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # img = cv2.GaussianBlur(img, (9, 9), 3)
            # sobelX = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # x方向的梯度
            # sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
            img = cv2.GaussianBlur(img, (15, 15), 15)
            # img = cv2.normalize(sobelX, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            trnData[i] = img
            # cv2.imshow('img', np.concatenate([img, sobelX], axis=-1))
            cv2.imshow('img', img)
            cv2.waitKey(1)
            print(i)
        valData = wf.create_dataset('stimVal', shape=(len(oriValData), 128, 128))
        for i in range(len(valData)):
            img = valData[i]
            img = cv2.GaussianBlur(img, (15, 15), 15)
            # sobelX = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # x方向的梯度
            # sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
            # sobelX = cv2.GaussianBlur(sobelX, (9, 9), 1.5)
            # img = cv2.normalize(sobelX, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            valData[i] = img
            print(i)


# def vim1_voxel_select(dt_key):
#     file = "/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat"
#     # k: dataTrnS1 dataTrnS2 dataValS1 dataValS2 roiS1 roiS2 voxIdxS1 voxIdxS2
#     rois = ['V1', 'V2', 'V3', 'V3A', 'V3B', 'V4']
#     roi_idx = get_vim1_roi_idx('roiS1', file)
#     with h5py.File(file, 'r') as erf:
#         data = erf[dt_key]

def vim1_fmap_select(threshold, rois, wd):
    # with open('performance_roi_1236.json', 'r') as fp:
    #     performanceList = np.array(json.load(fp))
    #     minIdx = np.argsort(performanceList, axis=0)
    #     idx = np.argsort(minIdx, axis=0)[0]
    #     return idx
    with open('performance_roi_{}_wd_{}.json'.format(''.join(map(str, rois)), str(wd)), 'r') as fp:
        performance = np.array(json.load(fp))
        idx = np.where(performance < threshold)
        return idx[0]


def vim1_use_weighted_norm2_to_find_k(dt_key, rois, wd, threshold, subject):
    idx = vim1_fmap_select(threshold, rois, wd)
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
                   'r') as ebdf:
        embeds = ebdf['embeds'][:, idx]
        with h5py.File(
                "/data1/home/guangjie/Data/vim1/regressed_feature_map/subject_1/dataTrnS1/subject_1_roi_1_regressed_fmap_all_wd0.03.hdf5",
                'r') as fmapf:
            latent_data = fmapf['latent'][:, :, :, idx]
            with h5py.File(
                    "/data1/home/guangjie/Data/vim1/selected_k_from_regressed_feature_map/subject_{}_{}_roi_{}_wd_{}_td_{}_k.hdf5".format(
                        subject, dt_key, ''.join(map(str, rois)), str(wd), str(threshold)), 'w') as sf:
                k_data = sf.create_dataset('k', shape=(latent_data.shape[0], 32, 32))
                for i in range(latent_data.shape[0]):
                    expand_latent = latent_data[i][:, :, np.newaxis, :]
                    dist = np.linalg.norm(expand_latent - embeds, axis=-1)
                    k = np.argmin(dist, axis=-1)
                    k_data[i] = k
                    print(i)

    print('end')


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def select_valid_voxel(rois, subject):
    rt = vim1_fmri_transpose_dataset("/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat", rois,
                                     'dataTrnS{}'.format(subject), subject)
    validIdx = set(range(len(rt)))
    invalidIdx = []
    mask = [0, 0, 0, 0, 0]
    for i in range(len(rt)):
        voxel = rt[i]
        for j, v in enumerate(voxel[:-5]):
            if v == 0 and sum(voxel[j:j + 5] == mask):
                invalidIdx.append(i)
                break
    validIdx = validIdx - set(invalidIdx)
    with open("vim1_subject_{}_roi_{}_voxel_select.json".format(subject, ''.join(map(str, rois))),
              'w') as fp:
        json.dump(sorted(list(validIdx)), fp)
    # return sorted(list(validIdx))


def vim2_select_valid_voxel(rois, subject):
    rt = vim2_fmri_transpose_dataset(
        "/data1/home/guangjie/Data/vim-2-gallant/orig/VoxelResponses_subject{}.mat".format(subject), rois,
        'rt')
    validIdx = set(range(len(rt)))
    invalidIdx = []
    mask = [0, 0, 0, 0, 0]
    for i in range(len(rt)):
        voxel = rt[i]
        for j, v in enumerate(voxel[:-5]):
            if v == 0 and sum(voxel[j:j + 5] == mask):
                invalidIdx.append(i)
                break
    validIdx = validIdx - set(invalidIdx)
    with open("vim2_subject_{}_roi_{}_voxel_select.json".format(subject, ''.join(map(str, rois))),
              'w') as fp:
        json.dump(sorted(list(validIdx)), fp)
    # return sorted(list(validIdx))


def vim1_create_normlized_zq(dt_key):
    scaler = preprocessing.MinMaxScaler()
    # scaler = preprocessing.StandardScaler()
    # scaler = preprocessing.Normalizer()
    with h5py.File(
            "/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/k_from_vqvae_{}.hdf5".format(dt_key),
            'r') as kf:
        k = kf['k'][:].astype(np.uint64)
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", 'r') as ef:
            embeds = ef['embeds'][:]
            with h5py.File(
                    "/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/zq_of_{}_fmap_mm_scaler_18x18_blur5.hdf5".format(
                        dt_key), 'w') as zqf:
                latent = zqf.create_dataset('latent', shape=(len(k), 18, 18, 128))
                for i, _k in enumerate(k):
                    fmap = embeds[_k].reshape(1024, 128)  # .transpose()
                    fmap = scaler.fit_transform(fmap).reshape(32, 32, 128)  # .transpose()
                    # normlized_fmap = np.clip((fmap - np.min(fmap)) / np.max(fmap - np.min(fmap)), 0, 1)
                    # normlized_fmap = fmap - np.mean(fmap)
                    # ori = cv2.resize(fmap[:, :, 0], (16, 16))
                    # ori = np.clip((ori - np.min(ori)) / np.max(ori - np.min(ori)), 0, 1)
                    for j in range(128):
                        fm = fmap[:, :, j]
                        # norm_fm = np.clip((fm - np.min(fm)) / np.max(fm - np.min(fm)), 0, 1)
                        # fmap[:, :, j] = scaler.fit_transform(fm)
                        blur_fm = cv2.GaussianBlur(fm, (5, 5), 0)
                        small_blur_fm = cv2.resize(blur_fm, (18, 18))
                        # small_blur_fm = cv2.GaussianBlur(small_fm, (3, 3), 0)
                        # laplacian = cv2.Laplacian(small_blur_fm, -1)
                        latent[i, :, :, j] = small_blur_fm  # blur_small_fm
                        # norm_blur_fm = np.clip((blur_fm - np.min(blur_fm)) / np.max(blur_fm - np.min(blur_fm)), 0, 1)
                        # grid = np.concatenate((small_blur_fm, laplacian), axis=1)
                        #
                        # cv2.imshow('img', small_blur_fm)
                        # cv2.waitKey(1)
                        # print('img')
                    # fmap = fmap.reshape(256, 128)  # .transpose()
                    # fmap = scaler.fit_transform(fmap).reshape(16, 16, 128)  # .transpose()
                    # grid = np.concatenate((ori, fmap[:, :, 0]), axis=1)
                    # cv2.imshow('img', grid)
                    # cv2.waitKey(1)
                    # print('img')
                    # latent[i] = fmap
                    print(i)


def vim2_create_normlized_zq(dt_key):
    scaler = preprocessing.MinMaxScaler()
    # scaler = preprocessing.StandardScaler()
    # scaler = preprocessing.Normalizer()
    with h5py.File(
            "/data1/home/guangjie/Data/vim2/exprimentData/k_from_vqvae_{}_frame_1.hdf5".format(dt_key),
            'r') as kf:
        k = kf['k'][:].astype(np.uint64)
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", 'r') as ef:
            embeds = ef['embeds'][:]
            with h5py.File(
                    "/data1/home/guangjie/Data/vim2/exprimentData/zq_of_{}_fmap_mm_scaler_18x18_blur5.hdf5".format(
                        dt_key), 'w') as zqf:
                latent = zqf.create_dataset('latent', shape=(len(k), 18, 18, 128))
                for i, _k in enumerate(k):
                    fmap = embeds[_k].reshape(1024, 128)  # .transpose()
                    fmap = scaler.fit_transform(fmap).reshape(32, 32, 128)  # .transpose()
                    # normlized_fmap = np.clip((fmap - np.min(fmap)) / np.max(fmap - np.min(fmap)), 0, 1)
                    # normlized_fmap = fmap - np.mean(fmap)
                    # ori = cv2.resize(fmap[:, :, 0], (16, 16))
                    # ori = np.clip((ori - np.min(ori)) / np.max(ori - np.min(ori)), 0, 1)
                    for j in range(128):
                        fm = fmap[:, :, j]
                        # norm_fm = np.clip((fm - np.min(fm)) / np.max(fm - np.min(fm)), 0, 1)
                        # fmap[:, :, j] = scaler.fit_transform(fm)
                        blur_fm = cv2.GaussianBlur(fm, (5, 5), 0)
                        small_blur_fm = cv2.resize(blur_fm, (18, 18))
                        # small_blur_fm = cv2.GaussianBlur(small_fm, (3, 3), 0)
                        # laplacian = cv2.Laplacian(small_blur_fm, -1)
                        latent[i, :, :, j] = small_blur_fm  # blur_small_fm
                        # norm_blur_fm = np.clip((blur_fm - np.min(blur_fm)) / np.max(blur_fm - np.min(blur_fm)), 0, 1)
                        # grid = np.concatenate((small_blur_fm, laplacian), axis=1)
                        #
                        # cv2.imshow('img', small_blur_fm)
                        # cv2.waitKey(1)
                        # print('img')
                    # fmap = fmap.reshape(256, 128)  # .transpose()
                    # fmap = scaler.fit_transform(fmap).reshape(16, 16, 128)  # .transpose()
                    # grid = np.concatenate((ori, fmap[:, :, 0]), axis=1)
                    # cv2.imshow('img', grid)
                    # cv2.waitKey(1)
                    # print('img')
                    # latent[i] = fmap
                    print(i)


def show_mask():
    mat = sio.loadmat("/data1/home/guangjie/Data/vim-1/mask.mat")
    mask = mat['mask']
    mask = cv2.resize(mask, (32, 32))
    cv2.imshow('img', mask)
    cv2.waitKey(1)
    print('ok')


def recover_fmaps(dt_key):
    scaler = preprocessing.MinMaxScaler()
    with h5py.File(
            "/data1/home/guangjie/Data/vim2/extract_from_vqvae/k_from_vqvae_{}_frame_1.hdf5".format(dt_key),
            'r') as kf:
        k = kf['k'][:].astype(np.uint64)
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", 'r') as ef:
            embeds = ef['embeds'][:]
            with h5py.File(
                    "/data1/home/guangjie/Data/vim2/regressed_feature_map/subject_1/rt/subject_1_rt_roi_v1lhv1rhv2lhv2rhv3alhv3arhv3blhv3brhv3lhv3rhv4lhv4rh_regressed_fmap_all_wd_0.03_normalizedFmap_no_mm_18x18_blur3_nolinearConv_Adam_bn_res_32x32.hdf5",
                    'r') as latentf:
                latent = latentf['latent']
                with h5py.File(
                        "/data1/home/guangjie/Data/vim2/regressed_feature_map/subject_1/rt/subject_1_rt_roi_v1lhv1rhv2lhv2rhv3alhv3arhv3blhv3brhv3lhv3rhv4lhv4rh_regressed_fmap_all_wd_0.03_normalizedFmap_no_mm_18x18_blur3_nolinearConv_Adam_bn_inverse_32x32.hdf5",
                        'w') as inversef:
                    inverse_latent = inversef.create_dataset('latent', shape=latent.shape)
                    for i, _k in enumerate(k):
                        fmap = embeds[_k].reshape(1024, 128)  # .transpose()
                        scaler.fit(fmap)  # .reshape(32, 32, 128) #.transpose()
                        inverse_fmap = scaler.inverse_transform(latent[i].reshape(1024, 128))  #
                        inverse_latent[i] = inverse_fmap.reshape(32, 32, 128)
                        print(i)


if __name__ == '__main__':
    # transform_jpg_to_hdf5_of_purdue_dataset('sv', 'test', 5, frame_idx=0)  # st train 18
    # get_top_k_idx_of_purdue_data(k=5000)
    # concatenate_purdue_train_fmri(topk=8000)
    # concatenate_purdue_test_fmri(topk=8000, repeat=2, subject=1)
    # extract_vim2_voxel()
    # filter_voxel_of_v1234(subject=3)
    # filter_voxel_of_v1234_uniform_sample(subject=3)
    # vim1_voxel_select('dataTrn')
    # vim1_fmap_select()
    # vim1_use_weighted_norm2_to_find_k(dt_key='dataTrnS1', rois=[1, 2], wd=0.005, threshold=0.003, subject=1)
    # select_valid_voxel(rois=[6, 7], subject=1)
    # vim2_select_valid_voxel(rois=['v1lh', 'v1rh', 'v2lh', 'v2rh', 'v3alh', 'v3arh', 'v3blh', 'v3brh', 'v3lh', 'v3rh', 'v4lh', 'v4rh'], subject=1)
    # print(len(idx))
    # vim1_stimuli_gaussian_blur()
    # vim1_create_normlized_zq('st')
    # vim2_create_normlized_zq('st')
    # show_mask()
    recover_fmaps('st')
    print('end')
