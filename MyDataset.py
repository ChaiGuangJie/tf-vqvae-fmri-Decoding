import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import scipy.io as sio
import json
from sklearn import preprocessing


class Stimuli_Dataset(Dataset):
    def __init__(self, file, dt_key, transpose=True):
        f = h5py.File(file, 'r')
        self.dt = f[dt_key]  # shape = (108000,3,128,128)
        self.transpose = transpose

    def __getitem__(self, item):
        if self.transpose:
            data = self.dt[item].transpose((2, 1, 0)) / 255.0
        else:
            data = self.dt[item] / 225.0
        return data

    def __len__(self):
        return self.dt.shape[0]


class vqvae_ze_dataset(Dataset):
    def __init__(self, file):
        zef = h5py.File(file, 'r')
        self.latent = zef['latent']  # shape = (540,1024,128) latent

    def __getitem__(self, item):
        return self.latent[item].reshape((32, 32, 128))

    def __len__(self):
        return self.latent.shape[0]


class vqvae_zq_dataset(Dataset):
    def __init__(self, file, frame_idx=0, time_step=15):
        zqf = h5py.File(file, 'r')
        self.zq = zqf['zq']
        self.frame_idx = frame_idx
        self.time_step = time_step

    def __getitem__(self, item):
        return self.zq[self.frame_idx + item * self.time_step]

    def __len__(self):
        return self.zq.shape[0] // self.time_step


class vqvae_k_dataset(Dataset):
    def __init__(self, kfile, frame_idx=None, time_step=15):
        kf = h5py.File(kfile, 'r')
        if not frame_idx:
            self.k = kf['k'][:]  # .reshape(-1, 1024).astype(np.float32)
        else:
            self.k = kf['k'][frame_idx::time_step]

    def __getitem__(self, item):
        return self.k[item]

    def __len__(self):
        return self.k.shape[0]


class vqvae_one_frame_k_dataset(Dataset):
    def __init__(self, kfile):
        kf = h5py.File(kfile, 'r')
        self.k = kf['k'][:]

    def __getitem__(self, item):
        return self.k[item]

    def __len__(self):
        return self.k.shape[0]


class fmri_k_dataset(Dataset):
    def __init__(self, fmri_file, k_file, fmri_key, frame_idx, latence_idx, time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        self.resp = np.nan_to_num(fmrif[fmri_key][:])
        kf = h5py.File(k_file, 'r')
        self.k = kf['k'][frame_idx::time_step].reshape(-1, 1024).astype(np.int64)
        self.latence_idx = latence_idx
        assert self.resp.shape[-1] == self.k.shape[0]

    def __getitem__(self, item):
        fmri = self.resp[:, item]
        k = self.k[item, self.latence_idx]  # / 512  # , self.row
        return fmri, k

    def __len__(self):
        return self.resp.shape[-1]


# class fmri_vector_dataset(Dataset):
#     def __init__(self, fmri_file, k_file, embeds, mean, std, fmri_key='rt', frame_idx=0, latent_idx=0, time_step=15):
#         fmrif = h5py.File(fmri_file, 'r')
#         self.response = (fmrif[fmri_key][:] - mean) / std
#         kf = h5py.File(k_file, 'r')
#         self.k_data = kf['k'][:].reshape(-1, 32 * 32)[frame_idx::time_step, latent_idx]  # shape=(108000,32,32)
#         self.embeds = embeds
#         # self.frame_idx = frame_idx
#         # self.latent_idx = latent_idx
#         # self.time_step = time_step
#         self.mean = mean
#         self.std = std
#
#     def __getitem__(self, item):
#         fmri = self.response[:, item]
#         # fmri = np.nan_to_num(fmri)
#         # fmri = (fmri - self.mean) / self.std
#         # ks = self.k_data[item % self.time_step + self.frame_idx].flatten()  # shape = (32,32) (1024,)
#         # k = ks[self.latent_idx]
#         k = self.k_data[item]
#         vector = self.embeds[k]
#
#         return fmri, vector
#
#     def __len__(self):
#         return self.response.shape[-1]

class fmri_vector_dataset(Dataset):
    def __init__(self, fmri_file, zq_file, mean, std, fmri_key, latent_key, frame_idx=0, latent_idx=0, time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        if not mean or not std:
            self.response = fmrif[fmri_key][:]
        else:
            self.response = (fmrif[fmri_key][:] - mean) / std  # todo 这样可以？
        zqf = h5py.File(zq_file, 'r')
        self.zq = zqf[latent_key][frame_idx::time_step]  # shape=(108000,32,32,128)

        # self.frame_idx = frame_idx
        self.latent_idx = latent_idx
        # self.time_step = time_step
        self.mean = mean
        self.std = std

    def __getitem__(self, item):
        fmri = self.response[:, item]
        # fmri = np.nan_to_num(fmri)
        # fmri = (fmri - self.mean) / self.std

        vector = self.zq[item].reshape(1024, 128)

        return fmri, vector[self.latent_idx]

    def __len__(self):
        return self.response.shape[-1]


class fmri_vector_k_dataset(Dataset):
    def __init__(self, fmri_file, latent_file, k_file, mean, std, fmri_key, latent_key, frame_idx=0, latent_idx=0,
                 time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        if not mean or not std:
            self.response = fmrif[fmri_key][:]
        else:
            self.response = (fmrif[fmri_key][:] - mean) / std  # todo 这样可以？
        zqf = h5py.File(latent_file, 'r')
        self.zq = zqf[latent_key][frame_idx::time_step]  # shape=(108000,32,32,128)

        kf = h5py.File(k_file, 'r')
        self.k = kf['k'][frame_idx::time_step].reshape(-1, 1024).astype(np.int64)
        # self.frame_idx = frame_idx
        self.latent_idx = latent_idx
        # self.time_step = time_step
        self.mean = mean
        self.std = std

    def __getitem__(self, item):
        fmri = self.response[:, item]
        # fmri = np.nan_to_num(fmri)
        # fmri = (fmri - self.mean) / self.std

        vector = self.zq[item].reshape(1024, 128)
        k = self.k[item, self.latent_idx]

        return fmri, vector[self.latent_idx], k

    def __len__(self):
        return self.response.shape[-1]


class fmri_latence_dataset(Dataset):
    '''获得一个fmri和其对应的15帧视频'''

    def __init__(self, fmri_file, latence_file, k_file, fmri_key, latence_key, latence_idx, time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        self.resp = fmrif[fmri_key][:]  # 加载到内存加快读取
        latencef = h5py.File(latence_file, 'r')
        self.latence = latencef[latence_key]
        kf = h5py.File(k_file, 'r')
        self.k = kf['k']

        self.row = latence_idx / 32
        self.col = latence_idx % 32
        self.time_step = time_step
        assert self.resp.shape[-1] == self.latence.shape[0] / time_step

    def __getitem__(self, item):
        fmri = self.resp[:, item]
        begin = item * self.time_step
        end = begin + self.time_step
        latence = self.latence[begin:end, self.row, self.col]  # todo 检查这样跟flatten()之后再[latence_idx]有什么区别
        k = self.k[begin:end, self.row, self.col]
        return fmri, latence, k

    def __len__(self):
        return self.resp.shape[-1]


class fmri_dataset(Dataset):
    def __init__(self, fmri_file, mean, std, dt_key):  # todo fmri file 的 mean，std 必须对应同一个被试
        f = h5py.File(fmri_file, 'r')
        if not mean or not std:
            self.resp = f[dt_key][:]
        else:
            self.resp = (f[dt_key][:] - mean) / std

    def __getitem__(self, item):
        return self.resp[:, item]

    def __len__(self):
        return self.resp.shape[-1]


class fmri_fpoint_dataset(Dataset):
    def __init__(self, fmri_file, k_file, embeds_file, fmri_key, frame_idx, fpoint_idx, mean=None, std=None,
                 time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        if not mean or not std:
            self.resp = fmrif[fmri_key][:]
        else:
            self.resp = (fmrif[fmri_key][:] - mean) / std  # todo 这样可以？

        kf = h5py.File(k_file, 'r')
        row_column = fpoint_idx / 128  # todo 会自动类型转换？ 用//替换
        self.latent_idx = fpoint_idx % 128
        row = row_column / 32
        column = row_column % 32
        self.k = kf['k'][frame_idx::time_step, row, column].astype(np.int64)
        embedf = h5py.File(embeds_file, 'r')
        self.embeds = embedf['embeds'][:]
        self.frame_idx = frame_idx
        self.time_step = time_step

    def __getitem__(self, item):
        fmri = self.resp[:, item]
        k = self.k[item]
        fpoint = self.embeds[k, self.latent_idx]
        return fmri, fpoint

    def __len__(self):
        return self.resp.shape[-1]


class fmri_fmap_dataset(Dataset):
    def __init__(self, fmri_file, k_file, embeds_file, fmri_key, frame_idx, fmap_idx, time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        self.resp = fmrif[fmri_key][:]
        kf = h5py.File(k_file, 'r')
        self.k = kf['k'][frame_idx::time_step].reshape(-1, 1024).astype(np.int64)
        embedf = h5py.File(embeds_file, 'r')
        self.embeds = embedf['embeds'][:]
        self.frame_idx = frame_idx
        self.fmap_idx = fmap_idx
        self.time_step = time_step

    def __getitem__(self, item):
        fmri = self.resp[:, item]
        k = self.k[item]
        fmap = self.embeds[k][:, self.fmap_idx]
        return fmri, fmap

    def __len__(self):
        return self.resp.shape[-1]


class fmri_fmap_all_k_dataset(Dataset):
    def __init__(self, fmri_file, k_file, embeds_file, fmri_key, dt_key, frame_idx, fmap_idx, sample_interval=36,
                 time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        self.resp = fmrif[fmri_key][:]
        kf = h5py.File(k_file, 'r')
        st_idx = np.sort(np.array(range(frame_idx, 108000, time_step)))
        st_per_frame_idx = set(range(7200))
        st_test_idx = set(range(0, 7200, sample_interval))
        st_train_idx = st_per_frame_idx - st_test_idx
        st_train_idx = np.sort(np.array(list(st_train_idx)))
        st_test_idx = np.sort(np.array(list(st_test_idx)))
        if dt_key == 'rt':
            idx = [st_idx[st_train_idx]][0]
            self.k = kf['k'][:][idx].reshape(-1, 1024).astype(np.int64)
        elif dt_key == 'rv':
            idx = [st_idx[st_test_idx]][0]
            self.k = kf['k'][:][idx].reshape(-1, 1024).astype(np.int64)
        else:
            raise AttributeError('dt_key error')
        embedf = h5py.File(embeds_file, 'r')
        self.embeds = embedf['embeds'][:]
        self.frame_idx = frame_idx
        self.fmap_idx = fmap_idx

    def __getitem__(self, item):
        fmri = self.resp[:, item]
        k = self.k[item]
        fmap = self.embeds[k][:, self.fmap_idx]
        return fmri, fmap

    def __len__(self):
        return self.resp.shape[-1]


class fmri_fmap_slice_dataset(Dataset):
    def __init__(self, fmri_file, k_file, embeds_file, fmri_key, frame_idx, fmap_idx, slice_idx, time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        self.resp = fmrif[fmri_key][:]
        kf = h5py.File(k_file, 'r')
        self.k = kf['k'][frame_idx::time_step, slice_idx, :].astype(np.int64)  # [frame_idx::time_step, :, slice_idx]
        embedf = h5py.File(embeds_file, 'r')
        self.embeds = embedf['embeds'][:]  # shape = (512,128)
        # self.frame_idx = frame_idx
        self.fmap_idx = fmap_idx
        # self.time_step = time_step

    def __getitem__(self, item):
        fmri = self.resp[:, item]
        k = self.k[item]
        feat_slice_map = self.embeds[k][:, self.fmap_idx]
        return fmri, feat_slice_map

    def __len__(self):
        return self.resp.shape[-1]


def get_vim2_fmri_mean_std(voxel_train_file, dt_key):
    with h5py.File(voxel_train_file, 'r') as vf:
        r = vf[dt_key][:].flatten()  # todo 需要test数据的mean，std vf[dt_key][roi_idx, :].flatten()
        r = np.nan_to_num(r)
        mean = np.mean(r)
        std = np.std(r)
    return mean, std


def get_purdue_fmri_mean_std(voxel_train_file, dt_key):
    with h5py.File(voxel_train_file, 'r') as f:
        r = f[dt_key][:].flatten()
        r = np.nan_to_num(r)
        mean = np.mean(r)
        std = np.std(r)
    return mean, std


class purdue_fmri_fmap_dataset(Dataset):
    def __init__(self, fmri_file, k_file, embeds_file, fmri_key, frame_idx, fmap_idx, mean=None, std=None):
        fmrif = h5py.File(fmri_file, 'r')
        # self.fmri = fmrif[fmri_key][:]
        if not mean or not std:
            self.response = fmrif[fmri_key][:]
        else:
            self.response = (fmrif[fmri_key][:] - mean) / std  # todo 这样可以？
        kf = h5py.File(k_file, 'r')
        self.k = kf['k'][:].reshape(-1, 1024).astype(np.int64)
        embedf = h5py.File(embeds_file, 'r')
        self.embeds = embedf['embeds'][:]

        self.frame_idx = frame_idx
        self.fmap_idx = fmap_idx

    def __getitem__(self, item):
        fmri = self.response[:, item]
        k = self.k[item]
        fmap = self.embeds[k][:, self.fmap_idx]
        return fmri, fmap  # todo nan

    def __len__(self):
        return self.response.shape[-1]


class vim2_predict_and_true_k_dataset(Dataset):
    def __init__(self, predict_latent_file, k_file, frame_idx, latent_idx, time_step=15):
        predict_f = h5py.File(predict_latent_file, 'r')
        self.predict_latent = predict_f['latent'][:].reshape(-1, 1024, 128)  # shape = (len,32,32,128)
        kf = h5py.File(k_file, 'r')
        self.k = kf['k'][frame_idx::time_step].reshape(-1, 1024).astype(np.int64)
        self.latent_idx = latent_idx
        assert len(self.predict_latent) == len(self.k)

    def __getitem__(self, item):
        latent = self.predict_latent[item, self.latent_idx, :]
        k = self.k[item, self.latent_idx]
        return latent, k

    def __len__(self):
        return len(self.predict_latent)


class vim2_predict_and_all_true_k_dataset(Dataset):
    def __init__(self, predict_latent_file, k_file, latent_idx):
        predict_f = h5py.File(predict_latent_file, 'r')
        self.predict_latent = predict_f['latent'][:].reshape(-1, 1024, 128)  # shape = (len,32,32,128)
        kf = h5py.File(k_file, 'r')
        self.k = kf['k'][:].reshape(-1, 1024).astype(np.int64)
        self.latent_idx = latent_idx
        assert len(self.predict_latent) == len(self.k)

    def __getitem__(self, item):
        latent = self.predict_latent[item, self.latent_idx, :]
        k = self.k[item, self.latent_idx]
        return latent, k

    def __len__(self):
        return len(self.predict_latent)


class vim2_predict_latent_dataset(Dataset):
    def __init__(self, predict_latent_file, latent_idx):
        predict_f = h5py.File(predict_latent_file, 'r')
        self.predict_latent = predict_f['latent'][:]  # .reshape(-1, 1024, 128)  # shape = (len,32,32,128)
        # self.latent_idx = latent_idx
        self.row = latent_idx // 32
        self.col = latent_idx % 32

    def __getitem__(self, item):
        latent = self.predict_latent[item, self.row, self.col, :]
        return latent

    def __len__(self):
        return len(self.predict_latent)


class vim1_fmri_dataset(Dataset):
    def __init__(self, file, voxel_select_file, rois, dt_key, subject, start=0):
        idxAll = self.get_vim1_roi_idx(file, 'roiS{}'.format(subject), rois)
        self.n_voxels = len(idxAll)
        select_fp = open(voxel_select_file)
        select_idx = json.load(select_fp)
        voxelf = h5py.File(file, 'r')
        self.resp = np.nan_to_num(voxelf[dt_key][:, idxAll][start:, select_idx])

    def __getitem__(self, item):
        return self.resp[item].astype(np.float32)  # todo [item]

    def __len__(self):
        return self.resp.shape[0]  # todo [0]

    def get_vim1_roi_idx(self, voxel_file, roi_key, roi_nums):
        with h5py.File(voxel_file, 'r') as vrf:
            roi = vrf[roi_key][:].flatten()
            idxAll = []
            for i in roi_nums:
                idx = np.where(roi == i)[0]
                idxAll = np.concatenate([idxAll, idx])
            idxAll = np.sort(idxAll).astype(np.uint64)
            return idxAll


class vim1_fmri_transpose_dataset(Dataset):
    def __init__(self, file, rois, dt_key, subject):
        idxAll = self.get_vim1_roi_idx(file, 'roiS{}'.format(subject), rois)
        self.n_voxels = len(idxAll)
        voxelf = h5py.File(file, 'r')
        self.resp = np.nan_to_num(voxelf[dt_key][:, idxAll])

    def __getitem__(self, item):
        return self.resp[:, item].astype(np.float32)  # todo [item]

    def __len__(self):
        return self.resp.shape[1]  # todo [0]

    def get_vim1_roi_idx(self, voxel_file, roi_key, roi_nums):
        with h5py.File(voxel_file, 'r') as vrf:
            roi = vrf[roi_key][:].flatten()
            idxAll = []
            for i in roi_nums:
                idx = np.where(roi == i)[0]
                idxAll = np.concatenate([idxAll, idx])
            idxAll = np.sort(idxAll).astype(np.uint64)
            return idxAll


class vim1_stimuli_dataset(Dataset):
    def __init__(self, file, dtKey, n_channel):
        mat = sio.loadmat(file)
        # mat = h5py.File(file,'r')
        self.stim = mat[dtKey][:]
        self.stim = np.clip((self.stim - np.min(self.stim)) / np.max(self.stim - np.min(self.stim)), 0, 1)
        self.n_channel = n_channel

    def __getitem__(self, item):
        if self.n_channel > 1:
            stimItem = np.repeat(self.stim[item][:, :, np.newaxis], self.n_channel, axis=-1)
        else:
            stimItem = self.stim[item]
        return stimItem

    def __len__(self):
        return self.stim.shape[0]


class vim1_blur_stimuli_dataset(Dataset):
    def __init__(self, file, dtKey, n_channel):
        mat = sio.loadmat(file)
        # mat = h5py.File(file, 'r')
        self.stim = mat[dtKey][:]
        self.stim = np.clip((self.stim - np.min(self.stim)) / np.max(self.stim - np.min(self.stim)), 0, 1)
        self.n_channel = n_channel

    def __getitem__(self, item):
        if self.n_channel > 1:
            stimItem = np.repeat(self.stim[item][:, :, np.newaxis], self.n_channel, axis=-1)
        else:
            stimItem = self.stim[item]
        return stimItem

    def __len__(self):
        return self.stim.shape[0]


def get_vim1_roi_idx(voxel_file, roi_key, roi_nums):
    with h5py.File(voxel_file, 'r') as vrf:
        roi = vrf[roi_key][:].flatten()
        idxAll = []
        for i in roi_nums:
            idx = np.where(roi == i)[0]
            idxAll = np.concatenate([idxAll, idx])
        idxAll = np.sort(idxAll).astype(np.uint64)
        return idxAll


class vim1_fmri_k_dataset(Dataset):
    def __init__(self, fmri_file, voxel_select_file, k_file, embeds_file, fmri_key, fmap_idx, isTrain, rois,
                 split_point, subject, normalize=False):
        idxAll = get_vim1_roi_idx(fmri_file, 'roiS{}'.format(subject), rois)
        voxelf = h5py.File(fmri_file, 'r')
        select_fp = open(voxel_select_file)
        select_idx = json.load(select_fp)

        if isTrain:
            self.resp = np.nan_to_num(voxelf[fmri_key][:split_point, idxAll][:, select_idx])
        else:
            self.resp = np.nan_to_num(voxelf[fmri_key][split_point:, idxAll][:, select_idx])

        if normalize:
            mean = np.mean(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))
            std = np.std(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))
            self.resp = (self.resp - mean) / std

        kf = h5py.File(k_file, 'r')
        if isTrain:
            self.k = kf['k'][:split_point]
        else:
            self.k = kf['k'][split_point:]
        embedf = h5py.File(embeds_file, 'r')
        self.embeds = embedf['embeds'][:]
        self.fmap_idx = fmap_idx
        assert self.resp.shape[0] == self.k.shape[0]

    def __getitem__(self, item):
        voxel = self.resp[item].astype(np.float32)
        k = self.k[item].flatten().astype(np.uint64)
        fmap = self.embeds[k, self.fmap_idx]
        return voxel, fmap

    def __len__(self):
        return self.resp.shape[0]


class vim1_val_fmri_dataset(Dataset):
    def __init__(self, fmri_file, voxel_select_file, fmri_key, rois, subject, normalize):
        idxAll = get_vim1_roi_idx(fmri_file, 'roiS{}'.format(subject), rois)
        voxelf = h5py.File(fmri_file, 'r')
        select_fp = open(voxel_select_file)
        select_idx = json.load(select_fp)
        self.scaler = preprocessing.MinMaxScaler()
        # self.scaler = preprocessing.Normalizer()
        # self.scaler = preprocessing.StandardScaler()

        self.resp = np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx])


        if normalize:
            # mean = np.mean(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))
            # std = np.std(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))
            # self.resp = (self.resp - mean) / std
            self.resp = self.scaler.fit_transform(self.resp)

    def __getitem__(self, item):
        voxel = self.resp[item].astype(np.float32)
        return voxel

    def __len__(self):
        return self.resp.shape[0]


class vim1_fmri_k_latent_dataset(Dataset):
    def __init__(self, fmri_file, voxel_select_file, k_file, embeds_file, fmri_key, latent_idx, isTrain, rois,
                 split_point, subject, normalize=False):
        idxAll = get_vim1_roi_idx(fmri_file, 'roiS{}'.format(subject), rois)
        voxelf = h5py.File(fmri_file, 'r')
        select_fp = open(voxel_select_file)
        select_idx = json.load(select_fp)

        if isTrain:
            self.resp = np.nan_to_num(voxelf[fmri_key][:split_point, idxAll][:, select_idx])
        else:
            self.resp = np.nan_to_num(voxelf[fmri_key][split_point:, idxAll][:, select_idx])

        if normalize:
            mean = np.mean(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))
            std = np.std(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))
            self.resp = (self.resp - mean) / std

        kf = h5py.File(k_file, 'r')
        if isTrain:
            self.k = kf['k'][:split_point].reshape(-1, 1024)[:, latent_idx]
        else:
            self.k = kf['k'][split_point:].reshape(-1, 1024)[:, latent_idx]
        embedf = h5py.File(embeds_file, 'r')
        self.embeds = embedf['embeds'][:]
        self.latent_idx = latent_idx
        assert self.resp.shape[0] == self.k.shape[0]

    def __getitem__(self, item):
        voxel = self.resp[item].astype(np.float32)
        k = self.k[item].flatten().astype(np.uint64)
        latent = self.embeds[k]
        return voxel, latent

    def __len__(self):
        return self.resp.shape[0]


class vim1_fmri_fmap_dataset(Dataset):
    def __init__(self, fmri_file, voxel_select_file, latent_file, fmri_key, fmap_idx, isTrain, rois,
                 split_point, subject, normalize):
        idxAll = get_vim1_roi_idx(fmri_file, 'roiS{}'.format(subject), rois)
        voxelf = h5py.File(fmri_file, 'r')
        select_fp = open(voxel_select_file)
        select_idx = json.load(select_fp)

        latentf = h5py.File(latent_file, 'r')
        self.scaler = preprocessing.MinMaxScaler()
        # self.scaler = preprocessing.StandardScaler()
        # self.scaler = preprocessing.Normalizer()
        # self.scaler.fit(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]).transpose())
        # .fit(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]).transpose())
        if normalize:
            all_resp = self.scaler.fit_transform(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))#.transpose()
        else:
            all_resp = np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx])

        if isTrain:
            # self.resp = np.nan_to_num(voxelf[fmri_key][:split_point, idxAll][:, select_idx])
            self.resp = all_resp[:split_point]
            self.latent = latentf['latent'][:split_point, :, :, fmap_idx]

        else:
            self.resp = all_resp[split_point:]
            # self.resp = np.nan_to_num(voxelf[fmri_key][split_point:, idxAll][:, select_idx])
            self.latent = latentf['latent'][split_point:, :, :, fmap_idx]

        # if normalize:
        #     # mean = np.mean(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))
        #     # std = np.std(np.nan_to_num(voxelf[fmri_key][:, idxAll][:, select_idx]))
        #     # self.resp = (self.resp - mean) / std
        #     # self.resp = self.scaler.fit_transform(self.resp)
        #     # self.resp = self.scaler.transform(self.resp).transpose()
        #     pass

    def __getitem__(self, item):
        fmri = self.resp[item].astype(np.float32)
        fmap = self.latent[item].flatten()
        return fmri, fmap

    def __len__(self):
        return len(self.latent)


if __name__ == '__main__':
    # dataset = vim1_stimuli_dataset("/data1/home/guangjie/Data/vim-1/Stimuli.mat", 'stimTrn', 3)
    # dataset = vim1_fmri_dataset("/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat", [1, 2], 'dataTrnS1', 1)
    dataset = vim1_fmri_k_dataset(fmri_file="/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat",
                                  k_file="/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/k_from_vqvae_st.hdf5",
                                  embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
                                  fmri_key='dataTrnS1', fmap_idx=0, isTrain=False, rois=[1, 2], split_point=1700,
                                  subject=1)
    print(len(dataset))
    voxel, k = dataset[1]
    print(voxel.shape, k.shape)
    pass
