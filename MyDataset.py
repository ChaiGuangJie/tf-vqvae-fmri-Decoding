import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np


class Vim2_Stimuli_Dataset(Dataset):
    def __init__(self, file, dt_key):
        f = h5py.File(file, 'r')
        self.dt = f[dt_key]  # shape = (108000,3,128,128)

    def __getitem__(self, item):
        data = self.dt[item].transpose((2, 1, 0)) / 255.0
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
    def __init__(self, kfile, frame_idx, time_step):
        kf = h5py.File(kfile, 'r')
        self.k = kf['k'][frame_idx::time_step].reshape(-1, 1024).astype(np.float32)

    def __getitem__(self, item):
        return self.k[item]

    def __len__(self):
        return self.k.shape[0]


class fmri_k_dataset(Dataset):
    def __init__(self, fmri_file, k_file, fmri_key, frame_idx, time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        self.resp = fmrif[fmri_key][:]
        kf = h5py.File(k_file, 'r')
        self.k = kf['k'][frame_idx::time_step].reshape(-1, 1024).astype(np.float32)
        assert self.resp.shape[-1] == self.k.shape[0]

    def __getitem__(self, item):
        fmri = self.resp[:, item]
        vector = self.k[item] / 512  # , self.row
        return fmri, vector

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


def get_vim2_fmri_mean_std(voxel_train_file, dt_key):
    with h5py.File(voxel_train_file, 'r') as vf:
        r = vf[dt_key][:].flatten()  # todo 需要test数据的mean，std vf[dt_key][roi_idx, :].flatten()
        r = np.nan_to_num(r)
        mean = np.mean(r)
        std = np.std(r)
    return mean, std

# def get_latent_mean_std(latent_train_file):
#     with h5py.File(latent_train_file, 'r') as f:
#         latent = f['latent'][:].flatten()
#         latent = np.mean
