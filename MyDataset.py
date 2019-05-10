import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np


class vqvae_ze_dataset(Dataset):
    def __init__(self, file):
        zef = h5py.File(file, 'r')
        self.latent = zef['latent']  # shape = (540,1024,128)

    def __getitem__(self, item):
        return self.latent[item].reshape((32, 32, 128))

    def __len__(self):
        return self.latent.shape[0]


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
    def __init__(self, fmri_file, zq_file, mean, std, fmri_key='rt', frame_idx=0, latent_idx=0, time_step=15):
        fmrif = h5py.File(fmri_file, 'r')
        self.response = (fmrif[fmri_key][:] - mean) / std  # todo 这样可以？
        zqf = h5py.File(zq_file, 'r')
        self.zq = zqf['zq']  # shape=(108000,32,32,128)

        self.frame_idx = frame_idx
        self.latent_idx = latent_idx
        self.time_step = time_step
        self.mean = mean
        self.std = std

    def __getitem__(self, item):
        fmri = self.response[:, item]
        # fmri = np.nan_to_num(fmri)
        # fmri = (fmri - self.mean) / self.std

        vector = self.zq[self.frame_idx + item * self.time_step].reshape(1024, 128)

        return fmri, vector[self.latent_idx]

    def __len__(self):
        return self.response.shape[-1]


class fmri_dataset(Dataset):
    def __init__(self, fmri_file, mean, std, dt_key):  # todo fmri file 的 mean，std 必须对应同一个被试
        f = h5py.File(fmri_file, 'r')
        self.resp = (f[dt_key] - mean) / std

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
