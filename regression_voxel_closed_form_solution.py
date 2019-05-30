import h5py
import torch

if __name__ == '__main__':
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_train.hdf5",'r') as fmrif:
        rt = fmrif['rt'][:]

    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_st_train.hdf5",'r') as zqf:
        zq = zqf['latent']