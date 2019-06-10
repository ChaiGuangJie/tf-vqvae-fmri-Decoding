import h5py
import torch
# import slir
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    subject = 3
    frame_idx = 1
    time_step = 15
    # model = linear_model.LinearRegression()
    model = linear_model.Ridge(alpha=1)
    # model = slir.SparseLinearRegression(n_iter=1)
    ss = StandardScaler()

    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject3_v1234_rt_train.hdf5",
                   'r') as fmrif:
        rt_train = fmrif['rt'][:].transpose()
    rt_train = ss.fit_transform(rt_train)

    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject3_v1234_rt_test.hdf5",
                   'r') as fmrif:
        rt_test = fmrif['rt'][:].transpose()
    rt_test = ss.transform(rt_test)

    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_st.hdf5", 'r') as zqf:
        zq = zqf['zq']
        with h5py.File(
                "/data1/home/guangjie/Data/vim-2-gallant/regressed_feature_maps_by_sklearn/subject_{}/rv/frame_{}/predict_feature_maps_ss_ridge_1.hdf5".format(
                    subject, frame_idx), 'w') as rvf:
            rvLatent = rvf.create_dataset('latent', shape=(200, 32, 32, 128))
            with h5py.File(
                    "/data1/home/guangjie/Data/vim-2-gallant/regressed_feature_maps_by_sklearn/subject_{}/rt/frame_{}/predict_feature_maps_ss_ridge_1.hdf5".format(
                        subject, frame_idx), 'w') as rtf:
                rtLatent = rtf.create_dataset('latent', shape=(7000, 32, 32, 128))

                st_idx = np.sort(np.array(range(frame_idx, 108000, time_step)))
                st_per_frame_idx = set(range(7200))
                st_test_idx = set(range(0, 7200, 36))
                st_train_idx = st_per_frame_idx - st_test_idx
                st_train_idx = np.sort(np.array(list(st_train_idx)))
                st_test_idx = np.sort(np.array(list(st_test_idx)))
                # todo 检查逻辑
                for i in range(128):
                    print('ridge regression start ', i)
                    fpoint = zq[st_idx[st_train_idx], :, :, i].reshape(-1, 1024)  # shape = (7000,1024)
                    print('start fit')
                    model.fit(rt_train, fpoint)
                    print('start predict')
                    rt_pred = model.predict(rt_train)
                    rtLatent[:, :, :, i] = rt_pred.reshape(7000, 32, 32)

                    rv_pred = model.predict(rt_test)
                    rvLatent[:, :, :, i] = rv_pred.reshape(200, 32, 32)
                    print('end ', i)
