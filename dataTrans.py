import h5py
import tensorflow as tf
from MyDataset import vqvae_ze_dataset, Stimuli_Dataset, vqvae_one_frame_k_dataset, vim1_stimuli_dataset, \
    vim1_blur_stimuli_dataset
from torch.utils.data import DataLoader
from model import VQVAE, _imagenet_arch
import numpy as np
import os
import json
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def extract_k(file, Dataset=vqvae_ze_dataset):
    ze_dataset = Dataset(file)
    ze_dataloader = DataLoader(ze_dataset, batch_size=10, shuffle=False, num_workers=1)
    sess = tf.Session()
    with sess.as_default():
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/ze_embeds_from_vqvae_st.hdf5", 'r') as zf:
            e = zf['embeds'][:]
            # ze = zf['ze']
            with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st.hdf5", 'w') as kf:
                k_data = kf.create_dataset('k', shape=(len(ze_dataset), 32, 32))
                # e = embeds
                begin_idx = 0
                for step, ze_batch in enumerate(ze_dataloader):
                    # for i in range(ze.shape[0]):
                    end_idx = begin_idx + ze_batch.shape[0]
                    t = tf.expand_dims(ze_batch, axis=-2)
                    nt = tf.norm(t - e, axis=-1)
                    k = tf.argmin(nt, axis=-1)  # -> [latent_h,latent_w]
                    k_data[begin_idx:end_idx] = k.eval()
                    begin_idx = end_idx

                    if step % 100 == 0:
                        print(step)


def concatenate_ze(dt_key, frame_idx, n_lantent=1024):
    latents = []
    for i in range(n_lantent):
        with h5py.File(
                "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_1/{}/frame_{}/subject_1_frame_{}_ze_latent_{}.hdf5".format(
                    dt_key, frame_idx, frame_idx, i), 'r') as f:  # todo subject frame format
            latents.append(f['latent'][:])  # shape = (540,128) (108000,128)

    with h5py.File(
            "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_1/{}/frame_{}/subject_1_frame_{}_ze_latent_all.hdf5".format(
                dt_key, frame_idx, frame_idx),
            'w') as sf:
        all_latent_dataset = sf.create_dataset('latent', shape=(latents[0].shape[0], n_lantent, latents[0].shape[-1]))

        for j in range(n_lantent):
            all_latent_dataset[:, j, :] = latents[j]  # todo 检查这样对不对
            print(j)


def concatenate_fmaps(dt_key, frame_idx, rootDir, subject=1, n_fmaps=128):
    fmaps = []
    for i in range(n_fmaps):
        # /data1/home/guangjie/Data/vim-2-gallant/
        with h5py.File(os.path.join(rootDir,
                                    "regressed_zq_of_vqvae_by_feature_map/subject_{}/{}/frame_{}/subject_{}_frame_{}_zq_fmap_{}.hdf5".format(
                                        subject, dt_key, frame_idx, subject, frame_idx, i)), 'r') as rf:
            fmap = rf['fmap'][:]
            fmaps.append(fmap[:, :, np.newaxis])
    fmaps = np.concatenate(fmaps, axis=2)
    with h5py.File(os.path.join(rootDir,
                                "regressed_zq_of_vqvae_by_feature_map/subject_{}/{}/frame_{}/subject_{}_frame_{}_ze_fmap_all.hdf5".format(
                                    subject, dt_key, frame_idx, subject, frame_idx)), 'w') as wf:
        wf.create_dataset('latent', data=fmaps)


def rec_a_frame_img_from_ze(latentPath, savePath):
    os.makedirs(savePath[:savePath.rfind('/')], exist_ok=True)
    # os.makedirs(latentRootDir, exist_ok=True)
    # save_dir = os.path.join(saveRootDir, "subject{}/{}/frame_{}".format(
    #     subject, dt_key, frame_idx))
    # os.makedirs(save_dir, exist_ok=True)
    MODEL, K, D = ('models/imagenet/last.ckpt', 512, 128)
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        net = VQVAE(None, None, 0.25, x, K, D, _imagenet_arch, params, False)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess, MODEL)

    dataset = vqvae_ze_dataset(latentPath)
    # dataset = vqvae_ze_dataset(
    #     os.path.join(latentRootDir, "subject_{}/{}/frame_{}/subject_{}_frame_{}_ze_{}_all_wd003.hdf5".format(
    #         subject, dt_key, frame_idx, subject, frame_idx, postfix)))
    # dataset = vqvae_zq_dataset("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_sv.hdf5")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

    with h5py.File(savePath, 'w') as sf:
        rec_dataset = sf.create_dataset('rec', shape=(len(dataset), 128, 128, 3), dtype=np.uint8, chunks=True)
        begin_idx = 0
        for step, data in enumerate(dataloader):
            rec = sess.run(net.p_x_z, feed_dict={net.z_e: data})  # todo z_e z_q 直接喂给zq的话在验证集效果更差。。。
            rec = (rec * 255.0).astype(np.uint8)
            end_idx = begin_idx + len(rec)
            rec_dataset[begin_idx:end_idx] = rec
            begin_idx = end_idx
            print(step)


def rec_a_frame_img_from_zq(dt_key, frame_idx, subject=1):
    save_dir = "/data1/home/guangjie/Data/vim-2-gallant/rec_by_zq_of_vqvae/subject1/{}/frame_{}".format(
        dt_key, frame_idx)
    os.makedirs(save_dir, exist_ok=True)
    MODEL, K, D = ('models/imagenet/last.ckpt', 512, 128)
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        net = VQVAE(None, None, 0.25, x, K, D, _imagenet_arch, params, False)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess, MODEL)

    dataset = vqvae_ze_dataset(
        "/data1/home/guangjie/Data/vim-2-gallant/regressed_k_of_vqvae/subject_{}/{}/frame_{}_regressed_zq.hdf5".format(
            subject, dt_key, frame_idx))  # todo latent?
    # dataset = vqvae_zq_dataset("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_sv.hdf5")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

    with h5py.File(os.path.join(save_dir, "subject_{}_{}_frame_{}_rec_of_k.hdf5".format(subject, dt_key, frame_idx)),
                   'w') as sf:
        rec_dataset = sf.create_dataset('rec', shape=(len(dataset), 128, 128, 3), dtype=np.uint8)
        begin_idx = 0
        for step, data in enumerate(dataloader):
            rec = sess.run(net.p_x_z, feed_dict={net.z_q: data})  # todo z_e z_q 直接喂给zq的话在验证集效果更差。。。
            rec = (rec * 255.0).astype(np.uint8)
            end_idx = begin_idx + len(rec)
            rec_dataset[begin_idx:end_idx] = rec
            begin_idx = end_idx
            print(step)


def rec_a_frame_img_from_k(save_path, k_file):
    # save_dir = "/data1/home/guangjie/Data/vim-2-gallant/rec_by_ze_of_vqvae_use_k/subject1/{}/frame_{}".format(
    #     dt_key, frame_idx)
    # save_dir = os.path.join(save_dir, "subject{}/{}/frame_{}".format(subject, dt_key, frame_idx))
    # os.makedirs(save_dir, exist_ok=True)
    MODEL, K, D = ('models/imagenet/last.ckpt', 512, 128)
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        net = VQVAE(None, None, 0.25, x, K, D, _imagenet_arch, params, False)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess, MODEL)

    # dataset = vqvae_ze_dataset(
    #     "/data1/home/guangjie/Data/vim-2-gallant/regressed_k_of_vqvae/subject_{}/{}/frame_{}_regressed_zq.hdf5".format(
    #         subject, dt_key, frame_idx))  # todo latent?
    # dataset = vqvae_zq_dataset("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_sv.hdf5")
    dataset = vqvae_one_frame_k_dataset(kfile=k_file)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

    with h5py.File(save_path, 'w') as sf:
        rec_dataset = sf.create_dataset('rec', shape=(len(dataset), 128, 128, 3), dtype=np.uint8)
        begin_idx = 0
        for step, data in enumerate(dataloader):
            rec = sess.run(net.p_x_z, feed_dict={net.k: data})  # todo z_e z_q 直接喂给zq的话在验证集效果更差。。。
            rec = (rec * 255.0).astype(np.uint8)
            end_idx = begin_idx + len(rec)
            rec_dataset[begin_idx:end_idx] = rec
            begin_idx = end_idx
            print(step)


def extract_zq_from_vqvae(dt_key):
    MODEL, K, D = ('models/imagenet/last.ckpt', 512, 128)
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        net = VQVAE(None, None, 0.25, x, K, D, _imagenet_arch, params, False)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess, MODEL)

    dtKey = 'stimTrn' if dt_key == 'st' else 'stimVal'
    dataset = vim1_blur_stimuli_dataset("/data1/home/guangjie/Data/vim-1/Stimuli.mat", dtKey, 3)
    # dataset = Stimuli_Dataset("/data1/home/guangjie/Data/vim-2-gallant/orig/Stimuli.mat", dt_key)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)

    with h5py.File(
            "/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/ze_from_vqvae_{}.hdf5".format(dt_key),
            'w') as sf:
        ze_dataset = sf.create_dataset('latent', shape=(len(dataset), 32, 32, 128))
        begin_idx = 0
        for step, data in enumerate(dataloader):
            ze = sess.run(net.z_e, feed_dict={x: data})
            end_idx = begin_idx + len(ze)
            ze_dataset[begin_idx:end_idx] = ze
            begin_idx = end_idx
            print(step)


def extract_k_rec_from_vqvae(dt_key):
    MODEL, K, D = ('models/imagenet/last.ckpt', 512, 128)
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        net = VQVAE(None, None, 0.25, x, K, D, _imagenet_arch, params, False)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess, MODEL)

    # dataset = Stimuli_Dataset(
    #     "/data1/home/guangjie/Data/purdue/exprimentData/Stimuli/Stimuli_{}_frame_{}.hdf5".format(
    #         'train' if dt_key == 'st' else 'test', frame_idx), dt_key, transpose=False)
    dtKey = 'stimTrn' if dt_key == 'st' else 'stimVal'
    dataset = vim1_blur_stimuli_dataset("/data1/home/guangjie/Data/vim-1/Stimuli.hdf5", dtKey, 3)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)
    os.makedirs('/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae', exist_ok=True)
    with h5py.File(
            "/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/rec_from_vqvae_{}.hdf5".format(
                dt_key), 'w') as recf:
        rec_dataset = recf.create_dataset('rec', shape=(len(dataset), 128, 128, 3))
        with h5py.File(
                "/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/k_from_vqvae_{}.hdf5".format(
                    dt_key), 'w') as kf:
            k_dataset = kf.create_dataset('k', shape=(len(dataset), 32, 32))
            begin_idx = 0
            for step, data in enumerate(dataloader):
                k, rec = sess.run((net.k, net.p_x_z), feed_dict={x: data})
                end_idx = begin_idx + len(rec)
                rec_dataset[begin_idx:end_idx] = rec
                k_dataset[begin_idx:end_idx] = k
                begin_idx = end_idx
                print(step)


def split_rt_to_train_test():
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt.hdf5", 'r') as rtf:
        rt = rtf['rt']
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_train.hdf5",
                       'w') as trainf:
            trainf.create_dataset('rt', data=rt[:, :6000])
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_test.hdf5",
                       'w') as testf:
            testf.create_dataset('rt', data=rt[:, 6000:])


def split_st_to_train_test():
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/orig/Stimuli.mat", 'r') as rtf:
        st = rtf['st']
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/Stimuli_st_train.hdf5", 'w') as trainf:
            trainf.create_dataset('st', data=st[:90000])
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/Stimuli_st_test.hdf5", 'w') as testf:
            testf.create_dataset('st', data=st[90000:])


def split_zq_to_train_test():
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_st.hdf5", 'r') as zqf:
        zq = zqf['zq']
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_st_train.hdf5", 'w') as trainf:
            trainf.create_dataset('zq', data=zq[:90000])
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_st_test.hdf5", 'w') as testf:
            testf.create_dataset('zq', data=zq[90000:])


def build_zq_from_k_embeds(k_file, embeds_file, save_file):
    with h5py.File(embeds_file, 'r') as ebdf:
        embeds = ebdf['embeds'][:]
        with h5py.File(k_file, 'r') as kf:
            kdata = kf['k']
            with h5py.File(save_file, 'w') as sf:
                zqdata = sf.create_dataset('latent', shape=(kdata.shape[0], 32, 32, 128))
                for i, k in enumerate(kdata):
                    zq = embeds[k]
                    zqdata[i] = zq
                    print(i)


def use_weighted_norm2_to_find_k(weight_file, fmap_file, save_file):
    with open(weight_file, 'r') as fp:
        select = list(json.load(fp))
        weight = np.zeros((128))
        weight[select] = 1
        # weight = torch.as_tensor(json.load(fp)['weight'], dtype=torch.float).cuda()
        weight = torch.as_tensor(weight, dtype=torch.float).cuda()
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
                       'r') as ebdf:
            embeds = torch.as_tensor(ebdf['embeds'][:]).cuda()
            with h5py.File(fmap_file, 'r') as fmapf:
                latent_data = fmapf['latent']
                ks = []
                for i in range(latent_data.shape[0]):
                    latent = torch.as_tensor(latent_data[i][:, :, np.newaxis, :]).cuda()
                    element_wise_diff = (latent - embeds) ** 2
                    element_wise_weighted_diff = element_wise_diff * weight
                    weight_diff = torch.sum(element_wise_weighted_diff, dim=3)
                    k = torch.argmin(weight_diff, dim=2)  # .reshape(32, 32)
                    ks.append(k)
                    print(i)
                ks = torch.stack(ks, dim=0)
                with h5py.File(save_file, 'w') as sf:
                    sf.create_dataset('k', data=ks.cpu().numpy())
                    print('end')


if __name__ == '__main__':
    # extract_k(file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/ze_embeds_from_vqvae_st.hdf5")
    # concatenate_fmaps('rv', frame_idx=0, rootDir="/data1/home/guangjie/Data/purdue/exprimentData/")
    # concatenate_ze('rt', frame_idx=0)
    # rec_a_frame_img_from_ze('rv', frame_idx=0, rootDir="/data1/home/guangjie/Data/purdue/exprimentData/")
    # rec_a_frame_img_from_ze('rv', frame_idx=0,
    #                         latentRootDir="/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_map_slice/",
    #                         saveRootDir="/data1/home/guangjie/Data/vim-2-gallant/rec_by_ze_of_vqvae_slice")
    #####################################################################################################################
    rec_a_frame_img_from_ze(
        latentPath="/data1/home/guangjie/Data/vim2/regressed_feature_map/subject_1/rt/subject_1_rt_roi_v1lhv1rhv2lhv2rhv3alhv3arhv3blhv3brhv3lhv3rhv4lhv4rh_regressed_fmap_all_wd_0.02_normalizedFmap_ss_mm_18x18_blur5_resize_32x32_inverse_fit.hdf5",
        savePath="/data1/home/guangjie/Data/vim2/rec_by_ze_of_vqvae_fmap/subject1/rt/subject_1_rt_roi_v1lhv1rhv2lhv2rhv3alhv3arhv3blhv3brhv3lhv3rhv4lhv4rh_regressed_fmap_all_wd_0.02_normalizedFmap_ss_mm_18x18_blur5_resize_32x32_inverse_fit_rec.hdf5")
    #####################################################################################################################
    # extract_k_rec_from_vqvae('sv')
    ######################################################################################################################################
    # rec_a_frame_img_from_k(save_path="/data1/home/guangjie/Data/vim1/rec_by_k/subject_1_dataTrnS1_roi_12_wd_0.005_td_0.001_rec.hdf5",
    #                        k_file="/data1/home/guangjie/Data/vim1/selected_k_from_regressed_feature_map/subject_1_dataTrnS1_roi_12_wd_0.005_td_0.001_k.hdf5")
    #######################################################################################################################################
    # "/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_latent/"
    # "/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_map/"
    # "/data1/home/guangjie/Data/vim-2-gallant/rec_by_ze_of_vqvae_fmap"
    # rec_a_frame_img_from_zq('rt', frame_idx=0, subject=1)
    # extract_zq_from_vqvae('st')
    # split_rt_to_train_test()
    # split_st_to_train_test()
    # split_zq_to_train_test()
    # build_zq_from_k_embeds(
    #     k_file="/data1/home/guangjie/Data/vim-2-gallant/regressed_k_of_vqvae/subject_1/rt/frame_0_regressed_k.hdf5",
    #     embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
    #     save_file="/data1/home/guangjie/Data/vim-2-gallant/regressed_k_of_vqvae/subject_1/rt/frame_0_regressed_zq.hdf5")

    # use_weighted_norm2_to_find_k(
    #     weight_file="/data1/home/guangjie/Project/python/tf-vqvae/vim2_select_voxel_manual.json",
    #     fmap_file="/data1/home/guangjie/Data/vim2/regressed_feature_map/subject_1/rt/subject_1_rt_roi_v1lhv1rhv2lhv2rhv3alhv3arhv3blhv3brhv3lhv3rhv4lhv4rh_regressed_fmap_all_wd_0.03_normalizedFmap_no_mm_18x18_blur3_nolinearConv_Adam_bn_inverse_32x32.hdf5",
    #     save_file="/data1/home/guangjie/Data/vim2/weighted_k/subject_1_rt_roi_v1lhv1rhv2lhv2rhv3alhv3arhv3blhv3brhv3lhv3rhv4lhv4rh_wd_0.03_normalizedFmap_no_mm_18x18_blur3_nolinearConv_Adam_bn_inverse_32x32_weighted_k.hdf5")

    # rec_a_frame_img_from_k(save_path="/data1/home/guangjie/Data/vim2/rec_by_k_of_vqvae/subject_1_rt_roi_v1lhv1rhv2lhv2rhv3alhv3arhv3blhv3brhv3lhv3rhv4lhv4rh_wd_0.03_normalizedFmap_no_mm_18x18_blur3_nolinearConv_Adam_bn_inverse_32x32_weighted_k_rec.hdf5",
    #                        k_file="/data1/home/guangjie/Data/vim2/weighted_k/subject_1_rt_roi_v1lhv1rhv2lhv2rhv3alhv3arhv3blhv3brhv3lhv3rhv4lhv4rh_wd_0.03_normalizedFmap_no_mm_18x18_blur3_nolinearConv_Adam_bn_inverse_32x32_weighted_k.hdf5")
    # extract_k_rec_from_vqvae('st')
    print('end')
