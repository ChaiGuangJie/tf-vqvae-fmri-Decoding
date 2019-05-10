import h5py
import tensorflow as tf
from MyDataset import vqvae_ze_dataset, vqvae_ze_dataset
from torch.utils.data import DataLoader
from model import VQVAE, _imagenet_arch
import numpy as np
import os


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


def concatenate_ze(dt_key, n_lantent=1024):
    latents = []
    for i in range(n_lantent):
        with h5py.File(
                "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_1/{}/frame_0/subject_1_frame_0_ze_latent_{}.hdf5".format(
                    dt_key, i), 'r') as f:
            latents.append(f['latent'][:])  # shape = (540,128) (108000,128)

    with h5py.File(
            "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_1/{}/frame_0/subject_1_frame_0_ze_latent_all.hdf5".format(
                dt_key),
            'w') as sf:
        all_latent_dataset = sf.create_dataset('latent', shape=(latents[0].shape[0], n_lantent, latents[0].shape[-1]))

        for j in range(n_lantent):
            all_latent_dataset[:, j, :] = latents[j]  # todo 检查这样对不对
            print(j)


def rec_a_frame_img_from_ze(dt_key):
    save_dir = "/data1/home/guangjie/Data/vim-2-gallant/rec_by_ze_of_vqvae/subject1/{}/frame_0".format(
        dt_key)
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
        "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_1/{}/frame_0/subject_1_frame_0_ze_latent_all.hdf5".format(
            dt_key))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)

    with h5py.File(os.path.join(save_dir, "subject_1_{}_frame_0_rec.hdf5".format(dt_key)), 'w') as sf:
        rec_dataset = sf.create_dataset('rec', shape=(len(dataset), 128, 128, 3), dtype=np.uint8)
        begin_idx = 0
        for step, data in enumerate(dataloader):
            rec = sess.run(net.p_x_z, feed_dict={net.z_e: data})
            rec = (rec * 255.0).astype(np.uint8)
            end_idx = begin_idx + len(rec)
            rec_dataset[begin_idx:end_idx] = rec
            begin_idx = end_idx
            print(step)


if __name__ == '__main__':
    # extract_k(file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/ze_embeds_from_vqvae_st.hdf5")
    # concatenate_ze('rva0')
    rec_a_frame_img_from_ze('rva0')
