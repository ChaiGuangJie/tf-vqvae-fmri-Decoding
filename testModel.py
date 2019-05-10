from six.moves import xrange
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import h5py
from model import VQVAE, _imagenet_arch
from torch.utils.data import Dataset, DataLoader


class Vim2_Stimuli_Dataset(Dataset):
    def __init__(self, file, dt_key):
        f = h5py.File(file, 'r')
        self.dt = f[dt_key]  # shape = (108000,3,128,128)

    def __getitem__(self, item):
        data = self.dt[item].transpose((2, 1, 0)) / 255.0
        return data

    def __len__(self):
        return self.dt.shape[0]


# sys.path.append('../models/research/slim')
# from datasets import imagenet
#
# slim = tf.contrib.slim

# train_dataset = imagenet.get_split('train',"/data1/home/guangjie/Data/Imagenet/ILSVRC2012_img_train_128/")
# valid_dataset = imagenet.get_split('validation',"/data1/home/guangjie/Data/Imagenet/ILSVRC2012_img_train_128/")
# def _build_batch(dataset,batch_size,num_threads):
#     with tf.device('/cpu'):
#         provider = slim.dataset_data_provider.DatasetDataProvider(
#             dataset,
#             num_readers=num_threads,
#             common_queue_capacity=20*batch_size,
#             common_queue_min=10*batch_size,
#             shuffle=True)
#         image,label = provider.get(['image','label'])
#         pp_image = tf.image.resize_images(image,[128,128]) / 255.0
#
#         images,labels = tf.train.batch(
#             [pp_image,label],
#             batch_size=batch_size,
#             num_threads=num_threads,
#             capacity=5*batch_size)
#         return images, labels
# train_ims,train_labels = _build_batch(train_dataset,16,4)
# valid_ims,valid_labels = _build_batch(valid_dataset,16,1)

MODEL, K, D = ('models/imagenet/last.ckpt', 512, 128)

# with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/orig/Stimuli.mat", 'r') as f:
# st0 = f['st'][:500:50, :, :, :].transpose((0, 3, 2, 1)) / 255.0  # shape = (108000,3,128,128)
# st1 = f['st'][500:1000:50, :, :, :].transpose((0, 3, 2, 1)) / 255.0
# st0 = st0[np.newaxis, :]
# st = f['st'][0]
# extend_st = st[np.newaxis, :]

# slices = tf.data.Dataset.from_tensor_slices(extend_st)
# next_item = slices.make_one_shot_iterator().get_next() #todo

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


def draw(images):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(20, 20))
    for n, image in enumerate(images):
        a = fig.add_subplot(2, 5, n + 1)
        a.imshow((image * 255.0).astype(np.uint8))
        # a.imshow(image)
        a.axis('off')
        a.set_aspect('equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.close()


dataset = Vim2_Stimuli_Dataset("/data1/home/guangjie/Data/vim-2-gallant/orig/Stimuli.mat", 'sv')
dataLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

for step, data in enumerate(dataLoader):
    ze, zq, rec = sess.run((net.z_e, net.z_q, net.p_x_z), feed_dict={x: data})
    print(step)

# valid_origin = sess.run(next_item)
# z_q = sess.run(net.z_q, feed_dict={x: valid_origin})
# print('z_q:', z_q)
# z_q = np.random.normal(-1.0, 1.0, (10, 32, 32, 128))
# valid_recons0, embeds0 = sess.run((net.p_x_z, net.embeds), feed_dict={net.z_q: z_q})
#
# z_q = np.random.normal(-1.0, 1.0, (10, 32, 32, 128))
# valid_recons1, embeds1 = sess.run((net.p_x_z, net.embeds), feed_dict={net.z_q: z_q})
# print(embeds0 == embeds1)
# valid_recons = sess.run(net.p_x_z, feed_dict={x: valid_origin})
# draw(valid_origin)
# draw(valid_recons)
print('end')
