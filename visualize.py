import h5py
import tensorflow as tf
import numpy as np
from visdom import Visdom
import time


def show_z_flow(viz, time_step=15):
    # 初始化15个窗口
    win_list = []
    for j in range(time_step):
        win = viz.heatmap(X=[np.arange(1, 512)], opts=dict(width=1680, height=160, title='win_{} heatmap'.format(j)))
        win_list.append(win)
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st.hdf5", 'r') as kf:
        k = kf['k']
        idx = np.arange(1, 1024)
        for i in range(k.shape[0]):
            k_item = k[i].flatten()
            if i == 0:
                idx = np.argsort(k_item)
            sorted_k_item = k_item[idx]
            win_list[i % time_step] = viz.heatmap(X=[sorted_k_item], win=win_list[i % time_step])
            idx = np.argsort(k_item)
            print(i)


def show_z_flow_and_rec():
    # 初始化15个heatmap窗口 + 一个用于显示重建图像/原图像的窗口
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/rec_from_vqvae_st.hdf5", 'r') as recf:
        rec = recf['rec']
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st.hdf5", 'r') as kf:
            k = kf['k']
            for k_item in range(k.shape[0]):
                # draw
                pass
            print(k.shape)
            # z_q = tf.gather(embeds,k)


def testHeatmap(viz):
    viz.heatmap(
        # X=np.random.randint(low=1, high=512, size=(1, 1024), dtype=np.uint16),
        X=[np.arange(1, 512)],
        # np.outer(np.arange(1, 6), np.arange(1, 11)),
        opts=dict(
            title='heatmap',
            width=1680,
            height=180
            # columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
            # rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
            # colormap='Electric',
        )
    )


# 可视化原始图像与重建图像的第frame_idx帧
def show_rec_one_frame(viz, dt_key, frame_idx=0, time_step=15):
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/orig/Stimuli.mat", 'r') as orif:
        ori_data = orif['st' if dt_key == 'rt' else 'sv']
        with h5py.File(
                "/data1/home/guangjie/Data/vim-2-gallant/rec_by_ze_of_vqvae/subject1/{}/frame_0/subject_1_{}_frame_0_rec.hdf5".format(
                    dt_key, dt_key),
                'r') as recf:
            rec_data = recf['rec']

            win = viz.images(np.random.randn(2, 3, 128, 128), opts={'title': 'ori vs rec'})
            for i in range(rec_data.shape[0]):
                ori = ori_data[frame_idx + i * time_step].transpose((0, 2, 1))
                rec = rec_data[i].transpose((2, 0, 1))
                grid = np.array([ori, rec])
                viz.images(grid, win=win, opts={'title': 'ori vs rec'})
                print(i)
                time.sleep(1)


if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='visualize')
    assert viz.check_connection(timeout_seconds=3)

    # testHeatmap(viz)
    # show_z_flow(viz)
    show_rec_one_frame(viz, 'rva0')
    print('end')
    viz.close()