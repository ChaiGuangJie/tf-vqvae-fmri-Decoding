import h5py
import numpy as np
import time
from visdom import Visdom


def get_frames_data(file, dt_key, count):
    with h5py.File(file, 'r') as f:
        data = f[dt_key][:count]
    return data


# 可视化原始图像与重建图像的第frame_idx帧
def show_rec_one_frame(viz, dt_key, frame_idx, count):
    ori_data = get_frames_data("/data1/home/guangjie/Data/purdue/exprimentData/Stimuli/Stimuli_{}_frame_{}.hdf5".format(
        'train' if dt_key == 'st' else 'test', frame_idx), dt_key, count)
    # rec_data = get_frames_data(
    #     "/data1/home/guangjie/Data/purdue/exprimentData/extract_from_vqvae/rec_from_vqvae_{}_frame_{}.hdf5".format(
    #         dt_key, frame_idx), 'rec', count)
    rec_data = get_frames_data(
        "/data1/home/guangjie/Data/purdue/exprimentData/rec_by_ze_of_vqvae/subject1/rt/frame_0/subject_1_rt_frame_0_rec.hdf5",
        'rec', count)
    win = viz.images(np.random.randn(2, 3, 128, 128), opts={'title': 'ori vs rec'})
    for i, (ori, rec) in enumerate(zip(ori_data, rec_data)):
        ori = ori.transpose((2, 0, 1)).astype(np.uint8)  # .transpose((0, 2, 1)) [:, :, ::-1]
        rec = (rec.transpose((2, 0, 1)) * 255.0).astype(np.uint8)  # / 255.0  # [:, :, ::-1]
        grid = np.array([ori, rec])
        viz.images(grid, win=win, opts={'title': 'ori vs rec'})
        print(i)
        time.sleep(0.5)


if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='purdue data visualize')
    assert viz.check_connection(timeout_seconds=3)
    show_rec_one_frame(viz, 'sv', frame_idx=0, count=100)
