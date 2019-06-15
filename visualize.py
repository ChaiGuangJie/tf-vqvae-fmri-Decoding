import h5py
import tensorflow as tf
import numpy as np
from visdom import Visdom
import time
import json
import os
import torch
import scipy.io as sio
from MyDataset import vim1_stimuli_dataset, vim1_fmri_transpose_dataset
import cv2


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


def show_vim1_ori_and_rec(viz, dt_key, rec_file, start, end):
    ori_file = "/data1/home/guangjie/Data/vim-1/Stimuli.mat"
    ori_data = vim1_stimuli_dataset(ori_file, dt_key, 3)
    with h5py.File(rec_file, 'r') as recf:
        rec_data = recf['rec']
        win = viz.images(np.random.randn(2, 3, 128, 128), opts={'title': 'ori vs rec'})
        for i in np.arange(start, end):
            ori = ori_data[i].transpose((2, 0, 1))
            rec = rec_data[i].transpose((2, 0, 1)) / 255.0
            # rec = np.repeat(rec_data[i][:, :, 0][:, :, np.newaxis], 3, axis=-1).transpose((2, 0, 1)) / 255.0
            grid = np.array([ori, rec])
            viz.images(grid, win=win, opts={'title': 'ori vs rec'})
            print(i)
            time.sleep(1.5)


def show_rec_one_frame_uniform_sample(viz, dt_key, frame_idx, rec_path, sample_interval, time_step=15):
    # with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/orig/Stimuli.mat", 'r') as orif: #
    #     ori_data = orif['st' if dt_key == 'rt' else 'sv']
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/orig/Stimuli.mat", 'r') as orif:  #
        ori_data = orif['st']
        with h5py.File(rec_path, 'r') as recf:
            rec_data = recf['rec']

            win = viz.images(np.random.randn(2, 3, 128, 128), opts={'title': 'ori vs rec'})
            st_idx = np.sort(np.array(range(frame_idx, 108000, time_step)))
            st_per_frame_idx = set(range(7200))
            st_test_idx = set(range(0, 7200, sample_interval))
            st_train_idx = st_per_frame_idx - st_test_idx
            st_train_idx = np.sort(np.array(list(st_train_idx)))
            st_test_idx = np.sort(np.array(list(st_test_idx)))

            for i in range(0, 200):
                if dt_key == 'rt':
                    ori = ori_data[st_idx[st_train_idx[i]]].transpose((0, 2, 1))
                elif dt_key == 'rv':
                    ori = ori_data[st_idx[st_test_idx[i]]].transpose((0, 2, 1))
                rec = rec_data[i].transpose((2, 0, 1))
                grid = np.array([ori, rec])
                viz.images(grid, win=win, opts={'title': 'ori vs rec'})
                print(i)
                time.sleep(0.8)


def show_gen_k_heatmap(dt_key, frame_idx=0, time_step=15):
    with h5py.File(
            "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_1/{}/frame_0/subject_1_frame_0_ze_latent_all.hdf5".format(
                dt_key), 'r') as zef:
        ze_data = zef['latent']
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
                       'r') as ebdf:
            embeds = ebdf['embeds'][:]
            with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_{}.hdf5".format(
                    'st' if dt_key == 'rt' else 'sv'), 'r') as kf:
                true_k_data = kf['k']
                for i in range(10):
                    ze = ze_data[i].reshape((32, 32, 1, 128))
                    t = np.linalg.norm(ze - embeds, axis=-1)
                    k = np.argmin(t, -1).flatten()
                    true_k = true_k_data[frame_idx + i * time_step].flatten()
                    diff = abs(true_k - k)
                    print(np.mean(diff))


def test_window_callback(viz):
    txt = 'This is a write demo notepad. Type below. Delete clears text:<br>'
    callback_text_window = viz.text(txt)
    viz.a = 1

    def type_callback(event):
        if event['event_type'] == 'KeyPress':
            viz.a += 1
            curr_txt = str(viz.a)
            # curr_txt = event['pane_data']['content']
            viz.line(Y=np.zeros((7200,)), X=np.arange(0, 7200),
                     opts=dict(width=1680, height=300, title='voxel Number'))
            if event['key'] == 'Enter':
                curr_txt += '<br>'
            elif event['key'] == 'Backspace':
                curr_txt = curr_txt[:-1]
            elif event['key'] == 'Delete':
                curr_txt = txt
            elif len(event['key']) == 1:
                curr_txt += event['key']
            viz.text(curr_txt, win=callback_text_window)

    viz.register_event_handler(type_callback, callback_text_window)


def visualize_and_select_voxels(viz, subject):
    # voxel_win = viz.line(Y=np.zeros((7200,)), X=np.arange(0, 7200),
    #                      opts=dict(width=1680, height=300))
    with h5py.File(
            "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_ips_rt_rv_rva0.hdf5".format(
                subject),
            'r') as rtf:
        viz.rt = rtf['rt'][:]
        viz.rv = rtf['rva0'][:]

    viz.current_voxel = 0
    viz.voxel_length = viz.rt.shape[0]
    viz.select_voxel = []
    voxel_win = viz.line(Y=np.concatenate((viz.rt[viz.current_voxel], viz.rv[viz.current_voxel])), X=np.arange(0, 7740),
                         opts=dict(width=1600, height=300, title='voxel'))
    callbackwin = viz.text('Voxel Number {}'.format(viz.current_voxel))

    def type_callback(event):
        # global rt, current_voxel, voxel_win, voxel_length, viz, select_voxel
        if event['event_type'] == 'KeyPress':
            if event['key'] == 'ArrowDown' or event['key'] == 'ArrowUp' or event['key'] == 'Delete':
                if event['key'] == 'ArrowDown':
                    viz.select_voxel.append(viz.current_voxel)
                    viz.current_voxel += 1
                elif event['key'] == 'ArrowUp':
                    viz.select_voxel.pop()
                    viz.current_voxel -= 1
                elif event['key'] == 'Delete':
                    viz.current_voxel += 1

                if viz.current_voxel < viz.voxel_length:
                    viz.line(Y=np.concatenate((viz.rt[viz.current_voxel], viz.rv[viz.current_voxel])),
                             X=np.arange(0, 7740), win=voxel_win, update='replace',
                             opts=dict(width=1650, height=300, title='voxel Number {}'.format(viz.current_voxel)))
                    viz.text('voxel Number {}'.format(viz.current_voxel), win=callbackwin)
                else:
                    with open("selectVoxel_subject_{}_test_ips.json".format(subject), 'w') as fp:
                        json.dump(list(viz.select_voxel), fp)
                    viz.clear_event_handlers(voxel_win)

    viz.register_event_handler(type_callback, callbackwin)
    # a = input()


def show_weight_from_model(viz):
    from regression_feature_latent import WeightRegressionModel
    model_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressionLatentModelWD/subject_{}/frame_{}".format(3, 1)
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", 'r') as ebdf:
        embeds = ebdf['embeds'][:]
    model = WeightRegressionModel(128, embeds)
    weights = []
    for latent_idx in range(1024):
        model.load_state_dict(
            torch.load(
                os.path.join(model_dir, "subject_{}_frame_{}_regression_model_i_{}_o_{}_latent_{}.pth".format(
                    3, 1, 128, 128, latent_idx))))
        w = model.transWeight()
        weights.append(w)

    for i, w in enumerate(weights):
        viz.line(Y=w, X=np.arange(0, 128), opts=dict(width=1600, height=300, title='weight_{}'.format(i)))


def show_vim1_stimuli(viz, dt_key):
    from MyDataset import vim1_stimuli_dataset
    # file = "/data1/home/guangjie/Data/vim-1/BlurStimuli.hdf5"
    file = "/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/rec_from_vqvae_sv_blur.hdf5"
    win = viz.image(np.random.randn(3, 128, 128), opts={'title': 'ori'})
    stimuli = vim1_stimuli_dataset(file, dt_key, 3)
    for i in range(100):
        img = stimuli[i].transpose((2, 0, 1))
        viz.image(img, win=win)
        time.sleep(1)
        print(i)
    # f = sio.loadmat(file)
    # dt = f[dt_key]
    # dt = np.clip((dt - np.min(dt)) / np.max(dt - np.min(dt)), 0, 1)
    # for img in dt:
    #     viz.image(img, win=win)
    #     time.sleep(1)


def show_vim1_rec(viz, dt_key):
    file = "/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/rec_from_vqvae_{}_blur.hdf5".format(dt_key)
    win = viz.image(np.random.randn(3, 128, 128), opts={'title': 'ori vs rec'})
    with h5py.File(file, 'r') as recf:
        rec = recf['rec']
        for img in rec[1700:]:
            img = img.transpose((2, 0, 1))
            viz.image(img, win=win)
            time.sleep(1)


def visual_vim1_fmri(viz, rois, subject):
    from MyDataset import vim1_fmri_dataset
    viz.rt = vim1_fmri_transpose_dataset("/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat", rois,
                                         'dataTrnS{}'.format(subject), subject)
    # viz.rv = vim1_fmri_dataset("/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat", rois,
    #                            'dataValS{}'.format(subject), subject)

    viz.current_voxel = 0
    viz.voxel_length = viz.rt.n_voxels
    viz.select_voxel = []
    voxel_win = viz.line(Y=viz.rt[viz.current_voxel], X=np.arange(0, 1750),
                         opts=dict(width=1600, height=300, title='voxel'))
    callbackwin = viz.text('Voxel Number {}'.format(viz.current_voxel))

    def type_callback(event):
        # global rt, current_voxel, voxel_win, voxel_length, viz, select_voxel
        if event['event_type'] == 'KeyPress':
            if event['key'] == 'ArrowDown' or event['key'] == 'ArrowUp' or event['key'] == 'Delete':
                if event['key'] == 'ArrowDown':
                    viz.select_voxel.append(viz.current_voxel)
                    viz.current_voxel += 1
                elif event['key'] == 'ArrowUp':
                    viz.select_voxel.pop()
                    viz.current_voxel -= 1
                elif event['key'] == 'Delete':
                    viz.current_voxel += 1

                if viz.current_voxel < viz.voxel_length:
                    viz.line(Y=viz.rt[viz.current_voxel], X=np.arange(0, 1750), win=voxel_win, update='replace',
                             opts=dict(width=1600, height=300, title='voxel Number {}'.format(viz.current_voxel)))
                    viz.text('voxel Number {}'.format(viz.current_voxel), win=callbackwin)
                else:
                    with open("vim1_subject_{}_roi_{}_voxel_select.json".format(subject, ''.join(map(str, rois))),
                              'w') as fp:
                        json.dump(list(viz.select_voxel), fp)
                    viz.clear_event_handlers(voxel_win)

    viz.register_event_handler(type_callback, callbackwin)


def show_vqvae_feature_map(viz):
    stimf = sio.loadmat("/data1/home/guangjie/Data/vim-1/Stimuli.mat")
    stimTrn = stimf['stimTrn']
    with h5py.File("/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/rec_from_vqvae_st.hdf5",
                   'r') as decf:
        dec = decf['rec']
        with h5py.File("/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/k_from_vqvae_st.hdf5",
                       'r') as kf:
            k = kf['k'][:].astype(np.uint64)
            with h5py.File("/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/normlized_zq_of_st_fmap_mm_scaler_T_23x23_blur3.hdf3",
                           'r') as zef:
                ori_ze = zef['latent'][:]
                with h5py.File(
                        "/data1/home/guangjie/Data/vim2/extract_from_vqvae/zq_of_st_fmap_mm_scaler_23x23_blur3.hdf3",
                        'r') as zef:
                    ze = zef['latent'][:]
                    with h5py.File(
                            "/data1/home/guangjie/Data/vim1/rec_by_ze_of_vqvae_fmap/subject1/dataTrnS1/subject_1_dataTrnS1_roi_123456_rec_wd_0.06_normalize_blur.hdf5",
                            'r') as recf:
                        rec = recf['rec'][:] / 255.0
                        with h5py.File(
                                "/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
                                'r') as ef:
                            embeds = ef['embeds'][:]
                            ori_dec_rec_win = viz.images(np.random.randn(3, 3, 128, 128))
                            fmap_win = viz.images(np.random.randn(128, 1, 32, 32))
                            ori_ze_win = viz.images(np.random.randn(128, 1, 23, 23))
                            ze_win = viz.images(np.random.randn(128, 1, 23, 23))
                            for i, (_stim, _dec, _rec, _k, _oze, _ze) in enumerate(
                                    zip(stimTrn, dec, rec, k, ori_ze, ze)):
                                if i < 1699: continue
                                # _stim = ((_stim + 0.5) * 255).astype(np.uint8)
                                _stim = np.clip((_stim - np.min(_stim)) / np.max(_stim - np.min(_stim)), 0, 1)
                                _stim = np.repeat(_stim[:, :, np.newaxis], 3, axis=-1)
                                fmap = embeds[_k].transpose((2, 0, 1))[:, np.newaxis, :, :]
                                fmap = np.clip((fmap - np.min(fmap)) / np.max(fmap - np.min(fmap)), 0, 1)

                                _oze = _oze.transpose((2, 0, 1))[:, np.newaxis, :, :]
                                _oze = np.clip((_oze - np.min(_oze)) / np.max(_oze - np.min(_oze)), 0, 1)

                                _ze = _ze.transpose((2, 0, 1))[:, np.newaxis, :, :]  # / np.max(_ze)
                                _ze = np.clip((_ze - np.min(_ze)) / np.max(_ze - np.min(_ze)), 0, 1)

                                viz.images(
                                    [_stim.transpose((2, 0, 1)), _dec.transpose((2, 0, 1)), _rec.transpose((2, 0, 1))],
                                    win=ori_dec_rec_win)
                                viz.images(fmap, win=fmap_win)
                                viz.images(_oze, win=ori_ze_win)
                                viz.images(_ze, win=ze_win)
                                # fmaps = np.concatenate(fmap, 1)
                                # cv2.imshow('fmaps', fmap)
                                # cv2.imshow('img', np.concatenate([_stim, _rec], axis=-2))  #
                                # cv2.waitKey(1)
                                print(i)


def show_normlized_zq(viz):
    with h5py.File("/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/normlized_zq_of_st.hdf5",
                   'r') as zqf:
        zq = zqf['latent']
        fmap_win = viz.images(np.random.randn(128, 1, 32, 32))
        for i, _zq in enumerate(zq):
            fmap = _zq.transpose((2, 0, 1))[:, np.newaxis, :, :]
            viz.images(fmap, win=fmap_win)
            print(i)


if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='visualizeTest')
    assert viz.check_connection(timeout_seconds=3)

    # testHeatmap(viz)
    # show_z_flow(viz)
    # test_window_callback(viz)
    # visualize_and_select_voxels(viz, subject=3)
    # show_rec_one_frame(viz, 'rv', frame_idx=0,
    #                    recRootDir="/data1/home/guangjie/Data/vim-2-gallant/rec_by_ze_of_vqvae_use_k")
    ############################################################
    # stimTrn
    # show_vim1_ori_and_rec(viz, 'stimTrn',
    #                       "/data1/home/guangjie/Data/vim1/rec_by_ze_of_vqvae_fmap/subject1/dataTrnS1/subject_1_dataTrnS1_roi_12_rec_wd_0.001_normalizedFmap.hdf5",
    #                       start=0, end=1750)
    # show_weight_from_model(viz)
    # show_vim1_stimuli(viz,'stimTrn')
    # show_vim1_rec(viz, 'st')
    ################################################################
    # show_rec_one_frame_uniform_sample(viz=viz, dt_key='rt', frame_idx=1,
    #                                   rec_path="/data1/home/guangjie/Data/vim-2-gallant/rec_by_ze_of_vqvae_fmap/subject3/rt/frame_1/subject_3_frame_1_rec_wd03.hdf5",
    #                                   sample_interval=36)
    # show_vim1_stimuli(viz, 'stimTrn')  # stimVal
    # show_vim1_rec(viz, 'st')
    # "/data1/home/guangjie/Data/vim-2-gallant/rec_by_ze_of_vqvae_fmap"
    # show_gen_k_heatmap('rv')
    # _Y = np.linspace(-5, 5, 100)
    # Y = np.column_stack((_Y * _Y, np.sqrt(_Y + 5)))
    # X = np.column_stack((_Y, _Y))
    # viz.line(
    #     Y=Y,
    #     X=X,
    #     opts=dict(markers=False),
    # )
    # visual_vim1_fmri(viz, rois=[1,2,3], subject=1)
    #############################################
    show_vqvae_feature_map(viz)
    # show_normlized_zq(viz)
    a = input()
    print('end')
    # viz.close()
