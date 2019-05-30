import torch
import torch.nn as nn
from MyDataset import fmri_dataset, fmri_fmap_dataset, fmri_fpoint_dataset
import h5py
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module
        # self.activate = nn.Tanh()

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
        return out  # self.activate(out)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, a=-0.0005, b=0.0005)
        # if m.bias:
        nn.init.uniform_(m.bias, a=-0.0005, b=0.0005)
        # m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.normal_(0.0, 0.02)


def train(viz, model, dataloader, train_win_mse, logIter, optimiser, criterion, train_global_idx):
    model.train()
    for step, (fmri, point) in enumerate(dataloader):
        # model.zero_grad()
        optimiser.zero_grad()
        fmri = fmri.cuda()
        point = point.cuda()

        out = model(fmri)
        mseLoss = criterion(out, point)
        loss = mseLoss
        loss.backward()

        optimiser.step()

        if step % logIter == 0:
            if train_win_mse:
                viz.line(Y=mseLoss.view(1), X=train_global_idx, win=train_win_mse, update="append",
                         opts={'title': 'train mse loss'})
                # viz.line(Y=distLoss.view(1), X=train_global_idx, win=train_win_dist, update="append",
                #          opts={'title': 'train dist loss'})
                train_global_idx += 1
                print('step_{}_train_loss : {}'.format(step, loss.item()))


def test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx):
    model.eval()
    # test_global_idx = np.array([0])
    with torch.no_grad():
        mse_loss_list = []
        # dist_loss_list = []
        for step, (fmri, point) in enumerate(test_dataloader):
            fmri = fmri.cuda()
            point = point.cuda()

            out = model(fmri)
            mseLoss = criterion(out, point)
            # loss = mseLoss

            mse_loss_list.append(mseLoss)
            # dist_loss_list.append(distLoss)

        mse_mean_loss = sum(mse_loss_list) / len(mse_loss_list)
        # dist_mean_loss = sum(dist_loss_list) / len(dist_loss_list)
        if test_win_mse:
            viz.line(Y=mse_mean_loss.view(1), X=test_global_idx, win=test_win_mse, update="append",
                     opts={'title': 'test mse loss'})
            # viz.line(Y=dist_mean_loss.view(1), X=test_global_idx, win=test_win_dist, update="append",
            #          opts={'title': 'test dist loss'})
            test_global_idx += 1
            # print('test_mse_loss : {},test_dist_loss:{}'.format(mse_mean_loss.item(), dist_mean_loss.item()))
            print('test_mse_loss : {}'.format(mse_mean_loss.item()))
        return mse_mean_loss  # todo


def train_one_frame(viz, init_weights, mean, std, epochs, lr, weight_decay, logIterval, drawline, frame_idx,
                    fpoint_start, fpoint_end, batch_size, num_workers, i_dim, o_dim, subject=1):
    saveDir = "/data1/home/guangjie/Data/vim-2-gallant/regressionFeaturePointModel/subject_{}/frame_{}".format(subject,
                                                                                                               frame_idx)
    os.makedirs(saveDir, exist_ok=True)
    model = LinearRegressionModel(i_dim, o_dim).cuda()
    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent
    test_loss_of_frame = []
    for p_idx in np.arange(fpoint_start, fpoint_end):
        model.apply(init_weights)
        dataset = fmri_fpoint_dataset(
            fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_train.hdf5".format(
                subject), k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_train.hdf5",
            embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
            fmri_key='rt', frame_idx=frame_idx, fpoint_idx=p_idx, mean=mean, std=std)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = fmri_fpoint_dataset(
            fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_test.hdf5".format(
                subject), k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_test.hdf5",
            embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
            fmri_key='rt', frame_idx=frame_idx, fpoint_idx=p_idx, mean=mean, std=std)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        train_global_idx = np.array([0])
        test_global_idx = np.array([0])

        if drawline:
            train_win_mse = viz.line(Y=np.array([0]))
            # train_win_dist = viz.line(Y=np.array([0]))
            test_win_mse = viz.line(Y=np.array([0]))
            # test_win_dist = viz.line(Y=np.array([0]))
        else:
            train_win_mse = None
            # train_win_dist = None
            test_win_mse = None
            # test_win_dist = None
        test_loss_of_point = []
        for ep in range(epochs):
            train(viz, model, dataloader, train_win_mse, logIterval, optimiser, criterion,
                  train_global_idx)
            test_loss = test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx)
            print('frame_{}_point_{}_epoch_{}_test_loss: {}'.format(frame_idx, p_idx, ep, test_loss))
            test_loss_of_point.append(test_loss.cpu().item())
            # todo 调整 lr
        test_loss_of_frame.append(test_loss_of_point)
        del train_win_mse
        del test_win_mse
        torch.save(model.state_dict(),
                   os.path.join(saveDir, "subject_{}_frame_{}_regression_model_i_{}_o_{}_fpoint_{}.pth".format(
                       subject, frame_idx, i_dim, o_dim, p_idx)))
    return test_loss_of_frame


# def train_one_frame(viz, init_weights, mean, std, epochs, lr, weight_decay, logIterval, drawline, frame_idx,
#                     fmap_start, fmap_end, batch_size=128, num_workers=0, i_dim=4917, o_dim=128, subject=1):
#     model = LinearRegressionModel(i_dim, o_dim).cuda()
#     # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
#     model.apply(init_weights)
#     # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
#     criterion = nn.MSELoss()  # Mean Squared Loss
#     optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent
#
#     saveDir = '/data1/home/guangjie/Data/vim-2-gallant/regressionFeatureMapModel/subject_{}/frame_{}'.format(subject,
#                                                                                                              frame_idx)
#     os.makedirs(saveDir, exist_ok=True)
#     test_loss_of_frame = []
#     for fmap_idx in np.arange(fmap_start, fmap_end):  # range(n_lantent):
#         model.apply(init_weights)
#         # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
#         dataset = fmri_fmap_dataset(
#             fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_train.hdf5".format(
#                 subject), k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_train.hdf5",
#             embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
#             fmri_key='rt', frame_idx=frame_idx, fmap_idx=fmap_idx)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
#         test_dataset = fmri_fmap_dataset(
#             fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_test.hdf5".format(
#                 subject), k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_test.hdf5",
#             embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
#             fmri_key='rt', frame_idx=frame_idx, fmap_idx=fmap_idx)
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
#         train_global_idx = np.array([0])
#         test_global_idx = np.array([0])
#
#         if drawline:
#             train_win_mse = viz.line(Y=np.array([0]))
#             # train_win_dist = viz.line(Y=np.array([0]))
#             test_win_mse = viz.line(Y=np.array([0]))
#             # test_win_dist = viz.line(Y=np.array([0]))
#         else:
#             train_win_mse = None
#             # train_win_dist = None
#             test_win_mse = None
#             # test_win_dist = None
#         test_loss_of_fmap = []
#         for ep in range(epochs):
#             train(viz, model, dataloader, train_win_mse, logIterval, optimiser, criterion,
#                   train_global_idx)
#             test_loss = test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx)
#             print('frame_{}_fmap_{}_epoch_{}_test_loss: {}'.format(frame_idx, fmap_idx, ep, test_loss))
#             test_loss_of_fmap.append(test_loss.cpu().item())
#             # todo 调整 lr
#         test_loss_of_frame.append(test_loss_of_fmap)
#         del train_win_mse
#         del test_win_mse
#         torch.save(model.state_dict(),
#                    os.path.join(saveDir, "subject_{}_frame_{}_regression_model_i_{}_o_{}_fmap_{}.pth".format(
#                        subject, frame_idx, i_dim, o_dim, fmap_idx)))
#     return test_loss_of_frame


def apply_regression_to_fmri(dt_key, frame_idx, n_point, subject, model_in_dim, model_out_dim):
    model_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressionFeaturePointModel/subject_{}/frame_{}".format(
        subject, frame_idx)
    save_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_point/subject_{}/{}/frame_{}".format(
        subject, dt_key, frame_idx)
    os.makedirs(save_dir, exist_ok=True)
    # model = NonLinearRegressionModel(model_in_dim, model_out_dim).cuda()
    model = LinearRegressionModel(model_in_dim, model_out_dim).cuda()
    # mean, std = get_vim2_fmri_mean_std(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5",
    #     'rt')  # todo 归一化方式
    mean, std = None, None
    dataset = fmri_dataset(
        "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_{}.hdf5".format(
            'train' if dt_key == 'rt' else 'test'), mean, std, 'rt')  # todo train test
    # dataset = fmri_dataset(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5",
    #     mean, std, dt_key)  # todo
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    with h5py.File(os.path.join(save_dir, "subject_{}_frame_{}_ze_fpoint_all.hdf5".format(subject, frame_idx)),
                   'w') as sf:
        latent = sf.create_dataset('latent', shape=(len(dataset), 32, 32, 128))

        for p_idx in range(n_point):
            model.load_state_dict(
                torch.load(
                    os.path.join(model_dir, "subject_{}_frame_{}_regression_model_i_{}_o_{}_fpoint_{}.pth".format(
                        subject, frame_idx, i_dim, o_dim, p_idx))))
            row_column = p_idx // 128
            latent_idx = p_idx % 128
            row = row_column // 32
            column = row_column % 32

            with torch.no_grad():
                begin_idx = 0
                model.eval()
                for step, fmri in enumerate(dataloader):
                    out = model(fmri.cuda())
                    end_idx = begin_idx + len(out)
                    latent[begin_idx:end_idx, row, column, latent_idx] = out.squeeze().cpu().numpy()  # 需要cpu().numpy()?
                    begin_idx = end_idx
            print(p_idx)


def train_frames(viz, i_dim, o_dim, lr, weight_decay, init_weights, epochs, logIterval, drawline, frame_start,
                 frame_end, fpoint_start, fpoint_end, batch_size, num_workers, subject):
    # mean, std = get_vim2_fmri_mean_std(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_train.hdf5", dt_key='rt')
    mean, std = None, None
    # o_mean,o_std = get_vector_mean_std()
    testloss_of_all_fmaps = {}
    for frame_idx in np.arange(frame_start, frame_end):
        print('{} frame begin:'.format(frame_idx))
        testloss_list = train_one_frame(viz, init_weights, mean, std, epochs, lr, weight_decay, logIterval,
                                        drawline=drawline, frame_idx=frame_idx, fpoint_start=fpoint_start,
                                        fpoint_end=fpoint_end, batch_size=batch_size, num_workers=num_workers,
                                        i_dim=i_dim, o_dim=o_dim, subject=subject)
        testloss_of_all_fmaps[str(frame_idx)] = testloss_list
    return testloss_of_all_fmaps


def show_regression_performance(model_in_dim, model_out_dim, frame_idx=0, latent_idx=0, time_step=15):
    model = LinearRegressionModel(model_in_dim, model_out_dim).cuda()
    model.load_state_dict(
        torch.load(
            "/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_1/frame_{}/subject_1_regression_model_i_4917_o_128_latent_{}.pth".format(
                frame_idx, latent_idx)))
    criterion = nn.MSELoss()  # Mean Squared Loss

    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5",
                   'r') as vf:
        rt_data = vf['rt']
        rv_data = vf['rv']
        rva0_data = vf['rva0']
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_st.hdf5", 'r') as st_zq_f:
            st_zq_data = st_zq_f['zq']
            with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_sv.hdf5", 'r') as sv_zq_f:
                sv_zq_data = sv_zq_f['zq']
                with torch.no_grad():
                    model.eval()
                    rt_loss_list, rv_loss_list = [], []
                    for i in range(100):  # n 个时刻
                        rt = torch.from_numpy(rt_data[:, i]).cuda()
                        o_rt = model(rt)
                        zq_st_frame_latent = torch.from_numpy(
                            st_zq_data[frame_idx + i * time_step].reshape(1024, 128)[latent_idx]).cuda()
                        rt_loss = criterion(o_rt, zq_st_frame_latent)
                        rt_loss_list.append(rt_loss.cpu().numpy())

                        rv = torch.from_numpy(rv_data[:, i]).cuda()
                        o_rv = model(rv)
                        zq_sv_frame_latent = torch.from_numpy(
                            sv_zq_data[frame_idx + i * time_step].reshape(1024, 128)[latent_idx]).cuda()
                        rv_loss = criterion(o_rv, zq_sv_frame_latent)
                        rv_loss_list.append(rv_loss.cpu().numpy())

                        # print(i, ' rt_loss:', rt_loss, 'rv_loss:', rv_loss)
                    print(latent_idx, np.mean(rt_loss_list), np.mean(rv_loss_list))
                    # rva0 = rva0_data[:, i].cuda()
                    # o_rva0 = model(rva0)


def show_regression_performance_all(frame_idx=14, latent_idx=0, time_step=15):
    criterion = nn.MSELoss()  # Mean Squared Loss

    with h5py.File(
            "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_1/rt/frame_0/subject_1_frame_0_ze_latent_all.hdf5",
            'r') as reg_rt_zq_f:
        reg_rt_zq_data = reg_rt_zq_f['latent']
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_st.hdf5", 'r') as st_zq_f:
            st_zq_data = st_zq_f['zq'][frame_idx::time_step]
            for i in range(100):
                reg_zq = torch.from_numpy(reg_rt_zq_data[i])
                zq = torch.from_numpy(st_zq_data[i].reshape(1024, 128))
                loss = criterion(reg_zq, zq)
                print(loss)

    with h5py.File(
            "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_1/rv/frame_0/subject_1_frame_0_ze_latent_all.hdf5",
            'r') as reg_rv_zq_f:
        reg_rv_zq_data = reg_rv_zq_f['latent']
        with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_sv.hdf5", 'r') as sv_zq_f:
            sv_zq_data = sv_zq_f['zq'][frame_idx::time_step]
            for i in range(100):
                reg_zq = torch.from_numpy(reg_rv_zq_data[i])
                zq = torch.from_numpy(sv_zq_data[i].reshape(1024, 128))
                loss = criterion(reg_zq, zq)
                print(loss)


if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='fpoint regression')
    assert viz.check_connection(timeout_seconds=3)
    torch.manual_seed(7)

    lr = 0.001  # best:0.2
    weight_decay = 0.005  # best:0.01 todo 调整此参数，改变test loss 随train loss 下降的程度
    epochs = 60  # best:200 50
    logIterval = 30
    subject = 1
    i_dim = 4854
    o_dim = 1
    n_frames = 15
    # todo frame_1 Adam w_d 0.1
    # show_regression_performance_all()
    # for i in range(1024):
    # show_regression_performance(model_in_dim=i_dim, model_out_dim=o_dim, frame_idx=0, latent_idx=0)
    fpoint_start = 121000
    fpoint_end = 131072
    # with open("testlosslog/for_feature_point/test_loss_{}_{}_wd_{}.json".format(fpoint_start, fpoint_end, weight_decay),
    #           'w') as fp:
    # lossdict = train_frames(viz, i_dim, o_dim, lr, weight_decay, init_weights, epochs, logIterval, drawline=False,
    #                         frame_start=10, frame_end=11, fpoint_start=fpoint_start, fpoint_end=fpoint_end,
    #                         batch_size=128, num_workers=0, subject=1)
    #     json.dump({"loss": lossdict}, fp)

    apply_regression_to_fmri(dt_key='rt', frame_idx=0, n_point=131072, subject=1, model_in_dim=i_dim,
                             model_out_dim=o_dim)
