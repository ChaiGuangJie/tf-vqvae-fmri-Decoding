import torch
import torch.nn as nn
from MyDataset import fmri_vector_dataset, fmri_k_dataset, fmri_dataset, fmri_vector_k_dataset
import h5py
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", 'r') as ebf:
    embeds = torch.from_numpy(ebf['embeds'][:]).cuda()

normal_criterion = nn.MSELoss()


def constraintMSELoss(out, target):  # todo 优化
    global embeds, normal_criterion
    normal_loss = torch.mean((out - target) ** 2, dim=1)
    expand_out = out[:, None, :]
    dist = torch.mean((expand_out - embeds) ** 2, dim=2)
    # lt_idx = torch.ge(normal_loss[:, None], dist)  # todo 不是lt 是gt,更改为ge,防止空列表
    diff = normal_loss[:, None] - dist
    select_idx = torch.le(torch.tensor(0.0).cuda(), diff)
    argminloss = torch.mean(diff[select_idx])
    # argminloss = nn.functional.mse_loss(out, target)
    # return torch.exp(argminloss) +
    return normal_criterion(out,
                            target), argminloss  # -torch.log(argmaxloss)  # 1 / argmaxloss  # todo 超参数    #torch.log(argmaxloss)


def constraint2dist(out, target, k):
    global embeds, normal_criterion
    # dist1 = normal_criterion(out, target)
    dist2 = normal_criterion(out, embeds[k])

    normal_loss = torch.mean((out - embeds[k]) ** 2, dim=1)
    expand_out = out[:, None, :]
    dist = torch.mean((expand_out - embeds) ** 2, dim=2)
    # todo 防止空列表
    diff = normal_loss[:, None] - dist
    select_idx = torch.le(torch.tensor(0.0).cuda(), diff)
    argminloss = torch.mean(diff[select_idx])
    # ge_idx = torch.ge(normal_loss[:, None], dist)
    # return dist2, argminloss
    return torch.log(dist2 + 1), dist2  # torch.log(argminloss + 1)
    # return torch.exp(dist2) - 1, torch.exp(argminloss) - 1  # torch.exp(argminloss) - 1
    # argmaxloss = torch.mean(dist[ge_idx])
    # return (dist1 + dist2) / 2, argminloss
    # return (torch.log(dist1 + 1) + torch.log(dist2 + 1)) / 2, torch.log(argminloss+ 1)
    # return dist1, dist2 / argmaxloss - 1
    # return torch.log(dist1 + dist2 + 1), -torch.log(argmaxloss)*0.001
    # 0.001 / argmaxloss  # torch.exp(-argmaxloss * 0.1)  # 1 / argmaxloss
    # torch.log(1.8 - argmaxloss)  # -torch.log(argmaxloss) * 0.0001 dist1 +
    #  torch.exp((dist1 + dist2) / 2) - 1 #
    # torch.exp(-argmaxloss)*100  # argminloss  # torch.log(argminloss + 1)
    # return dist1, dist2


def constraint2distForFrames(outs, targets, ks):
    '''
    计算15帧fmri回归后的loss
    :param outs: shape=(batch_size,time_step,128)
    :param targets: shape=(batch_size,time_step,128)
    :param ks: shape=(batch_size,time_step,1)
    :return: (mse(out,ze)+mse(out,zq))/2,embeds中所有与out的距离小于target的距离差的均值
    '''
    global embeds, normal_criterion
    dist1 = normal_criterion(outs, targets)
    dist2 = normal_criterion(outs, embeds[ks])  # todo 检查

    # normal_loss = torch.mean((out - embeds[k]) ** 2, dim=1)
    # expand_out = out[:, None, :]
    # dist = torch.mean((expand_out - embeds) ** 2, dim=2)
    # # todo 防止空列表
    # diff = normal_loss[:, None] - dist
    # select_idx = torch.le(torch.tensor(0.0).cuda(), diff)
    # argminloss = torch.mean(diff[select_idx])
    # return (dist1 + dist2) / 2, argminloss


# # targetloss = nn.functional.mse_loss(out, target)
# batchloss_list = []
# for i, (o, t) in enumerate(zip(out, target)):
#     targetloss = nn.functional.mse_loss(o, t)
#     loss_list = []
#     for e in embeds:
#         loss_item = nn.functional.mse_loss(o, e)
#         if loss_item > targetloss:
#             loss_list.append(-loss_item)
#     # loss_list = torch.as_tensor(loss_list).cuda()
#     # t = torch.lt(targetloss, loss_list)
#     loss_list = torch.tensor(loss_list, requires_grad=True)
#     mean_loss = torch.mean(loss_list)
#     if not torch.isnan(mean_loss):  # todo 为什么会出现nan?
#         batchloss_list.append(mean_loss)
#     # 找到比target loss 大的 加负号加权求和返回返回
# # notargetloss = nn.functional.mse_loss(out[0], embeds)
# batchloss_list = torch.stack(batchloss_list)
# return torch.mean(batchloss_list).cuda() + nn.functional.mse_loss(out, target).cuda()


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module
        self.activate = nn.Tanh()

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
        return self.activate(out)


class NonLinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Calling Super Class's constructor
        self.linear1 = nn.Linear(input_dim, 512)  # 1024 256 128
        self.linear2 = nn.Linear(512, 256)  # 1024 256 128
        self.linear3 = nn.Linear(256, output_dim)  # 1024 256 128
        # self.linear2 = nn.Linear(4096, 2048)
        # self.linear3 = nn.Linear(2048, 1024)
        # self.linear3 = nn.Linear(256, output_dim)

        # nn.linear is defined in nn.Module
        # self.relu = nn.LeakyReLU()
        self.activate = nn.Tanh()
        # self.out_activate = nn.Tanh()

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.activate(self.linear1(x))
        out = self.activate(self.linear2(out))
        out = self.activate(self.linear3(out))
        # out = self.activate(self.linear3(out))
        # out = self.activate(self.linear4(out))
        # out = self.activate(self.linear2(out))
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, a=-0.0001, b=0.0001)
        # if m.bias:
        nn.init.uniform_(m.bias, a=-0.0001, b=0.0001)
        # m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.normal_(0.0, 0.02)


class LSTMRegressionModel(nn.Module):
    def __init__(self, in_size, hidden_size, time_step=15):
        super().__init__()
        self.lstmcell = nn.LSTMCell(in_size, hidden_size)
        self.time_step = time_step

    def forward(self, x):
        # todo hx,cx init
        hx = torch.full((x.shape[0], self.hidden_size), 0.01).cuda()  # todo 更改初始值会影响精度
        cx = torch.full((x.shape[0], self.hidden_size), 0.01).cuda()

        out_list = []
        for i in range(self.time_step):
            hx, cx = self.lstmcell(x, (hx, cx))
            out_list.append(hx)  # 每帧的回归输出

        return torch.from_numpy(out_list)


def train_lstm(viz, model, dataloader, optimiser, criterion, train_win_mse, train_win_dist, train_global_idx, logIter):
    model.train()
    for step, (fmri, latence, k) in enumerate(dataloader):
        # 一次性读15帧数据
        optimiser.zero_grad()
        fmri = fmri.cuda()
        latence = latence.cuda()
        out = model(fmri)
        mseloss, diffloss = criterion(out, latence, k)
        loss = mseloss + diffloss
        loss.backward()
        optimiser.step()
        if step % logIter == 0:
            if train_win_mse and train_win_dist:
                # viz.line(Y=torch.cat([mseLoss.view(1),distLoss.view(1)]).view(1,2), X=np.column_stack((train_global_idx,train_global_idx)), win=train_win,
                #          update="append",opts={'title': 'train loss'})
                viz.line(Y=mseloss.view(1), X=train_global_idx, win=train_win_mse, update="append",
                         opts={'title': 'train mse loss'})
                viz.line(Y=diffloss.view(1), X=train_global_idx, win=train_win_dist, update="append",
                         opts={'title': 'train dist loss'})
                # torch.cat([mseLoss.view(1),distLoss.view(1)]).view(1,2)
                # np.column_stack((train_global_idx,train_global_idx))
                train_global_idx += 1
                print('step_{}_train_loss : {}'.format(step, loss.item()))


def test_lstm(viz, model, dataloader, criterion, test_win_mse, test_win_dist, test_global_idx, logIter):
    model.eval()
    with torch.no_grad():
        mse_loss_list = []
        dist_loss_list = []
        for step, (fmri, latence, k) in enumerate(dataloader):
            # 一次性读15帧数据
            fmri = fmri.cuda()
            latence = latence.cuda()
            out = model(fmri)
            mseloss, diffloss = criterion(out, latence, k)
            # loss = mseloss + diffloss
            mse_loss_list.append(mseloss)
            dist_loss_list.append(diffloss)

        mse_mean_loss = sum(mse_loss_list) / len(mse_loss_list)
        dist_mean_loss = sum(dist_loss_list) / len(dist_loss_list)

        if test_win_mse and test_win_dist:
            viz.line(Y=mse_mean_loss.view(1), X=test_global_idx, win=test_win_mse, update="append",
                     opts={'title': 'test mse loss'})
            viz.line(Y=dist_mean_loss.view(1), X=test_global_idx, win=test_win_dist, update="append",
                     opts={'title': 'test dist loss'})
            test_global_idx += 1
            print('test_mse_loss : {},test_dist_loss:{}'.format(mse_mean_loss.item(), dist_mean_loss.item()))
        return mse_mean_loss  # todo


def train(viz, model, dataloader, train_win_mse, train_win_dist, logIter, optimiser, criterion, train_global_idx):
    model.train()
    for step, (fmri, k) in enumerate(dataloader):
        # model.zero_grad()
        optimiser.zero_grad()
        fmri = fmri.cuda()
        # vector = vector.cuda()

        out = model(fmri)
        mseLoss, distLoss = criterion(out, None, k)
        adjust_distLoss = distLoss  # (distLoss + 1e-5)
        # * 0.1  # * 2  # (distLoss + 1e-5) * 9300 * (mseLoss.detach().item() + 1e-5)
        #  todo##########################################
        loss = mseLoss + adjust_distLoss
        loss.backward()

        optimiser.step()

        if step % logIter == 0:
            if train_win_mse and train_win_dist:
                # viz.line(Y=torch.cat([mseLoss.view(1),distLoss.view(1)]).view(1,2), X=np.column_stack((train_global_idx,train_global_idx)), win=train_win,
                #          update="append",opts={'title': 'train loss'})
                viz.line(Y=mseLoss.view(1), X=train_global_idx, win=train_win_mse, update="append",
                         opts={'title': 'train mse loss'})
                viz.line(Y=distLoss.view(1), X=train_global_idx, win=train_win_dist, update="append",
                         opts={'title': 'train dist loss'})
                # torch.cat([mseLoss.view(1),distLoss.view(1)]).view(1,2)
                # np.column_stack((train_global_idx,train_global_idx))
                train_global_idx += 1
                print('step_{}_train_loss : {}'.format(step, loss.item()))


def test(viz, model, test_dataloader, test_win_mse, test_win_dist, criterion, test_global_idx):
    model.eval()
    # test_global_idx = np.array([0])
    with torch.no_grad():
        mse_loss_list = []
        dist_loss_list = []
        for step, (fmri, k) in enumerate(test_dataloader):
            fmri = fmri.cuda()
            # vector = vector.cuda()

            out = model(fmri)
            mseLoss, distLoss = criterion(out, None, k)
            loss = mseLoss

            mse_loss_list.append(mseLoss)
            dist_loss_list.append(distLoss)

        mse_mean_loss = sum(mse_loss_list) / len(mse_loss_list)
        dist_mean_loss = sum(dist_loss_list) / len(dist_loss_list)
        if test_win_mse and test_win_dist:
            viz.line(Y=mse_mean_loss.view(1), X=test_global_idx, win=test_win_mse, update="append",
                     opts={'title': 'test mse loss'})
            viz.line(Y=dist_mean_loss.view(1), X=test_global_idx, win=test_win_dist, update="append",
                     opts={'title': 'test dist loss'})
            test_global_idx += 1
            print('test_mse_loss : {},test_dist_loss:{}'.format(mse_mean_loss.item(), dist_mean_loss.item()))
        return mse_mean_loss  # todo


def train_one_frame(viz, frame_idx, latence_start, latence_end, init_weights, mean, std, epochs, lr, logIterval,
                    weight_decay, drawline, batch_size, num_workers, i_dim, o_dim, subject=1):
    os.makedirs(
        '/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_{}/frame_{}'.format(subject, frame_idx),
        exist_ok=True)
    test_loss_dict = {}
    for latent_idx in np.arange(latence_start, latence_end):  # range(n_lantent):
        model = LinearRegressionModel(i_dim, o_dim).cuda()
        # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
        criterion = constraint2dist  # constraintMSELoss  # nn.MSELoss()  # Mean Squared Loss
        optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent
        # optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimiser = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.apply(init_weights)
        # "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_train.hdf5".format(subject)
        dataset = fmri_k_dataset(
            fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/rt_train.hdf5",
            k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_train.hdf5",
            fmri_key='rt', frame_idx=frame_idx, latence_idx=latent_idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_test.hdf5".format(subject)
        test_dataset = fmri_k_dataset(
            fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/rt_test.hdf5",
            k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_test.hdf5",
            fmri_key='rt', frame_idx=frame_idx, latence_idx=latent_idx)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_global_idx = np.array([0])
        test_global_idx = np.array([0])

        if drawline:
            train_win_mse = viz.line(Y=np.array([0]))
            train_win_dist = viz.line(Y=np.array([0]))
            test_win_mse = viz.line(Y=np.array([0]))
            test_win_dist = viz.line(Y=np.array([0]))
        else:
            train_win_mse = None
            train_win_dist = None
            test_win_mse = None
            test_win_dist = None
        test_loss_list = []
        for ep in range(epochs):
            train(viz, model, dataloader, train_win_mse, train_win_dist, logIterval, optimiser, criterion,
                  train_global_idx)
            test_loss = test(viz, model, test_dataloader, test_win_mse, test_win_dist, criterion, test_global_idx)
            print('frame_{}_latent_{}_epoch_{}_test_loss: {}'.format(frame_idx, latent_idx, ep, test_loss))
            # todo 调整 lr
            test_loss_list.append(test_loss.cpu().numpy())

        del train_win_mse
        del train_win_dist
        del test_win_mse
        del test_win_dist
        torch.save(model.state_dict(),
                   '/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_{}/frame_{}/subject_{}_regression_model_i_{}_o_{}_latent_{}.pth'.format(
                       subject, frame_idx, subject, i_dim, o_dim, latent_idx))
        test_loss_dict[str(latent_idx)] = test_loss_list
    return test_loss_dict


def apply_regression_to_fmri(dt_key, frame_idx, subject=1, n_lantent=1024, model_in_dim=4917, model_out_dim=128,
                             dim_lantent=128):
    model_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_{}/frame_{}".format(subject, frame_idx)
    save_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_{}/{}/frame_{}".format(subject,
                                                                                                             dt_key,
                                                                                                             frame_idx)
    os.makedirs(save_dir, exist_ok=True)
    model = NonLinearRegressionModel(model_in_dim,
                                     model_out_dim).cuda()  # LinearRegressionModel(model_in_dim, model_out_dim).cuda()
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

    for latent_idx in range(n_lantent):
        # dataset = fmri_vector_dataset(
        #     fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_rv_rva0.hdf5".format(
        #         subject), zq_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_{}.hdf5".format(
        #         'st' if dt_key == 'rt' else 'sv'),
        #     mean=mean,
        #     std=std, fmri_key=dt_key, frame_idx=frame_idx, latent_idx=latent_idx)
        # dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
        # 逐次加载单个模型，抽取每张图片对应ze的第lantent_idx个隐含表征
        model.load_state_dict(
            torch.load(os.path.join(model_dir,
                                    "subject_{}_regression_model_i_{}_o_{}_latent_{}.pth".format(subject, model_in_dim,
                                                                                                 model_out_dim,
                                                                                                 latent_idx))))
        sf = h5py.File(os.path.join(save_dir, "subject_{}_frame_{}_ze_latent_{}.hdf5".format(
            subject, frame_idx, latent_idx)), 'w')  # todo zq
        latent = sf.create_dataset('latent', shape=(len(dataset), dim_lantent))
        with torch.no_grad():
            begin_idx = 0
            model.eval()
            for step, fmri in enumerate(dataloader):
                out = model(fmri.cuda())
                end_idx = begin_idx + len(out)
                latent[begin_idx:end_idx] = out.cpu().numpy()  # 需要cpu().numpy()?
                begin_idx = end_idx
        sf.close()
        print(latent_idx)


def train_all_frame(viz, frame_start, frame_end, latence_start, latence_end, lr, weight_decay, init_weights, epochs,
                    logIterval, drawline, batch_size, num_workers, i_dim, o_dim, subject):
    mean, std = None, None
    # o_mean,o_std = get_vector_mean_std()
    test_loss_dict = {}
    for frame_idx in np.arange(frame_start, frame_end):
        print('{} frame begin:'.format(frame_idx))
        test_loss = train_one_frame(viz=viz, frame_idx=frame_idx, latence_start=latence_start, latence_end=latence_end,
                                    init_weights=init_weights, mean=mean, std=std, epochs=epochs, lr=lr,
                                    logIterval=logIterval, weight_decay=weight_decay, drawline=drawline,
                                    batch_size=batch_size, num_workers=num_workers, i_dim=i_dim, o_dim=o_dim,
                                    subject=subject)
        test_loss_dict[str(frame_idx)] = test_loss
    return test_loss_dict


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
    viz = Visdom(server="http://localhost", env='latent regression')
    assert viz.check_connection(timeout_seconds=3)
    torch.manual_seed(7)

    lr = 0.01  # best:0.2
    weight_decay = 1  # best:0.01 todo 调整此参数，改变test loss 随train loss 下降的程度
    epochs = 30 # best:200 50
    logIterval = 30
    subject = 1
    i_dim = 73728  # 4854  #
    o_dim = 128
    n_frames = 15
    # todo frame_1 Adam w_d 0.1
    # show_regression_performance_all()
    # for i in range(1024):
    # show_regression_performance(model_in_dim=i_dim, model_out_dim=o_dim, frame_idx=0, latent_idx=0)
    latent_start = 3
    latent_end = 10
    with open("testlosslog/for_latent/test_loss_{}_{}_wd_{}.json".format(latent_start, latent_end, weight_decay),
              'w') as fp:
        test_loss = train_all_frame(viz=viz, frame_start=0, frame_end=1, latence_start=latent_start,
                                    latence_end=latent_end, lr=lr, weight_decay=weight_decay, init_weights=init_weights,
                                    epochs=epochs, logIterval=logIterval, drawline=True, batch_size=128, num_workers=0,
                                    i_dim=i_dim, o_dim=o_dim, subject=1)
        json.dump(test_loss, fp)

    # apply_regression_to_fmri('rv', frame_idx=0, subject=1, n_lantent=1024, model_in_dim=4917, model_out_dim=128,
    #                          dim_lantent=128)

    # viz.close()
    # for ep in range(epochs):
    #     # train
    #     for step, (fmri, vector) in enumerate(dataloader):
    #         model.zero_grad()
    #
    #         fmri = fmri.cuda()
    #         vector = vector.cuda()
    #
    #         out = model(fmri)
    #         loss = criterion(out, vector)
    #         loss.backward()
    #
    #         optimiser.step()
    #
    #         if step % saveIter == 0:
    #             print(loss)
    #             viz.line(Y=loss.view(1), X=train_global_idx, win=train_win, update="append",
    #                      opts={'title': 'train loss'})
    #             train_global_idx += 1
    #
    #     # test
    #     with torch.no_grad():
    #         loss_list = []
    #         for step, (fmri, vector) in enumerate(test_dataloader):
    #             fmri = fmri.cuda()
    #             vector = vector.cuda()
    #
    #             out = model(fmri)
    #             loss = criterion(out, vector)
    #
    #             loss_list.append(loss)
    #
    #         mean_loss = sum(loss_list) / len(loss_list)
    #         viz.line(Y=mean_loss.view(1), X=test_global_idx, win=test_win, update="append", opts={'title': 'test loss'})
    #         test_global_idx += 1
    # # viz.close()
