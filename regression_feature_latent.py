import torch
import torch.nn as nn
from MyDataset import fmri_dataset, vim2_predict_and_true_k_dataset, vim2_predict_latent_dataset, \
    vim2_predict_and_all_true_k_dataset
import h5py
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
    return normal_criterion(out, target), argminloss
    # -torch.log(argmaxloss)  # 1 / argmaxloss  # todo 超参数    #torch.log(argmaxloss)


def constraint2dist(out, k):
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
    return dist2, argminloss
    # return torch.log(dist2 + 1), dist2  # torch.log(argminloss + 1)
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


class MyRegressionModel(nn.Module):
    '''
    学习一个针对每层feature map的权重，用一维卷积实现
    '''

    def __init__(self, data_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(data_dim))
        self.bias = nn.Parameter(torch.ones(data_dim))
        nn.init.uniform_(self.weight, a=-0.005, b=0.005)
        nn.init.uniform_(self.bias, a=-0.005, b=0.005)

    def forward(self, x):
        out = self.weight * x + self.bias
        return out


class WeightRegressionModel(nn.Module):
    def __init__(self, data_dim, embeds):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(data_dim))  # 每个latent 对应一个weight
        # nn.init.uniform_(self.weight, a=-0.1, b=0.1)  # todo apply 外部实现
        self.pdist = nn.PairwiseDistance(p=2).cuda()
        # embedf = h5py.File(embeds_file)
        self.embeds = torch.as_tensor(embeds).cuda()
        # self.activate = torch.nn.Sigmoid()
        # self.embedding = nn.Embedding.from_pretrained(embeds)

    def transWeight(self):
        weight = (self.weight - torch.mean(self.weight)) / torch.sqrt(torch.var(self.weight) + 1e-5)
        weight = 1 / (1 + torch.exp(-7 * weight))  # self.activate(weight) #
        return weight

    def forward(self, *input):
        predict_latent = input[0]
        k = input[1]
        true_latent = self.embeds[k]  # torch.LongTensor([k])
        # self.weight = torch.nan
        weight = self.transWeight()
        # weight = (self.weight - torch.mean(self.weight))  / torch.sqrt(torch.var(self.weight) + 1e-5) #
        # pdist = self.pdist(predict_latent * self.weight, true_latent * self.weight)
        pdist = torch.mean((predict_latent * weight - true_latent * weight) ** 2, dim=1)
        # normal_loss = torch.mean((out - embeds[k]) ** 2, dim=1)
        expand_latent = predict_latent[:, None, :]
        expand_latent = expand_latent.expand(expand_latent.shape[0], 512, 128)
        expand_embeds = self.embeds[None, :]
        expand_embeds = expand_embeds.expand(expand_latent.shape[0], 512, 128)
        # dist = torch.mean((expand_out - embeds) ** 2, dim=2)
        # allDist = self.pdist(expand_latent * self.weight, expand_embeds * self.weight)
        allDist = torch.mean((expand_latent * weight - expand_embeds * weight) ** 2, dim=2)
        diff = pdist[:, None] - allDist  # todo
        select_idx = torch.le(torch.tensor(0.0).cuda(), diff)
        diffargmin = torch.mean(diff[select_idx])

        return torch.mean(pdist), diffargmin

    def find_weighted_k(self, predict_latent):
        weight = self.transWeight().cuda()

        expand_latent = predict_latent[:, None, :]
        expand_latent = expand_latent.expand(expand_latent.shape[0], 512, 128)
        expand_embeds = self.embeds[None, :]
        expand_embeds = expand_embeds.expand(expand_latent.shape[0], 512, 128)
        allDist = torch.mean((expand_latent * weight - expand_embeds * weight) ** 2, dim=2)
        k = torch.argmin(allDist, dim=1)  # todo 检查逻辑
        return k


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
        nn.init.uniform_(m.weight, a=-0.005, b=0.005)
        # if m.bias:
        nn.init.uniform_(m.bias, a=-0.005, b=0.005)
        # m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.normal_(0.0, 0.02)


def train(viz, model, dataloader, train_win_mse, train_win_dist, logIter, optimiser, criterion, train_global_idx):
    model.train()
    for step, (latent, k) in enumerate(dataloader):
        # model.zero_grad()
        optimiser.zero_grad()
        latent = latent.cuda()
        # vector = vector.cuda()
        mseLoss, distLoss = model(latent, k)
        # out = model(latent)
        # mseLoss, distLoss = criterion(out, k)
        # adjust_distLoss = distLoss  # (distLoss + 1e-5)
        # * 0.1  # * 2  # (distLoss + 1e-5) * 9300 * (mseLoss.detach().item() + 1e-5)
        #  todo##########################################
        loss = mseLoss + distLoss  # adjust_distLoss
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
        for step, (latent, k) in enumerate(test_dataloader):
            latent = latent.cuda()
            # vector = vector.cuda()
            mseLoss, distLoss = model(latent, k)
            # out = model(latent)
            # mseLoss, distLoss = criterion(out, k)
            # loss = mseLoss

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
    modelSaveRoot = '/data1/home/guangjie/Data/vim-2-gallant/regressionLatentModelSklearn/subject_{}/frame_{}'.format(
        subject, frame_idx)  # todo
    os.makedirs(modelSaveRoot, exist_ok=True)
    # model = MyRegressionModel(i_dim).cuda()
    # model = LinearRegressionModel(i_dim, o_dim).cuda()
    # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
    criterion = constraint2dist  # constraintMSELoss  # nn.MSELoss()  # Mean Squared Loss
    # optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent
    # optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimiser = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", 'r') as ebdf:
        embeds = ebdf['embeds'][:]
    test_loss_dict = {}
    for latent_idx in np.arange(latence_start, latence_end):  # range(n_lantent):
        # model.apply(init_weights)
        # model = MyRegressionModel(i_dim).cuda()
        model = WeightRegressionModel(i_dim, embeds).cuda()
        optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent
        # "/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_map/subject_{}/rt/frame_{}/subject_{}_frame_{}_ze_fmap_all.hdf5"
        dataset = vim2_predict_and_all_true_k_dataset(
            predict_latent_file="/data1/home/guangjie/Data/vim-2-gallant/regressed_feature_maps_by_sklearn/subject_{}/rt/frame_{}/predict_feature_maps.hdf5".format(
                subject, frame_idx),
            k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_train_frame_1_7000_uniform_sample.hdf5",
            latent_idx=latent_idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = vim2_predict_and_all_true_k_dataset(
            predict_latent_file="/data1/home/guangjie/Data/vim-2-gallant/regressed_feature_maps_by_sklearn/subject_{}/rv/frame_{}/predict_feature_maps.hdf5".format(
                subject, frame_idx),
            k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_test_frame_1_200_uniform_sample.hdf5",
            latent_idx=latent_idx)
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

        torch.save(model.state_dict(), os.path.join(modelSaveRoot,
                                                    "subject_{}_frame_{}_regression_model_i_{}_o_{}_latent_{}.pth".format(
                                                        subject, frame_idx, i_dim, o_dim, latent_idx)))
        test_loss_dict[str(latent_idx)] = test_loss_list
    return test_loss_dict


def apply_regression_to_fmri(dt_key, frame_idx, subject=1, n_lantent=1024, model_in_dim=4917, model_out_dim=128,
                             dim_lantent=128):
    model_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressionLatentModel/subject_{}/frame_{}".format(subject,
                                                                                                           frame_idx)
    save_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_latent/subject_{}/{}/frame_{}".format(
        subject,
        dt_key,
        frame_idx)
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


def apply_regression_to_fmri_concatenate(dt_key, frame_idx, n_latent, model_in_dim, model_out_dim, subject):
    model_dir = '/data1/home/guangjie/Data/vim-2-gallant/regressionLatentModelSklearn/subject_{}/frame_{}'.format(
        subject,
        frame_idx)
    save_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressed_k_of_vqvae_by_weighted_fmap/subject_{}/{}/frame_{}".format(
        subject, dt_key, frame_idx)
    os.makedirs(save_dir, exist_ok=True)
    # model = NonLinearRegressionModel(model_in_dim, model_out_dim).cuda()
    # model = LinearRegressionModel(model_in_dim, model_out_dim).cuda()
    # model = MyRegressionModel(model_in_dim).cuda()
    with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", 'r') as ebdf:
        embeds = ebdf['embeds'][:]
    model = WeightRegressionModel(model_in_dim, embeds)
    # mean, std = get_vim2_fmri_mean_std(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5",
    #     'rt')  # todo 归一化方式
    # dataset = vim2_predict_latent_dataset(
    #     predict_latent_file="/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_map/subject_{}/{}/frame_{}/subject_{}_frame_{}_ze_fmap_all.hdf5".format(
    #         subject, dt_key, frame_idx, subject, frame_idx), latent_idx=0)
    dataset = vim2_predict_latent_dataset(
        "/data1/home/guangjie/Data/vim-2-gallant/regressed_feature_maps_by_sklearn/subject_{}/{}/frame_{}/predict_feature_maps.hdf5".format(
            subject, dt_key, frame_idx), latent_idx=0)
    # 只是用一下len(dataset)
    with h5py.File(os.path.join(save_dir, "subject_{}_frame_{}_k_fmap_all_sklearn.hdf5".format(
            subject, frame_idx)), 'w') as sf:
        kDataset = sf.create_dataset('k', shape=(len(dataset), 32, 32), chunks=True)
        for latent_idx in range(n_latent):
            model.load_state_dict(
                torch.load(
                    os.path.join(model_dir, "subject_{}_frame_{}_regression_model_i_{}_o_{}_latent_{}.pth".format(
                        subject, frame_idx, i_dim, o_dim, latent_idx))))
            # dataset = vim2_predict_latent_dataset(
            #     predict_latent_file="/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_map/subject_{}/{}/frame_{}/subject_{}_frame_{}_ze_fmap_all.hdf5".format(
            #         subject, dt_key, frame_idx, subject, frame_idx), latent_idx=latent_idx)
            dataset = vim2_predict_latent_dataset(
                "/data1/home/guangjie/Data/vim-2-gallant/regressed_feature_maps_by_sklearn/subject_{}/{}/frame_{}/predict_feature_maps.hdf5".format(
                    subject, dt_key, frame_idx), latent_idx=latent_idx)
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
            row = latent_idx // 32
            col = latent_idx % 32
            with torch.no_grad():
                begin_idx = 0
                model.eval()
                for step, latent in enumerate(dataloader):
                    # out = model(latent.cuda())
                    k = model.find_weighted_k(latent.cuda())
                    end_idx = begin_idx + len(k)
                    kDataset[begin_idx:end_idx, row, col] = k.cpu().numpy()
                    begin_idx = end_idx
            print(latent_idx)


if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='latent regression')
    assert viz.check_connection(timeout_seconds=3)
    torch.manual_seed(7)

    lr = 0.002  # best:0.2
    weight_decay = 0  # best:0.01 todo 调整此参数，改变test loss 随train loss 下降的程度
    epochs = 100  # best:200 50
    logIterval = 30
    subject = 1
    i_dim = 128
    o_dim = 128
    n_frames = 15
    # todo frame_1 Adam w_d 0.1
    # show_regression_performance_all()
    # for i in range(1024):
    # show_regression_performance(model_in_dim=i_dim, model_out_dim=o_dim, frame_idx=0, latent_idx=0)
    latent_start = 768
    latent_end = 1024
    # with open("testlosslog/for_latent/test_loss_{}_{}_wd_{}.json".format(latent_start, latent_end, weight_decay),
    #           'w') as fp:
    # test_loss = train_all_frame(viz=viz, frame_start=1, frame_end=2, latence_start=latent_start,
    #                             latence_end=latent_end, lr=lr, weight_decay=weight_decay, init_weights=init_weights,
    #                             epochs=epochs, logIterval=logIterval, drawline=False, batch_size=128, num_workers=0,
    #                             i_dim=i_dim, o_dim=o_dim, subject=3)
    ##########################
    # model = WeightRegressionModel(128,
    #                               "/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5").cuda()
    # model.load_state_dict(torch.load(
    #     "/data1/home/guangjie/Data/vim-2-gallant/regressionLatentModel/subject_1/frame_0/subject_1_frame_0_regression_model_i_128_o_128_latent_875.pth"))
    # dataset = vim2_predict_and_true_k_dataset(
    #     predict_latent_file="/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_map/subject_{}/rt/frame_{}/subject_{}_frame_{}_ze_fmap_all.hdf5".format(
    #         subject, 0, subject, 0),
    #     k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_train.hdf5", frame_idx=0,
    #     latent_idx=latent_start)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    # for step,(predict_latent,_) in enumerate(dataloader):
    #     k = model.find_weighted_k(predict_latent.cuda())
    #     print(k)
    ##################################
    # json.dump(test_loss, fp)
    apply_regression_to_fmri_concatenate(dt_key='rv', frame_idx=1, n_latent=1024, model_in_dim=i_dim,
                                         model_out_dim=o_dim, subject=3)
    # apply_regression_to_fmri('rt', frame_idx=0, subject=1, n_lantent=1024, model_in_dim=4917, model_out_dim=128,
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
