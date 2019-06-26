import torch
import torch.nn as nn
from MyDataset import fmri_dataset, fmri_fmap_all_k_dataset, vim1_fmri_k_dataset, vim1_fmri_dataset, \
    vim1_val_fmri_dataset, vim1_fmri_fmap_dataset, vim1_fmri_pca_fmap_dataset
import h5py
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np
import os
import json
import scipy.io as sio
import cv2
# import torch.functional as F
import torch.nn.functional as F
from sklearn import preprocessing

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


class Vim1_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        mat = sio.loadmat("/data1/home/guangjie/Data/vim-1/mask.mat")
        mask = mat['mask']
        self.mask = torch.as_tensor(cv2.resize(mask, (18, 18)).flatten(), dtype=torch.float).cuda()

    def forward(self, input, target):
        input = self.mask * input
        target = self.mask * target
        return F.mse_loss(input, target, reduction='mean')


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
        # self.activate = nn.Tanh()

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
        return out  # self.activate(out)  # todo


class NonLinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Calling Super Class's constructor
        self.linear1 = nn.Linear(input_dim, 1024)  # 1024 256 128
        self.linear2 = nn.Linear(1024, 516)  # 1024 256 128
        self.linear3 = nn.Linear(516, output_dim)  # 1024 256 128
        # self.linear2 = nn.Linear(4096, 2048)
        # self.linear3 = nn.Linear(2048, 1024)
        # self.linear3 = nn.Linear(256, output_dim)

        # nn.linear is defined in nn.Module
        # self.relu = nn.LeakyReLU()
        # self.activate = nn.Tanh()
        self.activate = nn.LeakyReLU()
        self.outActivate = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d()
        # self.out_activate = nn.Tanh()

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.activate(self.linear1(x))
        out = self.activate(self.linear2(out))
        out = self.outActivate(self.linear3(out))
        # out = self.activate(self.linear3(out))
        # out = self.activate(self.linear3(out))
        # out = self.activate(self.linear4(out))
        # out = self.activate(self.linear2(out))
        return out


class LinearConvRegresssionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, 7)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 7)
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(32, 64, 3)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(32, 64, 3)
        # self.bn5 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(64, 1, 1)
        self.pool = nn.MaxPool2d(2, stride=1)
        self.out_dim = output_dim

    def forward(self, x):
        x = self.bn1(self.linear(x).view(-1, 1, 32, 32))  # self .bn1()
        x = self.bn2(self.pool(F.leaky_relu(self.conv1(x))))  #
        x = self.bn3(self.pool(F.leaky_relu(self.conv2(x))))
        # x = self.bn4(self.pool(F.leaky_relu(self.conv3(x))))
        # x = self.bn5(self.pool(F.leaky_relu(self.conv4(x))))
        x = F.leaky_relu(self.conv5(x))
        return x.view(-1, self.out_dim)


def init_weights(m):
    for layer in m.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                          nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
    # if type(m) == nn.Linear:
    #     # torch.nn.init.xavier_uniform_(m.weight)
    #     # torch.nn.init.xavier_uniform_(m.bias)
    #     nn.init.uniform_(m.weight, a=-0.1, b=0.1)  # 0.02 # 0.001
    #     nn.init.uniform_(m.bias, a=-0.1, b=0.1)
    #     # m.weight.data.normal_(0.0, 0.02)
    #     # m.bias.data.normal_(0.0, 0.02)
    # if type(m) == nn.Conv2d:
    #     nn.init.uniform_(m.weight, a=-0.1, b=0.1)  # 0.02 # 0.001
    #     nn.init.uniform_(m.bias, a=-0.1, b=0.1)


def train(viz, model, dataloader, train_win_mse, logIter, optimiser, criterion, train_global_idx):
    model.train()
    for step, (fmri, fmaps) in enumerate(dataloader):
        # model.zero_grad()
        optimiser.zero_grad()
        fmri = fmri.cuda()
        fmaps = fmaps.cuda()

        out = model(fmri)
        mseLoss = criterion(out, fmaps)
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
        for step, (fmri, fmap) in enumerate(test_dataloader):
            fmri = fmri.cuda()
            fmap = fmap.cuda()

            out = model(fmri)
            mseLoss = criterion(out, fmap)
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


def train_pipline(viz, fmri_key, init_weights, epochs, lr, weight_decay, logIterval, drawline, rois, split_point,
                  fmap_start, fmap_end, batch_size, num_workers, i_dim, o_dim, subject, normalize, saveModel,
                  showRegressedFmap, stIdx):
    # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
    # model = LinearConvRegresssionModel(i_dim, o_dim).cuda()
    criterion = nn.MSELoss()  # Vim1_MSE()  #
    # nn.MSELoss()  # Mean Squared Loss
    # optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimiser = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    saveDir = '/data1/home/guangjie/Data/vim1/regressionFeatureMapModelNormalizedFmap/subject_{}/{}_{}_18x18_Adam_bn'.format(
        subject, ''.join(map(str, rois)), str(weight_decay))
    os.makedirs(saveDir, exist_ok=True)
    test_loss_of_frame = []
    for fmap_idx in np.arange(fmap_start, fmap_end):  # range(n_lantent):
        model = LinearRegressionModel(i_dim, o_dim).cuda()
        # optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # model = LinearConvRegresssionModel(i_dim, o_dim).cuda()
        # model.apply(init_weights)
        # "/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat"
        dataset = vim1_fmri_pca_fmap_dataset(
            fmri_file="/data1/home/guangjie/Data/vim1/exprimentData/pca_whiten_resp_1024.hdf5",
            voxel_select_file="/data1/home/guangjie/Project/python/tf-vqvae/vim1_subject_{}_roi_{}_voxel_select.json".format(
                subject, ''.join(map(str, rois))),
            latent_file="/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/zq_of_st_fmap_mm_scaler_15x15_blur5.hdf5",
            fmri_key=fmri_key, fmap_idx=fmap_idx, isTrain=True, rois=rois,
            split_point=split_point, subject=subject, normalize=normalize)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # "/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat"
        test_dataset = vim1_fmri_pca_fmap_dataset(
            fmri_file="/data1/home/guangjie/Data/vim1/exprimentData/pca_whiten_resp_1024.hdf5",
            voxel_select_file="/data1/home/guangjie/Project/python/tf-vqvae/vim1_subject_{}_roi_{}_voxel_select.json".format(
                subject, ''.join(map(str, rois))),
            latent_file="/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/zq_of_st_fmap_mm_scaler_15x15_blur5.hdf5",
            fmri_key=fmri_key, fmap_idx=fmap_idx, isTrain=False, rois=rois,
            split_point=split_point, subject=subject, normalize=normalize)
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
        # test_loss_of_fmap = []
        last_test_loss = 100
        val_dataset = vim1_fmri_pca_fmap_dataset(
            fmri_file="/data1/home/guangjie/Data/vim1/exprimentData/pca_whiten_resp_1024.hdf5",
            voxel_select_file="/data1/home/guangjie/Project/python/tf-vqvae/vim1_subject_{}_roi_{}_voxel_select.json".format(
                subject, ''.join(map(str, rois))),
            latent_file="/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/zq_of_st_fmap_mm_scaler_15x15_blur5.hdf5",
            fmri_key=fmri_key, fmap_idx=fmap_idx, isTrain=False, rois=rois,
            split_point=0, subject=subject, normalize=normalize)
        scaler = preprocessing.MinMaxScaler()
        fmris, fmaps = val_dataset[stIdx]
        fmaps = np.clip(scaler.fit_transform(fmaps.transpose()).transpose(), 0, 1).reshape((-1, 1, 15, 15))
        val_fmaps = [fmaps]
        for ep in range(epochs):
            train(viz, model, dataloader, train_win_mse, logIterval, optimiser, criterion,
                  train_global_idx)
            test_loss = test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx)
            print('fmap_{}_epoch_{}_test_loss: {}'.format(fmap_idx, ep, test_loss))
            # test_loss_of_fmap.append(test_loss.cpu().item())
            # if test_loss > last_test_loss:
            #     break
            # else:
            #     last_test_loss = test_loss
            # todo 调整 lr
            if ep > epochs // 2:
                with torch.no_grad():
                    predict_fmaps = model(torch.as_tensor(fmris).cuda()).squeeze()
                    predict_fmaps = np.clip(
                        scaler.fit_transform(predict_fmaps.cpu().detach().numpy().transpose()).transpose(), 0,
                        1).reshape(
                        (-1, 1, 15, 15))
                    val_fmaps.append(predict_fmaps)
                # viz.images(fmaps)
                # viz.images(predict_fmaps)
        viz.images(np.concatenate(val_fmaps, axis=0), nrow=fmaps.shape[0])
        # test_loss_of_frame.append(test_loss_of_fmap)
        del train_win_mse
        del test_win_mse
        if saveModel:
            torch.save(model.state_dict(),
                       os.path.join(saveDir, "subject_{}_regression_model_roi_{}_i_{}_o_{}_fmap_{}.pth".format(
                           subject, ''.join(map(str, rois)), i_dim, o_dim, fmap_idx)))
        # if showRegressedFmap:
        #     dataset = vim1_fmri_pca_fmap_dataset(
        #         fmri_file="/data1/home/guangjie/Data/vim1/exprimentData/pca_resp_1024.hdf5",
        #         voxel_select_file="/data1/home/guangjie/Project/python/tf-vqvae/vim1_subject_{}_roi_{}_voxel_select.json".format(
        #             subject, ''.join(map(str, rois))),
        #         latent_file="/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/zq_of_st_fmap_mm_scaler_18x18_blur5.hdf5",
        #         fmri_key=fmri_key, fmap_idx=fmap_idx, isTrain=False, rois=rois,
        #         split_point=0, subject=subject, normalize=normalize)
        #     with torch.no_grad():
        #         scaler = preprocessing.MinMaxScaler()
        #         fmris, fmaps = dataset[stIdx]
        #         predict_fmaps = model(torch.as_tensor(fmris).cuda()).squeeze()
        #         predict_fmaps = np.clip(
        #             scaler.fit_transform(predict_fmaps.cpu().detach().numpy().transpose()).transpose(), 0, 1).reshape(
        #             (-1, 1, 18, 18))
        #         fmaps = np.clip(scaler.fit_transform(fmaps.transpose()).transpose(), 0, 1).reshape((-1, 1, 18, 18))
        #         # viz.images(fmaps)
        #         # viz.images(predict_fmaps)
        #         viz.images(np.concatenate([fmaps, predict_fmaps], axis=0), nrow=fmaps.shape[0])
    return test_loss_of_frame


def apply_regression_to_fmri_concatenate(dt_key, rois, subject, n_fmap, model_in_dim, model_out_dim, wd,
                                         normalize, postfix):
    model_dir = '/data1/home/guangjie/Data/vim1/regressionFeatureMapModelNormalizedFmap/subject_{}/{}_{}_18x18_Adam_bn'.format(
        subject, ''.join(map(str, rois)), str(wd))
    save_dir = "/data1/home/guangjie/Data/vim1/regressed_feature_map/subject_{}/{}".format(
        subject, dt_key)
    os.makedirs(save_dir, exist_ok=True)
    # model = NonLinearRegressionModel(model_in_dim, model_out_dim).cuda()
    # model = LinearRegressionModel(model_in_dim, model_out_dim).cuda()
    model = LinearConvRegresssionModel(model_in_dim, model_out_dim).cuda()
    # mean, std = get_vim2_fmri_mean_std(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5",
    #     'rt')  # todo 归一化方式
    dataset = vim1_val_fmri_dataset(fmri_file="/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat",
                                    voxel_select_file="/data1/home/guangjie/Project/python/tf-vqvae/vim1_subject_{}_roi_{}_voxel_select.json".format(
                                        subject, ''.join(map(str, rois))), fmri_key=dt_key, rois=rois, subject=subject,
                                    normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    with h5py.File(os.path.join(save_dir, "subject_{}_{}_roi_{}_regressed_fmap_all_wd_{}_normalizedFmap_{}.hdf5".format(
            subject, dt_key, ''.join(map(str, rois)), wd, postfix)), 'w') as sf:
        latent = sf.create_dataset('latent', shape=(len(dataset), 32, 32, 128), chunks=True)
        for fmap_idx in range(n_fmap):
            model.load_state_dict(
                torch.load(
                    os.path.join(model_dir, "subject_{}_regression_model_roi_{}_i_{}_o_{}_fmap_{}.pth".format(
                        subject, ''.join(map(str, rois)), model_in_dim, model_out_dim, fmap_idx))))
            with torch.no_grad():
                begin_idx = 0
                model.eval()
                for step, fmri in enumerate(dataloader):
                    out = model(fmri.cuda())
                    end_idx = begin_idx + len(out)
                    out = out.reshape(len(out), 18, 18).cpu().numpy()
                    res_out = np.zeros((len(out), 32, 32), dtype=np.float32)
                    for i, fmp in enumerate(out):
                        res_out[i] = cv2.resize(fmp, (32, 32))
                    latent[begin_idx:end_idx, :, :, fmap_idx] = res_out
                    begin_idx = end_idx
            print(fmap_idx)


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


def eval_model_performance(rois, batch_size, num_workers, i_dim, o_dim, subject, normalize):
    model = LinearRegressionModel(i_dim, o_dim).cuda()
    criterion = nn.MSELoss()  # Mean Squared Loss
    fmaps_loss = []
    for fmap_idx in range(128):
        test_dataset = vim1_fmri_k_dataset(fmri_file="/data1/home/guangjie/Data/vim-1/EstimatedResponses.mat",
                                           voxel_select_file="/data1/home/guangjie/Project/python/tf-vqvae/vim1_subject_{}_roi_{}_voxel_select.json".format(
                                               subject, ''.join(map(str, rois))),
                                           k_file="/data1/home/guangjie/Data/vim1/exprimentData/extract_from_vqvae/k_from_vqvae_st.hdf5",
                                           embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
                                           fmri_key='dataTrnS{}'.format(subject), fmap_idx=fmap_idx, isTrain=False,
                                           rois=rois, split_point=1700, subject=subject)
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model.load_state_dict(torch.load(
            "/data1/home/guangjie/Data/vim1/regressionFeatureMapModel{}/subject_{}/{}_{}/subject_{}_regression_model_roi_{}_i_{}_o_{}_fmap_{}.pth".format(
                'Normalize' if normalize else '', subject, ''.join(map(str, rois)), str(weight_decay), subject,
                ''.join(map(str, rois)), i_dim, o_dim, fmap_idx)))
        model.eval()
        fmap_loss = []
        for step, (fmri, fmap) in enumerate(dataloader):
            fmri = fmri.cuda()
            fmap = fmap.cuda()
            out = model(fmri)
            loss = criterion(out, fmap)
            fmap_loss.append(loss.cpu().item())
        fmaps_loss.append(np.mean(fmap_loss))
        print(fmap_idx)
    return fmaps_loss


if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='fmap regression')
    assert viz.check_connection(timeout_seconds=3)
    torch.manual_seed(7)

    lr = 0.00005  # 0.00006  # best:0.001
    weight_decay = 0.5  # 1  # 0.03  # best:0.01 todo 调整此参数，改变test loss 随train loss 下降的程度
    epochs = 500  # best:200 50
    logIterval = 30
    subject = 1
    i_dim = 1024  # 5167  # 2333#928  # 1535  # 5965# 3377 #  # 7500  # 5965  # 2084  # 7500#5965  # 3377  # 1294  # 3377# 2588  # 5965  # 1789#1789#2084 #2588  #
    # 1294  # 4321  #   6310  # 2771  # 3539  # 893  # 4854 frame_0:4917 4917  #4828  #Subject3V12:2471    Subject3V34:2383 subject3ips:893
    o_dim = 225  # 324  # 529  # 23*23 400  #
    n_frames = 15
    # todo frame_1 Adam w_d 0.1
    # show_regression_performance_all()
    # for i in range(1024):
    # show_regression_performance(model_in_dim=i_dim, model_out_dim=o_dim, frame_idx=0, latent_idx=0)
    frame_idx = 1
    fmap_start = 5
    fmap_end = 6
    # with open("testlosslog/for_feature_map/test_loss_{}_{}_wd_{}.json".format(fmap_start, fmap_end, weight_decay),
    #           'w') as fp:
    lossdict = train_pipline(viz, 'dataTrnS1', init_weights, epochs, lr, weight_decay, logIterval, drawline=True,
                             rois=[1, 2, 3, 4, 5, 6], split_point=1700, fmap_start=fmap_start, fmap_end=fmap_end,
                             batch_size=128, num_workers=0, i_dim=i_dim, o_dim=o_dim, subject=1, normalize=False,
                             saveModel=False, showRegressedFmap=True,
                             stIdx=[0, 3, 4, 5, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708])
    # apply_regression_to_fmri_concatenate(dt_key='dataTrnS1', rois=[6], subject=1, n_fmap=128,
    #                                      model_in_dim=i_dim, model_out_dim=o_dim, wd=weight_decay, normalize=False,
    #                                      postfix='no_mm_18x18_blur3_nolinearConv_Adam_bn_res_32x32')
    # for (roi, i) in [(1, 1294), (2, 2084), (3, 1789), (6, 1535)]:
    #     apply_regression_to_fmri_concatenate(dt_key='dataTrnS1', rois=[roi], subject=1, n_fmap=128,
    #                                          model_in_dim=i, model_out_dim=o_dim, wd=weight_decay)
    # performanceList = []
    # for (roi, i) in [(1, 1294), (2, 2084), (3, 1789), (6, 1535)]:
    # performance = eval_model_performance(rois=[1, 2,3,4,5], batch_size=50, num_workers=0, i_dim=5965,
    #                                      o_dim=1024, subject=1, normalize=False)
    # #     performanceList.append(performance)
    # # print(performanceList)
    # with open('performance_roi_12345_wd_{}.json'.format(str(0.005)), 'w') as fp:
    #     json.dump(performance, fp)
    # print('end')
