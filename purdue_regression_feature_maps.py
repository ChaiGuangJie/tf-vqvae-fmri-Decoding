import torch
import torch.nn as nn
from MyDataset import fmri_dataset, fmri_fmap_dataset, purdue_fmri_fmap_dataset, get_purdue_fmri_mean_std
import h5py
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        # self.linear2 = nn.Linear(512, 256)  # 1024 256 128
        self.linear2 = nn.Linear(512, output_dim)  # 1024 256 128
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
        # out = self.activate(self.linear3(out))
        # out = self.activate(self.linear3(out))
        # out = self.activate(self.linear4(out))
        # out = self.activate(self.linear2(out))
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, a=-0.001, b=0.001)
        # if m.bias:
        nn.init.uniform_(m.bias, a=-0.001, b=0.001)
        # m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.normal_(0.0, 0.02)


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


def train_one_frame(viz, init_weights, mean, std, epochs, lr, weight_decay, logIterval, drawline, frame_idx,
                    fmap_start, fmap_end, batch_size=128, num_workers=0, i_dim=4917, o_dim=128, subject=1):
    model = LinearRegressionModel(i_dim, o_dim).cuda()
    # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
    model.apply(init_weights)
    # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent

    saveDir = '/data1/home/guangjie/Data/purdue/exprimentData/regressionFeatureMapModel/subject_{}/frame_{}'.format(
        subject,
        frame_idx)
    os.makedirs(saveDir, exist_ok=True)
    test_loss_of_frame = []
    for fmap_idx in np.arange(fmap_start, fmap_end):  # range(n_lantent):
        model.apply(init_weights)
        # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
        dataset = purdue_fmri_fmap_dataset(
            fmri_file="/data1/home/guangjie/Data/purdue/exprimentData/fMRI/subject1_rt_fmri.hdf5",
            k_file="/data1/home/guangjie/Data/purdue/exprimentData/extract_from_vqvae/k_from_vqvae_st_frame_0.hdf5",
            embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
            fmri_key='rt', frame_idx=frame_idx, fmap_idx=fmap_idx, mean=mean, std=std)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = purdue_fmri_fmap_dataset(
            fmri_file="/data1/home/guangjie/Data/purdue/exprimentData/fMRI/subject1_rv_fmri.hdf5",
            k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_test.hdf5",
            embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
            fmri_key='rv', frame_idx=frame_idx, fmap_idx=fmap_idx, mean=mean, std=std)
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
        test_loss_of_fmap = []
        for ep in range(epochs):
            train(viz, model, dataloader, train_win_mse, logIterval, optimiser, criterion,
                  train_global_idx)
            test_loss = test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx)
            print('frame_{}_fmap_{}_epoch_{}_test_loss: {}'.format(frame_idx, fmap_idx, ep, test_loss))
            test_loss_of_fmap.append(test_loss.cpu().item())
            # todo 调整 lr
        test_loss_of_frame.append(test_loss_of_fmap)
        del train_win_mse
        del test_win_mse
        torch.save(model.state_dict(),
                   os.path.join(saveDir, "subject_{}_frame_{}_regression_model_i_{}_o_{}_fmap_{}.pth".format(
                       subject, frame_idx, i_dim, o_dim, fmap_idx)))
    return test_loss_of_frame


def apply_regression_to_fmri(dt_key, frame_idx, subject=1, n_fmap=128, model_in_dim=4917, model_out_dim=1024,
                             fmap_size=1024):
    model_dir = "/data1/home/guangjie/Data/purdue/exprimentData/regressionFeatureMapModel/subject_{}/frame_{}".format(
        subject, frame_idx)
    save_dir = "/data1/home/guangjie/Data/purdue/exprimentData/regressed_zq_of_vqvae_by_feature_map/subject_{}/{}/frame_{}".format(
        subject, dt_key, frame_idx)
    os.makedirs(save_dir, exist_ok=True)
    # model = NonLinearRegressionModel(model_in_dim, model_out_dim).cuda()
    model = LinearRegressionModel(model_in_dim, model_out_dim).cuda()
    # mean, std = get_vim2_fmri_mean_std(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5",
    #     'rt')  # todo 归一化方式
    mean, std = None, None
    dataset = fmri_dataset(
        "/data1/home/guangjie/Data/purdue/exprimentData/fMRI/subject{}_{}_fmri.hdf5".format(subject, dt_key), mean, std,
        dt_key)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    for fmap_idx in range(n_fmap):
        model.load_state_dict(
            torch.load(os.path.join(model_dir, "subject_{}_frame_{}_regression_model_i_{}_o_{}_fmap_{}.pth".format(
                subject, frame_idx, i_dim, o_dim, fmap_idx))))
        sf = h5py.File(os.path.join(save_dir, "subject_{}_frame_{}_zq_fmap_{}.hdf5".format(
            subject, frame_idx, fmap_idx)), 'w')  # todo zq
        fmap = sf.create_dataset('fmap', shape=(len(dataset), fmap_size))
        with torch.no_grad():
            begin_idx = 0
            model.eval()
            for step, fmri in enumerate(dataloader):
                out = model(fmri.cuda())
                end_idx = begin_idx + len(out)
                fmap[begin_idx:end_idx] = out.cpu().numpy()  # 需要cpu().numpy()?
                begin_idx = end_idx
        sf.close()
        print(fmap_idx)


def train_frames(viz, i_dim, o_dim, lr, weight_decay, init_weights, epochs, logIterval, drawline, frame_start,
                 frame_end, fmap_start, fmap_end, batch_size, num_workers, subject):
    # mean, std = get_vim2_fmri_mean_std(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_train.hdf5", dt_key='rt')
    # mean, std = None, None
    mean, std = get_purdue_fmri_mean_std(
        voxel_train_file="/data1/home/guangjie/Data/purdue/exprimentData/fMRI/subject1_rt_fmri.hdf5", dt_key='rt')
    testloss_of_all_fmaps = {}
    for frame_idx in np.arange(frame_start, frame_end):
        print('{} frame begin:'.format(frame_idx))
        testloss_list = train_one_frame(viz, init_weights, mean, std, epochs, lr, weight_decay, logIterval,
                                        drawline=drawline, frame_idx=frame_idx, fmap_start=fmap_start,
                                        fmap_end=fmap_end, batch_size=batch_size, num_workers=num_workers, i_dim=i_dim,
                                        o_dim=o_dim, subject=subject)
        testloss_of_all_fmaps[str(frame_idx)] = testloss_list
    return testloss_of_all_fmaps

if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='purdue regression 2')
    assert viz.check_connection(timeout_seconds=3)
    torch.manual_seed(7)

    lr = 0.02  # best:0.2
    weight_decay = 0.5  # best:0.1 todo 调整此参数，改变test loss 随train loss 下降的程度
    epochs = 60  # best:200 50
    logIterval = 30
    subject = 1
    i_dim = 8000
    o_dim = 1024
    n_frames = 15
    # todo frame_1 Adam w_d 0.1
    # show_regression_performance_all()
    # for i in range(1024):
    # show_regression_performance(model_in_dim=i_dim, model_out_dim=o_dim, frame_idx=0, latent_idx=0)
    fmap_start = 0
    fmap_end = 128
    with open("testlosslog/purdue_test_loss_{}_{}_wd_{}.json".format(fmap_start, fmap_end, weight_decay), 'w') as fp:
        lossdict = train_frames(viz, i_dim, o_dim, lr, weight_decay, init_weights, epochs, logIterval, drawline=True,
                                frame_start=0, frame_end=1, fmap_start=fmap_start, fmap_end=fmap_end, batch_size=128,
                                num_workers=0, subject=1)
        json.dump({"loss": lossdict}, fp)

    # apply_regression_to_fmri('rv', frame_idx=0, subject=1, n_fmap=128, model_in_dim=i_dim, model_out_dim=o_dim,
    #                          fmap_size=1024)
