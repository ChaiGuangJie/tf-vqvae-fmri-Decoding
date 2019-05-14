import torch.nn as nn
import torch
import os
import numpy as np
from MyDataset import fmri_k_dataset, fmri_dataset
from torch.utils.data import DataLoader
from visdom import Visdom
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
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
    for step, (fmri, vector) in enumerate(dataloader):
        # model.zero_grad()
        optimiser.zero_grad()
        fmri = fmri.cuda()
        vector = vector.cuda()

        out = model(fmri)
        mseLoss = criterion(out, vector)
        loss = mseLoss  # + adjust_distLoss
        loss.backward()

        optimiser.step()

        if step % logIter == 0:
            if train_win_mse:
                # viz.line(Y=torch.cat([mseLoss.view(1),distLoss.view(1)]).view(1,2), X=np.column_stack((train_global_idx,train_global_idx)), win=train_win,
                #          update="append",opts={'title': 'train loss'})
                viz.line(Y=mseLoss.view(1), X=train_global_idx, win=train_win_mse, update="append",
                         opts={'title': 'train mse loss'})
                # viz.line(Y=distLoss.view(1), X=train_global_idx, win=train_win_dist, update="append",
                #          opts={'title': 'train dist loss'})
                # torch.cat([mseLoss.view(1),distLoss.view(1)]).view(1,2)
                # np.column_stack((train_global_idx,train_global_idx))
                train_global_idx += 1
                print('step_{}_train_loss : {}'.format(step, loss.item()))


def test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx):
    model.eval()
    # test_global_idx = np.array([0])
    with torch.no_grad():
        mse_loss_list = []
        dist_loss_list = []
        for step, (fmri, vector) in enumerate(test_dataloader):
            fmri = fmri.cuda()
            vector = vector.cuda()

            out = model(fmri)
            mseLoss = criterion(out, vector)
            # loss = mseLoss

            mse_loss_list.append(mseLoss)
            # dist_loss_list.append(distLoss)

        mse_mean_loss = sum(mse_loss_list) / len(mse_loss_list)
        # dist_mean_loss = sum(dist_loss_list) / len(dist_loss_list)
        if test_win_mse:  # and test_win_dist
            viz.line(Y=mse_mean_loss.view(1), X=test_global_idx, win=test_win_mse, update="append",
                     opts={'title': 'test mse loss'})
            # viz.line(Y=dist_mean_loss.view(1), X=test_global_idx, win=test_win_dist, update="append",
            #          opts={'title': 'test dist loss'})
            test_global_idx += 1
            print('test_mse_loss : {}'.format(mse_mean_loss.item()))
        return mse_mean_loss  # todo


def train_all_frame(viz, model, init_weights, criterion, optimiser, epochs, logIterval,
                    drawline=True, frame_idx=0, batch_size=128, num_workers=0, i_dim=4917, o_dim=128, subject=1):
    saveDir = '/data1/home/guangjie/Data/vim-2-gallant/regressionKModel/subject_{}/frame_{}'.format(subject, frame_idx)
    os.makedirs(saveDir, exist_ok=True)
    model.apply(init_weights)

    dataset = fmri_k_dataset(
        fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_train.hdf5",
        k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_train.hdf5", fmri_key='rt',
        frame_idx=frame_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = fmri_k_dataset(
        fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_test.hdf5",
        k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_test.hdf5", fmri_key='rt',
        frame_idx=frame_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    train_global_idx = np.array([0])
    test_global_idx = np.array([0])

    if drawline:
        train_win_mse = viz.line(Y=np.array([0]))
        test_win_mse = viz.line(Y=np.array([0]))
    else:
        train_win_mse = None
        test_win_mse = None

    for ep in range(epochs):
        train(viz, model, dataloader, train_win_mse, logIterval, optimiser, criterion,
              train_global_idx)
        test_loss = test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx)
        print('frame_{}_epoch_{}_test_loss: {}'.format(frame_idx, ep, test_loss))
        # todo 调整 lr

    del train_win_mse
    del test_win_mse

    torch.save(model.state_dict(),
               os.path.join(saveDir,
                            "subject_{}_frame_{}_regression_k_model_i_{}_o_{}.pth".format(subject, frame_idx, i_dim,
                                                                                          o_dim)))


def apply_regression_to_fmri(fmri_file, dt_key, model, frame_idx, subject=1):
    # with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",'r') as ebdf:
    #     embeds = ebdf['embeds']
    saveDir = "/data1/home/guangjie/Data/vim-2-gallant/regressed_k_of_vqvae/subject_{}/{}/".format(subject, dt_key)
    os.makedirs(saveDir, exist_ok=True)
    model.load_state_dict(torch.load(
        "/data1/home/guangjie/Data/vim-2-gallant/regressionKModel/subject_{}/frame_{}/subject_{}_frame_{}_regression_k_model_i_4917_o_1024.pth".format(
            subject, frame_idx, subject, frame_idx)))
    mean, std = None, None
    dataset = fmri_dataset(fmri_file, mean, std, dt_key=dt_key)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    with h5py.File(os.path.join(saveDir, "frame_{}_regressed_k.hdf5".format(frame_idx)), 'w') as sf:
        k = sf.create_dataset('k', shape=(len(dataset), 32, 32), dtype=np.uint16)
        with torch.no_grad():
            model.eval()
            begin_idx = 0
            for step, fmri in enumerate(dataloader):
                out = model(fmri.cuda())
                end_idx = begin_idx + len(out)
                out = torch.clamp(out.view(-1, 32, 32) * 512, 0, 511).cpu().numpy().astype(np.uint16)
                k[begin_idx:end_idx] = out  # todo 限制大小到0-511
                begin_idx = end_idx
                print(step)


if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='regression')
    assert viz.check_connection(timeout_seconds=3)
    torch.manual_seed(7)

    lr = 0.2  # best:0.2
    weight_decay = 0.0001  # best:0.01 todo 调整此参数，改变test loss 随train loss 下降的程度
    epochs = 150  # best:200 50
    logIterval = 30
    subject = 1
    i_dim = 4917
    o_dim = 1024  # 32
    n_frames = 15

    model = LinearRegressionModel(i_dim, o_dim).cuda()

    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent

    train_all_frame(viz, model, init_weights, criterion, optimiser, epochs, logIterval, drawline=True, frame_idx=0,
                    batch_size=64, num_workers=4, i_dim=i_dim, o_dim=o_dim, subject=1)
    # apply_regression_to_fmri(
    #     fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_train.hdf5",
    #     dt_key='rt', model=model, frame_idx=0)
    print('end')
