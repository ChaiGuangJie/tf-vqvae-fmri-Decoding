import torch
import torch.nn as nn
from MyDataset import fmri_dataset, fmri_fmap_dataset, fmri_fmap_slice_dataset
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


def train(viz, model, dataloader, train_win_mse, logIter, optimiser, criterion, train_global_idx, frame_idx, fmap_idx,
          slice_idx):
    model.train()
    for step, (fmri, slice) in enumerate(dataloader):
        # model.zero_grad()
        optimiser.zero_grad()
        fmri = fmri.cuda()
        slice = slice.cuda()

        out = model(fmri)
        mseLoss = criterion(out, slice)
        loss = mseLoss
        loss.backward()

        optimiser.step()

        if step % logIter == 0:
            if train_win_mse:
                viz.line(Y=mseLoss.view(1), X=train_global_idx, win=train_win_mse, update="append",
                         opts={'title': 'frame {} train mse loss of fmap {} slice {}'.format(frame_idx, fmap_idx,
                                                                                             slice_idx)})
                train_global_idx += 1
                print('step_{}_train_loss : {}'.format(step, loss.item()))


def test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx, frame_idx, fmap_idx,
         slice_idx):
    model.eval()
    # test_global_idx = np.array([0])
    with torch.no_grad():
        mse_loss_list = []
        # dist_loss_list = []
        for step, (fmri, slice) in enumerate(test_dataloader):
            fmri = fmri.cuda()
            slice = slice.cuda()
            out = model(fmri)
            mseLoss = criterion(out, slice)
            # loss = mseLoss
            mse_loss_list.append(mseLoss)
            # dist_loss_list.append(distLoss)
        mse_mean_loss = sum(mse_loss_list) / len(mse_loss_list)
        # dist_mean_loss = sum(dist_loss_list) / len(dist_loss_list)
        if test_win_mse:
            viz.line(Y=mse_mean_loss.view(1), X=test_global_idx, win=test_win_mse, update="append",
                     opts={'title': 'frame {} test mse loss of fmap {} slice {}'.format(frame_idx, fmap_idx,
                                                                                        slice_idx)})
            test_global_idx += 1
            print('test_mse_loss : {}'.format(mse_mean_loss.item()))
        return mse_mean_loss


def train_one_slice_of_fmap(viz, frame_idx, fmap_idx, slice_idx, epochs, lr, weight_decay, logIterval, batch_size,
                            num_workers, drawline, rootDir, i_dim, o_dim, init_weights):
    model = LinearRegressionModel(i_dim, o_dim).cuda()
    # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
    model.apply(init_weights)
    # model = NonLinearRegressionModel(i_dim, o_dim).cuda()
    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent

    # saveDir = '/data1/home/guangjie/Data/vim-2-gallant/regressionSliceFeatureMapModel/subject_{}/frame_{}/slice_{}'.format(
    #     subject, frame_idx, slice_idx)
    os.makedirs(rootDir, exist_ok=True)
    test_loss_of_slice_fmap = []
    model.apply(init_weights)

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
    # todo mean std
    dataset = fmri_fmap_slice_dataset(
        fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_train.hdf5",
        k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_train.hdf5",
        embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", fmri_key='rt',
        frame_idx=frame_idx, fmap_idx=fmap_idx, slice_idx=slice_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataset = fmri_fmap_slice_dataset(
        fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_test.hdf5",
        k_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/k_from_vqvae_st_test.hdf5",
        embeds_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5", fmri_key='rt',
        frame_idx=frame_idx, fmap_idx=fmap_idx, slice_idx=slice_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loss_of_slice = []
    for ep in range(epochs):
        train(viz, model, dataloader, train_win_mse, logIterval, optimiser, criterion,
              train_global_idx, frame_idx, fmap_idx, slice_idx)
        test_loss = test(viz, model, test_dataloader, test_win_mse, criterion, test_global_idx, frame_idx, fmap_idx,
                         slice_idx)
        print('frame_{}_fmap_{}_slice_{}_epoch_{}_test_loss: {}'.format(frame_idx, fmap_idx, slice_idx, ep, test_loss))
        test_loss_of_slice.append(test_loss.cpu().item())
        # todo 调整 lr

    del train_win_mse
    del test_win_mse
    torch.save(model.state_dict(),
               os.path.join(rootDir, "frame_{}_fmap_{}_slice_{}_regression_model.pth".format(
                   frame_idx, fmap_idx, slice_idx)))
    return test_loss_of_slice


def train_slices_of_one_fmap(viz, slice_start, slice_end, frame_idx, fmap_idx, epochs, lr, weight_decay, logIterval,
                             batch_size, num_workers, drawline, rootDir, i_dim, o_dim, init_weights):
    rootDir = os.path.join(rootDir, 'fmap_{}'.format(fmap_idx))
    test_loss_of_slices = []
    for slice in np.arange(slice_start, slice_end):
        loss = train_one_slice_of_fmap(viz=viz, frame_idx=frame_idx, fmap_idx=fmap_idx, slice_idx=slice, epochs=epochs,
                                       lr=lr, weight_decay=weight_decay, logIterval=logIterval, batch_size=batch_size,
                                       num_workers=num_workers, drawline=drawline, rootDir=rootDir, i_dim=i_dim,
                                       o_dim=o_dim, init_weights=init_weights)
        test_loss_of_slices.append(loss)
    return test_loss_of_slices


def train_one_frame(viz, init_weights, epochs, lr, weight_decay, logIterval, drawline, rootDir, frame_idx,
                    fmap_start, fmap_end, slice_start, slice_end, batch_size, num_workers, i_dim, o_dim):
    rootDir = os.path.join(rootDir, 'frame_{}'.format(frame_idx))
    # os.makedirs(rootDir, exist_ok=True) #后面需要保存文件的时候再验证？
    test_loss_of_fmaps = []
    for fmap_idx in np.arange(fmap_start, fmap_end):  # range(n_lantent):
        loss = train_slices_of_one_fmap(viz=viz, slice_start=slice_start, slice_end=slice_end, frame_idx=frame_idx,
                                        fmap_idx=fmap_idx, epochs=epochs, lr=lr, weight_decay=weight_decay,
                                        logIterval=logIterval, batch_size=batch_size, num_workers=num_workers,
                                        drawline=drawline, rootDir=rootDir, i_dim=i_dim, o_dim=o_dim,
                                        init_weights=init_weights)
        test_loss_of_fmaps.append(loss)
    return test_loss_of_fmaps


def apply_regression_to_fmri(dt_key, frame_idx, n_fmap, n_slice, model_in_dim, model_out_dim, subject):
    save_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressed_zq_of_vqvae_by_feature_map_slice/subject_{}/{}/frame_{}".format(
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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    with h5py.File(os.path.join(save_dir, "subject_{}_frame_{}_ze_fmap_all.hdf5".format(subject, frame_idx)),
                   'w') as sf:  # todo zq
        fmap = sf.create_dataset('latent', shape=(len(dataset), 32, 32, 128))
        for fmap_idx in range(n_fmap):
            model_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressionFeatureMapSliceModel/subject_{}/frame_{}/fmap_{}".format(
                subject, frame_idx, fmap_idx)
            for slice_idx in range(n_slice):
                model.load_state_dict(
                    torch.load(os.path.join(model_dir,
                                            "frame_{}_fmap_{}_slice_{}_regression_model.pth".format(frame_idx, fmap_idx,
                                                                                                    slice_idx))))
                with torch.no_grad():
                    begin_idx = 0
                    model.eval()
                    for step, fmri in enumerate(dataloader):
                        out = model(fmri.cuda())  # 32
                        end_idx = begin_idx + len(out)
                        fmap[begin_idx:end_idx, slice_idx, :, fmap_idx] = out.cpu().numpy()  # todo :,slice_idx
                        begin_idx = end_idx
            print(fmap_idx)


def train_frames(viz, i_dim, o_dim, lr, weight_decay, init_weights, epochs, logIterval, drawline, rootDir, frame_start,
                 frame_end, fmap_start, fmap_end, slice_start, slice_end, batch_size, num_workers, subject):
    rootDir = os.path.join(rootDir, "subject_{}".format(subject))
    testloss_of_all_frames = {}
    for frame_idx in np.arange(frame_start, frame_end):
        print('{} frame begin:'.format(frame_idx))
        test_loss_one_frame = train_one_frame(viz, init_weights, epochs, lr, weight_decay, logIterval,
                                              drawline=drawline, rootDir=rootDir, frame_idx=frame_idx,
                                              fmap_start=fmap_start, fmap_end=fmap_end, slice_start=slice_start,
                                              slice_end=slice_end, batch_size=batch_size, num_workers=num_workers,
                                              i_dim=i_dim, o_dim=o_dim)
        testloss_of_all_frames[str(frame_idx)] = test_loss_one_frame
    return testloss_of_all_frames


if __name__ == '__main__':
    viz = Visdom(server="http://localhost", env='slice regression')
    assert viz.check_connection(timeout_seconds=3)
    torch.manual_seed(7)

    lr = 0.005  # best:0.2
    weight_decay = 0.05  # best:0.01 todo 调整此参数，改变test loss 随train loss 下降的程度
    epochs = 80  # best:200 50
    logIterval = 30
    subject = 1
    i_dim = 4854
    o_dim = 32
    n_frames = 15
    # todo frame_1 Adam w_d 0.1
    # show_regression_performance_all()
    # for i in range(1024):
    # show_regression_performance(model_in_dim=i_dim, model_out_dim=o_dim, frame_idx=0, latent_idx=0)
    fmap_start = 120
    fmap_end = 128
    # with open("testlosslog/test_loss_{}_{}_wd_{}_2.json".format(fmap_start, fmap_end, weight_decay), 'w') as fp:
    #     lossdict = train_frames(viz, i_dim, o_dim, lr, weight_decay, init_weights, epochs, logIterval, drawline=False,
    #                             rootDir="/data1/home/guangjie/Data/vim-2-gallant/regressionFeatureMapSlice2Model/",
    #                             frame_start=0, frame_end=1, fmap_start=fmap_start, fmap_end=fmap_end, slice_start=0,
    #                             slice_end=32, batch_size=128, num_workers=0, subject=1)
    #     json.dump({"loss": lossdict}, fp)

    apply_regression_to_fmri(dt_key='rv', frame_idx=0, n_fmap=128, n_slice=32, model_in_dim=i_dim, model_out_dim=o_dim,
                             subject=1)
    print('end')