import torch
import torch.nn as nn
from MyDataset import fmri_vector_dataset, get_vim2_fmri_mean_std, fmri_dataset
import h5py
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np


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
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        # if m.bias:
        nn.init.uniform_(m.bias, a=-0.1, b=0.1)
        # m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.normal_(0.0, 0.02)


def train(viz, model, dataloader, train_win, logIter, optimiser, criterion, train_global_idx):
    model.train()
    for step, (fmri, vector) in enumerate(dataloader):
        # model.zero_grad()
        optimiser.zero_grad()
        fmri = fmri.cuda()
        vector = vector.cuda()

        out = model(fmri)
        loss = criterion(out, vector)
        loss.backward()

        optimiser.step()

        if step % logIter == 0:
            if train_win:
                viz.line(Y=loss.view(1), X=train_global_idx, win=train_win, update="append",
                         opts={'title': 'train loss'})
                train_global_idx += 1
                print('step_{}_train_loss : {}'.format(step, loss.item()))


def test(viz, model, test_dataloader, test_win, criterion, test_global_idx):
    model.eval()
    # test_global_idx = np.array([0])
    with torch.no_grad():
        loss_list = []
        for step, (fmri, vector) in enumerate(test_dataloader):
            fmri = fmri.cuda()
            vector = vector.cuda()

            out = model(fmri)
            loss = criterion(out, vector)

            loss_list.append(loss)

        mean_loss = sum(loss_list) / len(loss_list)
        if test_win:
            viz.line(Y=mean_loss.view(1), X=test_global_idx, win=test_win, update="append", opts={'title': 'test loss'})
            test_global_idx += 1
            print('test_loss : {}'.format(mean_loss.item()))
        return mean_loss


def train_one_frame(viz, model, init_weights, criterion, optimiser, mean, std, epochs, logIterval,
                    drawline=True, frame_idx=0, batch_size=128, num_workers=4, i_dim=4917, o_dim=128, n_lantent=1024,
                    subject=1):
    # if not os.path.exists(
    #         '/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_{}/frame_{}'.format(subject, frame_idx)):
    os.makedirs(
        '/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_{}/frame_{}'.format(subject, frame_idx),
        exist_ok=True)
    for latent_idx in np.arange(0, 100):  # range(n_lantent):
        model.apply(init_weights)
        dataset = fmri_vector_dataset(
            fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_rv_rva0.hdf5".format(
                subject), zq_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_st.hdf5", mean=mean,
            std=std, fmri_key='rt', frame_idx=frame_idx, latent_idx=latent_idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = fmri_vector_dataset(
            fmri_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject{}_v1234_rt_rv_rva0.hdf5".format(
                subject), zq_file="/data1/home/guangjie/Data/vim-2-gallant/myOrig/zq_from_vqvae_sv.hdf5", mean=mean,
            std=std, fmri_key='rv', frame_idx=frame_idx, latent_idx=latent_idx)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        train_global_idx = np.array([0])
        test_global_idx = np.array([0])

        if drawline:
            train_win = viz.line(Y=np.array([0]))
            test_win = viz.line(Y=np.array([0]))
        else:
            train_win = None
            test_win = None

        for ep in range(epochs):
            train(viz, model, dataloader, train_win, logIterval, optimiser, criterion, train_global_idx)
            test_loss = test(viz, model, test_dataloader, test_win, criterion, test_global_idx)
            print('frame_{}_latent_{}_epoch_{}_test_loss: {}'.format(frame_idx, latent_idx, ep, test_loss))
        del train_win
        del test_win
        torch.save(model.state_dict(),
                   '/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_{}/frame_{}/subject_{}_regression_model_i_{}_o_{}_latent_{}.pth'.format(
                       subject, frame_idx, subject, i_dim, o_dim, latent_idx))


def apply_regression_to_fmri(dt_key, frame_idx=0, subject=1, n_lantent=1024, model_in_dim=4917, model_out_dim=128,
                             dim_lantent=128):
    model_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_{}/frame_{}".format(subject, frame_idx)
    save_dir = "/data1/home/guangjie/Data/vim-2-gallant/regressed_ze_of_vqvae/subject_{}/{}/frame_{}".format(subject,
                                                                                                             dt_key,
                                                                                                             frame_idx)
    os.makedirs(save_dir, exist_ok=True)
    model = LinearRegressionModel(model_in_dim, model_out_dim).cuda()
    # mean, std = get_vim2_fmri_mean_std(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5",
    #     'rt')  # todo 归一化方式
    mean, std = None, None
    dataset = fmri_dataset(
        "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5",
        mean, std, dt_key)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
    for lantent_idx in range(n_lantent):
        # 逐次加载单个模型，抽取每张图片对应ze的第lantent_idx个隐含表征
        model.load_state_dict(
            torch.load(os.path.join(model_dir,
                                    "subject_{}_regression_model_i_{}_o_{}_latent_{}.pth".format(subject, model_in_dim,
                                                                                                 model_out_dim,
                                                                                                 lantent_idx))))
        sf = h5py.File(os.path.join(save_dir, "subject_{}_frame_{}_ze_latent_{}.hdf5".format(
            subject, frame_idx, lantent_idx)), 'w')
        latent = sf.create_dataset('latent', shape=(len(dataset), dim_lantent))
        with torch.no_grad():
            begin_idx = 0
            for step, data in enumerate(dataloader):
                out = model(data.cuda())
                end_idx = begin_idx + len(out)
                latent[begin_idx:end_idx] = out.cpu().numpy()  # 需要cpu().numpy()?
                begin_idx = end_idx
        sf.close()
        print(lantent_idx)


def train_all_frame(viz, i_dim, o_dim, lr, init_weights, epochs, logIterval):
    model = LinearRegressionModel(i_dim, o_dim).cuda()

    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)  # Stochastic Gradient Descent

    # with h5py.File("/data1/home/guangjie/Data/vim-2-gallant/myOrig/imagenet128_embeds_from_vqvae.hdf5",
    #                'r') as ebdf:
    #     embeds = ebdf['embeds'][:]
    # mean, std = get_vim2_fmri_mean_std(
    #     "/data1/home/guangjie/Data/vim-2-gallant/myOrig/VoxelResponses_subject1_v1234_rt_rv_rva0.hdf5", dt_key='rt')
    mean, std = None, None
    # cuda:3 [12, 13, 14]
    for frame_idx in [1]:
        print('{} frame begin:'.format(frame_idx))
        train_one_frame(viz, model, init_weights, criterion, optimiser, mean, std, epochs, logIterval,
                        drawline=False, frame_idx=frame_idx, batch_size=256, num_workers=2, i_dim=i_dim, o_dim=o_dim,
                        n_lantent=1024, subject=1)
    print('end')


def show_regression_performance(model_in_dim, model_out_dim, frame_idx=0, latent_idx=0, time_step=15):
    model = LinearRegressionModel(model_in_dim, model_out_dim).cuda()
    model.load_state_dict(
        torch.load(
            "/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_1/frame_0/subject_1_regression_model_i_4917_o_128_latent_0.pth"))
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
                    for i in range(10):
                        rt = torch.from_numpy(rt_data[:, i]).cuda()
                        o_rt = model(rt)
                        zq_st_frame_latent = torch.from_numpy(
                            st_zq_data[frame_idx + i * time_step].reshape(1024, 128)[latent_idx]).cuda()
                        rt_loss = criterion(o_rt, zq_st_frame_latent)

                        rv = torch.from_numpy(rv_data[:, i]).cuda()
                        o_rv = model(rv)
                        zq_sv_frame_latent = torch.from_numpy(
                            sv_zq_data[frame_idx + i * time_step].reshape(1024, 128)[latent_idx]).cuda()
                        rv_loss = criterion(o_rv, zq_sv_frame_latent)

                        print(i, ' rt_loss:', rt_loss, 'rv_loss:', rv_loss)

                        # rva0 = rva0_data[:, i].cuda()
                        # o_rva0 = model(rva0)


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    viz = Visdom(server="http://localhost", env='regression')
    assert viz.check_connection(timeout_seconds=3)
    torch.manual_seed(7)

    lr = 0.25
    epochs = 200  # best:200
    logIterval = 30
    subject = 1
    i_dim = 4917
    o_dim = 128
    n_frames = 15

    show_regression_performance(model_in_dim=i_dim, model_out_dim=o_dim)
    # train_all_frame(viz, i_dim, o_dim, lr, init_weights, epochs, logIterval)
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
