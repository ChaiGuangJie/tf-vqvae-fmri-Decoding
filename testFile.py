import os

r_list = []
for i in range(1024):
    r_list.append(os.path.exists(
        '/data1/home/guangjie/Data/vim-2-gallant/regressionModel/subject_1/frame_0/subject_1_regression_model_i_4917_o_128_latent_{}.pth'.format(
            i)))

print(sum(r_list))
