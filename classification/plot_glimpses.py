import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

from utils import denormalize, bounding_box


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str, default="./plots/ram_6_8x8_2/",
                     help="path to directory containing pickle dumps")
    arg.add_argument("--epoch", type=int, required=True,
                     help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args['plot_dir'], args['epoch']


def main(plot_dir, epoch):

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # read in pickle files
    glimpses = pickle.load(
        open(plot_dir + "g_{}.p".format(epoch), "rb")
    )
    locations = pickle.load(
        open(plot_dir + "l_{}.p".format(epoch), "rb")
    )
    predicted_results = pickle.load(
        open(plot_dir + "pr_{}.p".format(epoch), "rb"))
    org_res = pickle.load(
        open(plot_dir + "r_{}.p".format(epoch), "rb"))
    # pdb.set_trace()

    # # grab useful params
    size = int(plot_dir.split('_')[2][0])
    num_anims = len(locations)
    num_cols = len(locations)
    img_shape = glimpses[0].shape[-1]
    # pdb.set_trace()

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    # # fig, axs = plt.subplots(nrows=2, ncols=num_anims)
    def showAll(i) :

        fig, axs = plt.subplots(nrows=1, ncols=num_cols)
        # # fig.set_dpi(100)
        # pdb.set_trace()
        # # plot base image
        im1 = np.transpose(glimpses[0][i], (1,2,0))
        color = 'r'
        for j, ax in enumerate(axs.flat):
            if j >= num_cols:
                # ax.imshow(imageWise[i][j%num_cols], cmap="Greys_r")
                im = np.transpose(glimpses[0][i], (1,2,0))
                ax.imshow(im)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            else :
                # ax.imshow(glimpses[i], cmap="Greys_r")
                # ax.imshow(glimpses[i])
                ax.imshow(im1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                c = coords[j][i]
                rect = bounding_box(
                    c[0], c[1], size, color
                )
                ax.add_patch(rect)

        plt.show()

    for i in range(glimpses[0].shape[0]):
        showAll(i)
        print(org_res[0][i], predicted_results[0][i])


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
