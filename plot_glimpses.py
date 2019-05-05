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
    # arg.add_argument("--plot_dir", type=str, required=True,
    #                  help="path to directory containing pickle dumps")
    # arg.add_argument("--epoch", type=int, required=True,
    #                  help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args['plot_dir'], args['epoch']


def main(plot_dir, epoch):

    # read in pickle files
    glimpses = pickle.load(
        open(plot_dir + "g_{}.p".format(epoch), "rb")
    )
    locations = pickle.load(
        open(plot_dir + "l_{}.p".format(epoch), "rb")
    )
    new_image = pickle.load(
        open(plot_dir + "pg_{}.p".format(epoch), "rb"))
    img_patch = pickle.load(
        open(plot_dir + "ip_{}.p".format(epoch), "rb"))
    # pdb.set_trace()

    glimpses = np.concatenate(glimpses)

    imageWise = np.zeros((len(new_image[0]), len(new_image), len(new_image[0][0]), len(new_image[0][0][0]), len(new_image[0][0][0][0])))

    # imageWise = [temp for i in range(new_image[0])]

    for i in range(len(new_image[0])):
        for j in range(len(new_image)):
            imageWise[i][j] = new_image[j][i]

    # # grab useful params
    size = int(plot_dir.split('_')[2][0])
    num_anims = len(locations)
    num_cols = imageWise.shape[1]
    img_shape = imageWise.shape[-1]

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    # # fig, axs = plt.subplots(nrows=2, ncols=num_anims)
    def showAll(i) :

        fig, axs = plt.subplots(nrows=2, ncols=num_cols)
        # # fig.set_dpi(100)
        # pdb.set_trace()
        # # plot base image
        im1 = np.transpose(glimpses[i], (1,2,0))
        color = 'r'
        for j, ax in enumerate(axs.flat):
            if j >= num_cols:
                # ax.imshow(imageWise[i][j%num_cols], cmap="Greys_r")
                im = np.transpose(imageWise[i][j%num_cols], (1,2,0))
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

    for i in range(imageWise.shape[0]):
        showAll(i)
            


    # def updateData(i):
    #     for j, ax in enumerate(axs.flat):

    # def updateData(i):
    #     # plt.figure()
    #     fig, axs = plt.subplots(nrows=1, ncols=num_cols)
    #     for j, ax in enumerate(axs.flat):
    #         ax.imshow(glimpses[j], cmap="Greys_r")
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #     color = 'r'
    #     co = coords[i]
    #     for j, ax in enumerate(axs.flat):
    #         for p in ax.patches:
    #             p.remove()
    #         c = co[j]
    #         rect = bounding_box(
    #             c[0], c[1], size, color
    #         )
    #         ax.add_patch(rect)
    #     # plt.imshow(fig)
    #     # plt.waitforbuttonpress()
    #     # fig1, axs1 = plt.subplots(nrows=1, ncols=num_cols)
        
    #     # pdb.set_trace()
    #     # plt.close()


    # for i in range(num_anims):
    #     updateData(i)
    # fig, axs = plt.subplots(nrows=1, ncols=num_cols)
    # for j, ax in enumerate(axs.flat):
    #     ax.imshow(new_image[j], cmap="Greys_r")
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()

    # animate
    # anim = animation.FuncAnimation(
    #     fig, updateData, frames=num_anims, interval=500, repeat=True
    # )

    # # save as mp4
    # name = plot_dir + 'epoch_{}.mp4'.format(epoch)
    # anim.save(name, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])


# def plot_images(plot_dir, epoch):
#     glimpses = pickle.load(
#         open(plot_dir + "g_{}.p".format(epoch), "rb")
#     )
#     locations = pickle.load(
#         open(plot_dir + "l_{}.p".format(epoch), "rb")
#     )
#     new_image = pickle.load(
#         open(plot_dir + "pg_{}.p".format(epoch), "rb"))

#     glimpses = np.concatenate(glimpses)
#     new_image = np.concatenate(new_image)

#     # grab useful params
#     size = int(plot_dir.split('_')[2][0])
#     num_anims = len(locations)
#     num_imgs = glimpses.shape[0]
#     img_shape = glimpses.shape[1]

#     # denormalize coordinates
#     coords = [denormalize(img_shape, l) for l in locations]
#     co_trans = np.transpose(coords)

#     fig, axs = plt.subplots(nrows=1, ncols=num_imgs)
#     # plot base image
#     for j, ax in enumerate(axs.flat):
#         ax.imshow(glimpses[j], cmap="Greys_r")
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#     def updateData(i):

#         for j, ax
    
    # fig=plt.figure(figsize=(8, 8))
    # columns = 4
    # rows = 5
    # for i in range(1, columns*rows +1):
    #     img = np.random.randint(10, size=(h,w))
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
