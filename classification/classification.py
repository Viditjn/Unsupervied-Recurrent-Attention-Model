# import torch
# import torch.nn.functional as F

# from torch.autograd import Variable
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# import os
# import time
# import shutil
# import pickle
# import random
# import pdb

# from tqdm import tqdm
# from utils import AverageMeter
# from model import RecurrentAttention
# from tensorboard_logger import configure, log_value

# def train_classification(self, epoch):
#         losses = AverageMeter()
#         accs = AverageMeter()

#         with tqdm(total=self.num_train) as pbar:
#             for i, (x, y) in enumerate(self.train_loader):
#                 if self.use_gpu:
#                     x, y = x.cuda(), y.cuda()
#                 x, y = Variable(x), Variable(y)
#                 self.load_checkpoint(best=False, epoch = str(10))
#                 # initialize location vector and hidden state
#                 self.batch_size = x.shape[0]
#                 # here check the size of image
#                 # h_t, l_t, prev_l_t = self.reset()

#                 # save images
#                 imgs = []
#                 imgs.append(x[0:9])

#                 # extract the glimpses
#                 locs = []
#                 log_pi = []
#                 baselines = []
#                 pred_imgs = []
#                 img_patch = []

#                 # for t in range(self.num_glimpses - 1):
#                 for t in range(self.num_glimpses):
#                     # forward pass through model
#                     # h_t, l_t, b_t, p = self.model(x, l_t, h_t)
#                     h_t, l_t, b_t, new_img_patch, p = self.model(x, l_t, h_t)
#                     new_img_patch = new_img_patch.view(new_img_patch.shape[0], self.num_channels, self.original_img_size, self.original_img_size)
#                     # pdb.set_trace()
#                     final_image = final_image + new_img_patch
#                     temp = final_image - new_img_patch
#                     # pdb.set_trace()
#                     # self.fill_patch(prev_l_t, new_img_patch, final_image)
#                     prev_l_t = l_t
#                     # store
#                     locs.append(l_t[0:9])
#                     pred_imgs.append(final_image[0:9])
#                     img_patch.append(new_img_patch[0:9])
#                     baselines.append(b_t)
#                     log_pi.append(p)

#                 # last iteration
#                 # h_t, l_t, b_t, log_probas, p = self.model(
#                 #     x, l_t, h_t, last=True
#                 # )
#                 # for i in range(final_image.shape[0]):
#                 #     final_image[i] = (final_image[i] - final_image[i].mean() ) / final_image[i].std()
#                 # log_pi.append(p)
#                 # baselines.append(b_t)
#                 # locs.append(l_t[0:9])

#                 # convert list to tensors and reshape
#                 baselines = torch.stack(baselines).transpose(1, 0)
#                 log_pi = torch.stack(log_pi).transpose(1, 0)

#                 # calculate reward
#                 # predicted = torch.max(log_probas, 1)[1]
#                 # R = (predicted.detach() == y).float()
#                 # R = R.unsqueeze(1).repeat(1, self.num_glimpses)

#                 # find mse for the image formed
#                 R = F.mse_loss(x, final_image)                

#                 # compute losses for differentiable modules
#                 # loss_action = F.nll_loss(log_probas, y)
#                 loss_baseline = F.mse_loss(baselines, R)

#                 # compute reinforce loss
#                 # summed over timesteps and averaged across batch
#                 adjusted_reward = R - baselines.detach()
#                 # adjusted_reward = R
#                 loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
#                 loss_reinforce = torch.mean(loss_reinforce, dim=0)
#                 loss_reinforce = loss_reinforce*loss_reinforce
#                 loss_reinforce = torch.sqrt_(loss_reinforce)

#                 # sum up into a hybrid loss
#                 # loss = loss_action + loss_baseline + loss_reinforce
#                 # loss = loss_baseline + loss_reinforce
#                 loss = R + loss_reinforce
#                 # pdb.set_trace()

#                 # compute accuracy
#                 # correct = (predicted == y).float()
#                 # acc = 100 * (correct.sum() / len(y))

#                 # fake data compute accuracy
#                 # correct = random.randint(0,1)
#                 acc = torch.Tensor([100 * (random.random())])

#                 # store
#                 losses.update(loss.data[0], x.size()[0])
#                 accs.update(acc.data[0], x.size()[0])

#                 # compute gradients and update SGD
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 # measure elapsed time
#                 toc = time.time()
#                 batch_time.update(toc-tic)

#                 pbar.set_description(
#                     (
#                         "{:.1f}s - loss: {:.3f} - R: {:.3f} - reinforce_r: {:.3f}".format(
#                             (toc-tic), loss.data[0], R, loss_reinforce
#                         )
#                     )
#                 )
#                 pbar.update(self.batch_size)
#                 # pdb.set_trace()
#                 # save predicated images
#                 # pred_imgs.append(final_image[:9])

#                 # dump the glimpses and locs
#                 if plot:
#                     if self.use_gpu:
#                         imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
#                         locs = [l.cpu().data.numpy() for l in locs]
#                         pred_imgs = [pg.cpu().data.numpy().squeeze() for pg in pred_imgs]
#                         img_patch = [ipg.cpu().data.numpy().squeeze() for ipg in img_patch]
#                     else:
#                         imgs = [g.data.numpy().squeeze() for g in imgs]
#                         locs = [l.data.numpy() for l in locs]
#                         pred_imgs = [pg.data.numpy().squeeze() for pg in pred_imgs]
#                         img_patch = [ipg.data.numpy().squeeze() for ipg in img_patch]
#                     pickle.dump(
#                         imgs, open(
#                             self.plot_dir + "g_{}.p".format(epoch+1),
#                             "wb"
#                         )
#                     )
#                     pickle.dump(
#                         locs, open(
#                             self.plot_dir + "l_{}.p".format(epoch+1),
#                             "wb"
#                         )
#                     )
#                     pickle.dump(
#                         pred_imgs, open(
#                             self.plot_dir + "pg_{}.p".format(epoch+1),
#                             "wb"
#                         )
#                     )
#                     pickle.dump(
#                         img_patch, open(
#                             self.plot_dir + "ip_{}.p".format(epoch+1),
#                             "wb"
#                         )
#                     )

#                 # log to tensorboard
#                 if self.use_tensorboard:
#                     iteration = epoch*len(self.train_loader) + i
#                     log_value('train_loss', losses.avg, iteration)
#                     log_value('train_acc', accs.avg, iteration)

#             return losses.avg, accs.avg