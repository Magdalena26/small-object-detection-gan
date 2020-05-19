from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
from model import FasterRCNNVGG16
from utils.config_pcgan import opt
from data.dataset import SmallImageTestDataset, DatasetAugmented
from model.faster_rcnn_vgg16_gan import FasterRCNNVGG16_GAN
import torch.nn as nn
from torch.utils import data as data_
from model.Discriminator import Discriminator
from trainer_pcgan import FasterRCNNTrainer
from utils import array_tool as at
import torch
from utils.eval_tool import eval_detection_voc
import torch.optim as optim
# fix for ulimit
import numpy as np
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
import time
from tqdm import tqdm

torch.cuda.set_device('cuda:2')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[0], rlimit[1]))

# matplotlib.use('agg')

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bike',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'person',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        '''print("shape: ", imgs.shape)'''
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])

        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())

        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    i=0
    for ap in result['ap']:
        print(('AP for {} = {:.4f}'.format(VOC_BBOX_LABEL_NAMES[i], ap)))
        i+=1
    return result


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad_ and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

#
# def clip_grad_norm_(model, max_norm, norm_type=2):
#     r"""Clips gradient norm of an iterable of parameters.
#
#     The norm is computed over all gradients together, as if they were
#     concatenated into a single vector. Gradients are modified in-place.
#
#     Arguments:
#         parameters (Iterable[Tensor]): an iterable of Tensors that will have
#             gradients normalized
#         max_norm (float or int): max norm of the gradients
#         norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
#             infinity norm.
#
#     Returns:
#         Total norm of the parameters (viewed as a single vector).
#     """
#     parameters = model.parameters()
#     parameters = list(filter(lambda p: p.grad is not None, parameters))
#     max_norm = float(max_norm)
#     norm_type = float(norm_type)
#     if norm_type == float('inf'):
#         total_norm = max(p.grad.data.abs().max() for p in parameters)
#     else:
#         total_norm = 0
#         for p in parameters:
#             param_norm = p.grad.data.norm(norm_type)
#             total_norm += param_norm ** norm_type
#         total_norm = total_norm ** (1. / norm_type)
#     clip_coef = max_norm / (total_norm + 1e-6)
#     if clip_coef < 1:
#         for p in parameters:
#             p.grad.data.mul_(clip_coef)
#     return total_norm

#
# def custom_viz(kernels, path=None, cols=4, size=None):
#     N = kernels.shape[0]
#     C = 16
#
#     total_cols = N * C
#     req_cols = cols
#     num_rows = int(np.ceil(total_cols / req_cols))
#     pos = range(1, total_cols + 1)
#
#     fig = plt.figure(1)
#     # fig.tight_layout()
#     k = 0
#     for i in range(kernels.shape[0]):
#         for j in range(16):
#             img = kernels[i][j]
#             ax = fig.add_subplot(num_rows, req_cols, pos[k])
#             ax.imshow(img, cmap='hot')
#             plt.axis('off')
#             k = k + 1
#     # if size:
#     # size_h, size_w = 200, 200
#     # set_size(size_h, size_w, ax)
#     if path:
#         plt.savefig(path, dpi=100)


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_losses(loss_type, losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("epoch " + str(epoch) + " " + loss_type)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('results-gan/losses_voc_generated_%s_%s.jpeg' % (loss_type, str(epoch)))


def save_map(values, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("epoch " + str(epoch) + " test_map")
    plt.plot(values)
    plt.xlabel("epoch")
    plt.ylabel("map")
    plt.savefig('results-gan/map_voc_generated_%s.jpeg' % (str(epoch)))


def train(**kwargs):
    opt._parse(kwargs)

    id_file_dir = 'ImageSets/Main/trainval_big_64.txt'
    img_dir = 'JPEGImages'
    anno_dir = 'AnnotationsBig'
    large_dataset = DatasetAugmented(opt, id_file=id_file_dir, img_dir=img_dir, anno_dir=anno_dir)
    dataloader_large = data_.DataLoader(large_dataset, \
                                        batch_size=1, \
                                        shuffle=True, \
                                        # pin_memory=True,
                                        num_workers=opt.num_workers)

    id_file_dir = 'ImageSets/Main/trainval_pcgan_generated_small.txt'
    img_dir = 'JPEGImagesPCGANGenerated'
    anno_dir = 'AnnotationsPCGANGenerated'

    small_dataset = DatasetAugmented(opt, id_file=id_file_dir, img_dir=img_dir, anno_dir=anno_dir)
    dataloader_small = data_.DataLoader(small_dataset, \
                                        batch_size=1, \
                                        shuffle=True, \
                                        # pin_memory=True,
                                        num_workers=opt.num_workers)

    small_test_dataset = SmallImageTestDataset(opt)
    dataloader_small_test = data_.DataLoader(small_test_dataset, \
                                             batch_size=1, \
                                             shuffle=True, \
                                             pin_memory=True,
                                             num_workers=opt.test_num_workers)

    print('{:d} roidb large entries'.format(len(dataloader_large)))
    print('{:d} roidb small entries'.format(len(dataloader_small)))
    print('{:d} roidb small test entries'.format(len(dataloader_small_test)))

    faster_rcnn = FasterRCNNVGG16_GAN()
    faster_rcnn_ = FasterRCNNVGG16()

    print('model construct completed')
    trainer_ = FasterRCNNTrainer(faster_rcnn_).cuda()

    netD = Discriminator()
    netD.apply(weights_init)

    faster_rcnn_.cuda()
    netD.cuda()

    lr = opt.LEARNING_RATE
    params_D = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params_D += [{'params': [value], 'lr': lr * 2, \
                              'weight_decay': 0}]
            else:
                params_D += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]

    optimizerD = optim.SGD(params_D, momentum=0.9)
    # optimizerG = optim.Adam(faster_rcnn.parameters(), lr=lr, betas=(0.5, 0.999))

    if not opt.gan_load_path:
        trainer_.load(opt.load_path)
        print('load pretrained faster rcnn model from %s' % opt.load_path)

        # optimizer_ = trainer_.optimizer
        state_dict_ = faster_rcnn_.state_dict()
        state_dict = faster_rcnn.state_dict()

        # for k, i in state_dict_.items():
        #     icpu = i.cpu()
        #     b = icpu.data.numpy()
        #     sz = icpu.data.numpy().shape
        #     state_dict[k] = state_dict_[k]
        state_dict.update(state_dict_)
        faster_rcnn.load_state_dict(state_dict)
        faster_rcnn.cuda()

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    if opt.gan_load_path:
        trainer.load(opt.gan_load_path, load_optimizer=True)
        print('load pretrained generator model from %s' % opt.gan_load_path)

    if opt.disc_load_path:
        state_dict_d = torch.load(opt.disc_load_path)
        netD.load_state_dict(state_dict_d['model'])
        optimizerD.load_state_dict(state_dict_d['optimizer'])
        print('load pretrained discriminator model from %s' % opt.disc_load_path)

    real_label = 1
    fake_label = 0

    # rpn_loc_loss = []
    # rpn_cls_loss = []
    # roi_loc_loss = []
    # roi_cls_loss = []
    # total_loss = []
    test_map_list = []

    criterion = nn.BCELoss()
    iters_per_epoch = min(len(dataloader_large), len(dataloader_small))
    best_map = 0
    device = torch.device("cuda:2" if (torch.cuda.is_available()) else "cpu")

    for epoch in range(1, opt.gan_epoch + 1):
        trainer.reset_meters()

        loss_temp_G = 0
        loss_temp_D = 0
        if epoch % (opt.lr_decay_step + 1) == 0:
            adjust_learning_rate(trainer.optimizer, opt.LEARNING_RATE_DECAY_GAMMA)
            adjust_learning_rate(optimizerD, opt.LEARNING_RATE_DECAY_GAMMA)
            lr *= opt.LEARNING_RATE_DECAY_GAMMA

        data_iter_large = iter(dataloader_large)
        data_iter_small = iter(dataloader_small)
        for step in tqdm(range(iters_per_epoch)):
            #####(1) Update Perceptual branch + generator(zero mapping)
            ####     Discriminator network: maximize log(D(x))+ log(1-D(G(z)))

            ##### Train with all_real batch
            ##### Format batch
            netD.zero_grad()
            data_large = next(data_iter_large)
            img, bbox_, label_, scale_ = data_large
            scale = at.scalar(scale_)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

            ##### Forward pass real batch through D
            # faster_rcnn.zero_grad()
            # trainer.optimizer.zero_grad()
            # trainer.optimizer.zero_grad()

            losses, pooled_feat, rois_label, conv1_feat = trainer.train_step_gan(img, bbox, label, scale)

            # if step < 1:
            #     custom_viz(conv1_feat.cpu().detach(), 'results-gan/features/large_orig_%s' % str(epoch))
            #     custom_viz(pooled_feat.cpu().detach(), 'results-gan/features/large_scaled_%s' % str(epoch))

            keep = rois_label != 0
            pooled_feat = pooled_feat[keep]

            real_b_size = pooled_feat.size(0)
            real_labels = torch.full((real_b_size,), real_label, device=device)

            output = netD(pooled_feat.detach()).view(-1)
            # print(output)

            ##### Calculate loss on all-real batch

            errD_real = criterion(output, real_labels)
            errD_real.backward()
            D_x = output.mean().item()

            ##### Train with all_fake batch
            # Generate batch of fake images with G
            data_small = next(data_iter_small)
            img, bbox_, label_, scale_ = data_small
            scale = at.scalar(scale_)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.optimizer.zero_grad()

            losses, fake_pooled_feat, rois_label, conv1_feat = trainer.train_step_gan_second(img, bbox, label, scale)

            # if step < 1:
            #     custom_viz(conv1_feat.cpu().detach(), 'results-gan/features/small_orig_%s' % str(epoch))
            #     custom_viz(fake_pooled_feat.cpu().detach(), 'results-gan/features/small_scaled_%s' % str(epoch))

            # select fg rois
            keep = rois_label != 0
            fake_pooled_feat = fake_pooled_feat[keep]
            # print(fake_pooled_feat)
            # print(torch.nonzero(torch.isnan(fake_pooled_feat.view(-1))))

            fake_b_size = fake_pooled_feat.size(0)
            fake_labels = torch.full((fake_b_size,), fake_label, device=device)

            # optimizerD.zero_grad()
            output = netD(fake_pooled_feat.detach()).view(-1)

            # calculate D's loss on the all_fake batch
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward(retain_graph=True)
            D_G_Z1 = output.mean().item()
            # add the gradients from the all-real and all-fake batches
            errD = errD_fake + errD_real
            # Update D
            optimizerD.step()

            ################################################
            #####(2) Update G network: maximize log(D(G(z)))
            ################################################
            faster_rcnn.zero_grad()

            fake_labels.fill_(real_label)

            output = netD(fake_pooled_feat).view(-1)

            # calculate gradients for G
            errG = criterion(output, fake_labels)
            errG += losses.total_loss
            errG.backward()
            D_G_Z2 = output.mean().item()

            clip_gradient(faster_rcnn, 10.)

            trainer.optimizer.step()

            loss_temp_G += errG.item()
            loss_temp_D += errD.item()

            if step % opt.plot_every == 0:
                if step > 0:
                    loss_temp_G /= (opt.plot_every + 1)
                    loss_temp_D /= (opt.plot_every + 1)

                # losses_dict = trainer.get_meter_data()
                #
                # rpn_loc_loss.append(losses_dict['rpn_loc_loss'])
                # roi_loc_loss.append(losses_dict['roi_loc_loss'])
                # rpn_cls_loss.append(losses_dict['rpn_cls_loss'])
                # roi_cls_loss.append(losses_dict['roi_cls_loss'])
                # total_loss.append(losses_dict['total_loss'])
                #
                # save_losses('rpn_loc_loss', rpn_loc_loss, epoch)
                # save_losses('roi_loc_loss', roi_loc_loss, epoch)
                # save_losses('rpn_cls_loss', rpn_cls_loss, epoch)
                # save_losses('total_loss', total_loss, epoch)
                # save_losses('roi_cls_loss', roi_cls_loss, epoch)

                print("[epoch %2d] lossG: %.4f lossD: %.4f, lr: %.2e"
                      % (epoch, loss_temp_G, loss_temp_D, lr))
                print("\t\t\trcnn_cls: %.4f, rcnn_box %.4f"
                      % (losses.roi_cls_loss, losses.roi_loc_loss))

                print("\t\t\trpn_cls: %.4f, rpn_box %.4f"
                      % (losses.rpn_cls_loss, losses.rpn_loc_loss))

                print('\t\t\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (D_x, D_G_Z1, D_G_Z2))
                loss_temp_D = 0
                loss_temp_G = 0

        eval_result = eval(dataloader_small_test, faster_rcnn, test_num=opt.test_num)
        test_map_list.append(eval_result['map'])
        save_map(test_map_list, epoch)

        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{}'.format(str(lr_),
                                                  str(eval_result['map']))
        print(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            timestr = time.strftime('%m%d%H%M')
            trainer.save(best_map=best_map, save_path='checkpoints-pcgan-generated/gan_fasterrcnn_%s' % timestr)

            save_dict = dict()

            save_dict['model'] = netD.state_dict()

            save_dict['optimizer'] = optimizerD.state_dict()
            save_path = 'checkpoints-pcgan-generated/discriminator_%s' % timestr
            torch.save(save_dict, save_path)


if __name__ == '__main__':
    import fire

    # train()
    fire.Fire()
