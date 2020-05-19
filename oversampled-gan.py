from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import DatasetAugmented, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import time
import torch

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

torch.cuda.set_device('cuda:2')

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[0], rlimit[1]))

matplotlib.use('agg')

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


VOC_BBOX_LABEL_NAMES_BG = (
    'background',
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


def eval(dataloader, faster_rcnn, test_num=10000, epoch=0, file=None):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
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
    i = 0
    if file is not None:
        file.write(str(epoch) + ',')
        file.write(str(result['map']) + ',')
    for ap in result['ap']:
        print(('AP for {} = {:.4f}'.format(VOC_BBOX_LABEL_NAMES[i], ap)))
        if file is not None:
            file.write(str(ap) + ",")
        i += 1
    if file is not None:
        file.write("\n")
    return result

def init_cols():
    cols = list()
    cols.append('epoch')
    cols.append('mAP')
    for i in range(len(VOC_BBOX_LABEL_NAMES)):
        cols.append(VOC_BBOX_LABEL_NAMES[i])
    return cols

def save_losses(loss_type, losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("epoch " + str(epoch) + " " + loss_type)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('losses/losses_gan_class2_%s_%s.jpeg' % (loss_type, str(epoch)))


def save_map(values, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("epoch " + str(epoch) + " test_map")
    plt.plot(values)
    plt.xlabel("epoch")
    plt.ylabel("map")
    plt.savefig('results/map_gan_class2_%s.jpeg' % (str(epoch)))

def save_conf_matrix(matrix, file):

    file = open(file, "w")
    cols = list()
    cols.append('x')


    for i in range(len(VOC_BBOX_LABEL_NAMES_BG)):
        cols.append(VOC_BBOX_LABEL_NAMES_BG[i])

    for i in range(len(cols)):
        file.write(cols[i])
        if i != len(cols) - 1:
            file.write(',')
    file.write('\n')

    for i in range(len(VOC_BBOX_LABEL_NAMES_BG)):

        file.write(VOC_BBOX_LABEL_NAMES_BG[i] + ',')
        for j in range(len(VOC_BBOX_LABEL_NAMES_BG)):
            file.write(str(matrix[i][j]))
            if j != len(VOC_BBOX_LABEL_NAMES_BG) - 1:
                file.write(',')
        file.write('\n')



def train(**kwargs):
    opt._parse(kwargs)

    results_file = 'fasterrcnn-oversampled-gan-class-2.csv'
    if os.path.exists(results_file):
        file = open(results_file, "w+")
    else:
        file = open(results_file, "w")
        columns = init_cols()
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

    id_file_dir = 'ImageSets/Main/train_oversampled_gan_class_v2.txt'
    img_dir = 'JPEGImagesOversampledGANClassv2'
    anno_dir = 'AnnotationsOversampledGANClassv2'


    #
    # id_file_dir = 'ImageSets/Main/train_oversampled_gan.txt'
    # img_dir = 'JPEGImagesOversampledGAN'
    # anno_dir = 'AnnotationsOversampledGAN'

    # id_file_dir = 'ImageSets/Main/train_oversampled_orig_class.txt'
    # img_dir = 'JPEGImagesOversampledClass'
    # anno_dir = 'AnnotationsOversampledClass'

    # id_file_dir = 'ImageSets/Main/train_oversampled_orig_all.txt'
    # img_dir = 'JPEGImagesOrigOversampledAll'
    # anno_dir = 'AnnotationsOrigOversampledAll'


    dataset = DatasetAugmented(opt, id_file=id_file_dir, img_dir=img_dir, anno_dir=anno_dir)
    print(len(dataset))
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=True, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0
    lr_ = opt.lr
    rpn_loc_loss = []
    rpn_cls_loss = []
    roi_loc_loss = []
    roi_cls_loss = []
    total_loss = []
    test_map_list = []

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:

                # plot loss
                losses_dict = trainer.get_meter_data()
                rpn_loc_loss.append(losses_dict['rpn_loc_loss'])
                roi_loc_loss.append(losses_dict['roi_loc_loss'])
                rpn_cls_loss.append(losses_dict['rpn_cls_loss'])
                roi_cls_loss.append(losses_dict['roi_cls_loss'])
                total_loss.append(losses_dict['total_loss'])

                save_losses('rpn_loc_loss', rpn_loc_loss, epoch)
                save_losses('roi_loc_loss', roi_loc_loss, epoch)
                save_losses('rpn_cls_loss', rpn_cls_loss, epoch)
                save_losses('total_loss', total_loss, epoch)
                save_losses('roi_cls_loss', roi_cls_loss, epoch)

        # print('conf matrix final')
        # print(trainer.roi_cm.conf)
        # save_conf_matrix(trainer.roi_cm.conf, 'confusion_matrix/oversampled-orig-class.csv')

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num,epoch=epoch, file=file)
        test_map_list.append(eval_result['map'])
        save_map(test_map_list, epoch)

        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))

        print(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints-gan-class-2/fasterrcnn_%s' % timestr
            # save_path = 'checkpoints2/fasterrcnn_%s' % timestr
            # save_path = 'checkpoints-class/fasterrcnn_%s' % timestr

            best_path = trainer.save(best_map=best_map, save_path=save_path)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13:
            break
    file.close()


if __name__ == '__main__':
    import fire

    fire.Fire()
