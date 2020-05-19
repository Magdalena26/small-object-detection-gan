from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
from model import FasterRCNNVGG16
from utils.config import opt
from data.dataset import TestTypeDataset
from model.faster_rcnn_vgg16_gan import FasterRCNNVGG16_GAN
import torch.nn as nn
from torch.utils import data as data_
from model.Discriminator import Discriminator
from trainer2 import FasterRCNNTrainer
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
import csv

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


def eval(dataloader, faster_rcnn, file=None, category=None, test_num=10000):
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
    i = 0
    if file is not None:
        file.write(str(category) + "," + str(result['map']) + ',')
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
    cols.append('type')
    cols.append('mAP')
    for i in range(len(VOC_BBOX_LABEL_NAMES)):
        cols.append(VOC_BBOX_LABEL_NAMES[i])
    return cols


def get_types():
    types = ['test_small_32', 'test_medium', 'test_big_64']

    return types


def get_id_list_files():
    id_list = ['ImageSets/Main/test_small_32.txt', 'ImageSets/Main/test_medium.txt', 'ImageSets/Main/test_big_64.txt']

    return id_list


def get_img_dirs():
    img_dirs = ['JPEGImages', 'JPEGImages', 'JPEGImages']

    return img_dirs


def get_anno_dirs():
    anno_dirs = ['AnnotationsSmall', 'AnnotationsMedium', 'AnnotationsBig']

    return anno_dirs




def eval_main(**kwargs):
    opt._parse(kwargs)

    opt.test_num = 10000
    opt.caffe_pretrain = True

    types = get_types()
    id_files = get_id_list_files()
    img_dirs = get_img_dirs()
    anno_dirs = get_anno_dirs()

    results_file = 'oversampled-pc-class.csv'

    if os.path.exists(results_file):
        file = open(results_file, "w+")
    else:
        file = open(results_file, "w")
        columns = init_cols()
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

    faster_rcnn = FasterRCNNVGG16_GAN()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    if opt.gan_load_path:
        trainer.load('checkpoints-pcgan/gan_fasterrcnn_05090509', load_optimizer=False)
        print('load pretrained generator model from %s' % opt.gan_load_path)
    else:
        print("provide path of the checkpoint to be loaded.")
        exit()

    for category, id_file, img_dir, anno_dir in zip(types, id_files, img_dirs, anno_dirs):
        testset = TestTypeDataset(opt, use_difficult=True, id_file=id_file, img_dir=img_dir, anno_dir=anno_dir)
        test_dataloader = data_.DataLoader(testset,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False, \
                                           pin_memory=True)

        print(category)
        eval_result = eval(test_dataloader, faster_rcnn, file=file, category=category, test_num=opt.test_num)
        print('test_map', eval_result['map'])

    file.close()





if __name__ == '__main__':
    import fire

    # train()
    fire.Fire()


# gan: checkpoints-pcgan/gan_fasterrcnn_05090509