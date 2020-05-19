from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import TestTypeDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
import torch
from utils.eval_tool import eval_detection_voc
import csv
from utils import array_tool as at

import resource

torch.cuda.set_device('cuda:0')

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

AUGUMENTED_CLASSES = (
    'aeroplane',
    'bird',
    'boat',
    'car',
    'chair',
    'diningtable',
    'horse',
    'person',
    'pottedplant',
    'sofa',
    'tvmonitor',
)

def eval(dataloader, faster_rcnn, category=None, file=None, test_num=10000):
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

    results_file = 'oversampled-orig-classm.csv'
    if os.path.exists(results_file):
        file = open(results_file, "w+")
    else:
        file = open(results_file, "w")
        columns = init_cols()
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

    for category, id_file, img_dir, anno_dir in zip(types, id_files, img_dirs, anno_dirs):
        testset = TestTypeDataset(opt, use_difficult=True, id_file=id_file, img_dir=img_dir, anno_dir=anno_dir)
        test_dataloader = data_.DataLoader(testset,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False, \
                                           pin_memory=True)

        faster_rcnn = FasterRCNNVGG16()
        trainer = FasterRCNNTrainer(faster_rcnn).cuda()
        if opt.load_path:
            trainer.load(opt.load_path)
            print('load pretrained model from %s' % opt.load_path)
        else:
            print("provide path of the checkpoint to be loaded.")
            exit()
        print(category)
        eval_result = eval(test_dataloader, faster_rcnn, category, file, test_num=opt.test_num)
        print('test_map', eval_result['map'])

    file.close()


if __name__ == '__main__':
    import fire

    fire.Fire()

    #eval_main()

# orig checkpoints/fasterrcnn_04251656_0.627993265854161
# orig class: checkpoints-class/fasterrcnn_04270953
# orig all: checkpoints2/fasterrcnn_04251557
# gan: checkpoints-gan/fasterrcnn_04271323
# gan class: checkpoints-gan-class/fasterrcnn_04281318
# gan class v2: checkpoints-gan-class-2/fasterrcnn_04291350