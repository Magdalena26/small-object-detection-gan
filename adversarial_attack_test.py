from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import FasterRCNNVGG16
import resource
from utils.config import opt
from data.dataset import TestFGSMDataset
from torch.utils import data as data_
from trainer2 import FasterRCNNTrainer
from tqdm import tqdm
from utils.eval_tool import eval_detection_voc
from utils import array_tool as at
from torchvision.utils import save_image
import torchvision.utils as vutils
import os
import csv

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

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


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    # print(sign_data_grad)
    perturbed_image = image + epsilon * sign_data_grad * 128
    # Adding clipping to maintain range
    perturbed_image = torch.clamp(perturbed_image, -128, 128)
    # Return the perturbed image
    return perturbed_image


def test(model, test_loader, epsilon, trainer, file):
    adv_examples = []
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    # Loop over all examples in test set
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, scales_) in tqdm(enumerate(test_loader)):
        scale = at.scalar(scales_)
        imgs = imgs.cuda().float()
        bbox = gt_bboxes_.cuda()
        label = gt_labels_.cuda()
        sizes = [sizes[0][0].item(), sizes[1][0].item()]

        imgs.requires_grad_(True)
        model.get_optimizer().zero_grad()
        losses = trainer.forward(imgs, bbox, label, scale)
        losses.total_loss.backward()

        # Collect datagrad
        data_grad = imgs.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(imgs, epsilon, data_grad)

        if ii < 10:
            stacked = torch.cat((imgs, perturbed_data), 0)
            save_image(vutils.make_grid(stacked, padding=2), "fgsm/images/%s_fgsm.jpg" % str(epsilon))
        perturbed_data = perturbed_data.cuda().float()

        pred_bboxes_, pred_labels_, pred_scores_ = model.predict(perturbed_data, [sizes])

        # Check for success
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

        if len(adv_examples) < 10:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((epsilon, adv_ex))

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    i = 0
    file.write(str(epsilon) + ",")
    for ap in result['ap']:
        print(('AP for {} = {:.4f}'.format(VOC_BBOX_LABEL_NAMES[i], ap)))
        file.write(str(ap) + ",")
        i += 1
    file.write(str(result['map']) + "\n")
    print('test_map', result['map'])
    return result['map'], adv_examples


def plot_mAP_vs_Eps(epsilons, accuracies):
    plt.figure(figsize=(10, 12))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig('fgsm/map_eps.jpg')


def init_cols():
    cols = list()
    cols.append('epsilon')
    for i in range(len(VOC_BBOX_LABEL_NAMES)):
        cols.append(VOC_BBOX_LABEL_NAMES[i])
    cols.append('mAP')
    return cols


def generate_adversarial_example(**kwargs):
    opt._parse(kwargs)
    epsilons = [0.0, .05, .1, .15, .2, .25, .3]
    opt.caffe_pretrain = True

    results_file = 'fgsm_faster_rcnn_results.csv'
    if os.path.exists(results_file):
        file = open(results_file, "w+")
    else:
        file = open(results_file, "w")
        columns = init_cols()
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

    testset = TestFGSMDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=True, \
                                       pin_memory=True)

    print(len(test_dataloader))
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    else:
        print("provide path of the checkpoint to be loaded.")
        exit()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        print("Running FGSM attack, epsilon " + str(eps))
        acc, ex = test(faster_rcnn, test_dataloader, eps, trainer, file)
        accuracies.append(acc)
        examples.append(ex)

    plot_mAP_vs_Eps(epsilons, accuracies)
    file.close()


if __name__ == "__main__":
    import fire

    fire.Fire()
