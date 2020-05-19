import os
import xml.etree.ElementTree as ET
from shutil import copyfile
import numpy as np
import shutil


class TestObjectsClasses:
    def __init__(self, small_obj_size, big_obj_size, data_dir, split_small, split_big, split_medium):
        self.small_obj_size = small_obj_size
        self.big_obj_size = big_obj_size
        self.data_dir = data_dir
        self.split_small = split_small
        self.split_medium = split_medium
        self.split_big = split_big

    def check_small_objects(self):
        print('Testing small objects from VOC pascal dataset')

        small_id_list_file = os.path.join(self.data_dir, 'ImageSets/Main/{0}.txt'.format(self.split_small))

        if not os.path.exists(small_id_list_file):
            print('File does not exists:' + small_id_list_file)
            return

        id_file = open(small_id_list_file, "r")

        annotations_small_dir = self.data_dir + 'AnnotationsSmall/'
        if not os.path.exists(annotations_small_dir):
            print('Annotation dir does not exists')
            return

        ids = [id_.strip() for id_ in open(small_id_list_file)]

        for id_ in ids:
            self.check_example(id_)

    def check_size(self, x, y):
        return abs(y - x) >= self.small_obj_size

    def check_example(self, id_):
        dest_file = self.data_dir + 'AnnotationsSmall/' + id_ + '.xml'

        if not os.path.exists(dest_file):
            print('Annotation file does not exists: ', dest_file)

        anno = ET.parse(dest_file)

        ann_len = len(anno.findall('object'))
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')

            xmin = int(bndbox_anno.find('xmin').text) - 1
            xmax = int(bndbox_anno.find('xmax').text) - 1
            ymin = int(bndbox_anno.find('ymin').text) - 1
            ymax = int(bndbox_anno.find('ymax').text) - 1

            if self.check_size(xmin, xmax) or self.check_size(ymin, ymax):
                print('Large object found!')

    def check_big_objects(self):
        print('Testing big objects from VOC pascal dataset')

        big_id_list_file = os.path.join(self.data_dir, 'ImageSets/Main/{0}.txt'.format(self.split_big))

        if not os.path.exists(big_id_list_file):
            print('File does not exists:' + big_id_list_file)
            return

        id_file = open(big_id_list_file, "r")

        annotations_big_dir = self.data_dir + 'AnnotationsBig/'
        if not os.path.exists(annotations_big_dir):
            print('Annotation dir does not exists')
            return

        ids = [id_.strip() for id_ in open(big_id_list_file)]

        for id_ in ids:
            self.check_example_big(id_)

    def check_size_big(self, x, y):
        return abs(y - x) < self.big_obj_size

    def check_example_big(self, id_):
        dest_file = self.data_dir + 'AnnotationsBig/' + id_ + '.xml'

        if not os.path.exists(dest_file):
            print('Annotation file does not exists: ', dest_file)

        anno = ET.parse(dest_file)

        ann_len = len(anno.findall('object'))
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')

            xmin = int(bndbox_anno.find('xmin').text) - 1
            xmax = int(bndbox_anno.find('xmax').text) - 1
            ymin = int(bndbox_anno.find('ymin').text) - 1
            ymax = int(bndbox_anno.find('ymax').text) - 1

            if self.check_size_big(xmin, xmax) or self.check_size_big(ymin, ymax):
                print('Small object found!')
                ann_len -= 1

    def check_medium_objects(self):
        print('Testing medium objects from VOC pascal dataset')

        medium_id_list_file = os.path.join(self.data_dir, 'ImageSets/Main/{0}.txt'.format(self.split_medium))

        if not os.path.exists(medium_id_list_file):
            print('File does not exists:' + medium_id_list_file)
            return

        id_file = open(medium_id_list_file, "r")

        annotations_medium_dir = self.data_dir + 'AnnotationsMedium/'
        if not os.path.exists(annotations_medium_dir):
            print('Annotation dir does not exists')
            return

        ids = [id_.strip() for id_ in open(medium_id_list_file)]

        for id_ in ids:
            self.check_example_medium(id_)

    def check_size_medium(self, x, y):
        return abs(y - x) < self.small_obj_size or abs(y - x) >= self.big_obj_size

    def check_example_medium(self, id_):
        dest_file = self.data_dir + 'AnnotationsMedium/' + id_ + '.xml'

        if not os.path.exists(dest_file):
            print('Annotation file does not exists: ', dest_file)

        anno = ET.parse(dest_file)

        ann_len = len(anno.findall('object'))
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')

            xmin = int(bndbox_anno.find('xmin').text) - 1
            xmax = int(bndbox_anno.find('xmax').text) - 1
            ymin = int(bndbox_anno.find('ymin').text) - 1
            ymax = int(bndbox_anno.find('ymax').text) - 1

            if self.check_size_medium(xmin, xmax) or self.check_size_medium(ymin, ymax):
                print('Small/Big object found!')
                ann_len -= 1



data_dir_voc = '/home/magda/Documents/orig/simple-faster-rcnn-pytorch/dataset/VOCdevkit/VOC2007/'

small_objects = TestObjectsClasses(32, 64, data_dir_voc, 'test_small_32', 'test_big_64', 'test_medium')
small_objects.check_small_objects()
small_objects.check_big_objects()
small_objects.check_medium_objects()
