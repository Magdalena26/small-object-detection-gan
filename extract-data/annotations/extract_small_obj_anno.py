import os
import xml.etree.ElementTree as ET
from shutil import copyfile
import numpy as np
import shutil


class VOCSmallObjects:
    def __init__(self, small_obj_size, data_dir, split_small, split):
        self.small_obj_size = small_obj_size
        self.data_dir = data_dir
        self.split_small = split_small
        self.split = split

    def extract_small_obj(self):
        print('Extracting small objects from VOC pascal dataset')

        small_id_list_file = os.path.join(self.data_dir, 'ImageSets/Main/{0}.txt'.format(self.split_small))

        if os.path.exists(small_id_list_file):
            print('Removing annotation ids file: ' + small_id_list_file)
            os.remove(small_id_list_file)

        id_file = open(small_id_list_file, "w")

        annotations_small_dir = self.data_dir + 'AnnotationsSmall/'
        # if os.path.exists(annotations_small_dir):
        #     print('Removing small annotation xmls.')
        #     shutil.rmtree(annotations_small_dir)
        os.makedirs(os.path.dirname(annotations_small_dir), exist_ok=True)

        id_list_file = os.path.join(self.data_dir, 'ImageSets/Main/{0}.txt'.format(self.split))
        ids = [id_.strip() for id_ in open(id_list_file)]
        count = 0

        for id_ in ids:
            small_objects_num = self.check_example(id_)
            count += small_objects_num
            if small_objects_num != 0:
                id_file.write(id_ + '\n')
            elif os.path.exists(annotations_small_dir + id_ + '.xml'):
                os.remove(annotations_small_dir + id_ + '.xml')

        print('Count:', count)
        id_file.close()

    def check_size(self, x, y):
        return abs(y - x) >= self.small_obj_size

    def check_example(self, id_):
        src_file = self.data_dir + 'Annotations/' + id_ + '.xml'
        dest_file = self.data_dir + 'AnnotationsSmall/' + id_ + '.xml'

        count = 0

        if os.path.exists(dest_file):
            os.remove(dest_file)

        copyfile(src_file, dest_file)

        anno = ET.parse(dest_file)
        root = anno.getroot()

        ann_len = len(anno.findall('object'))
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')

            xmin = int(bndbox_anno.find('xmin').text) - 1
            xmax = int(bndbox_anno.find('xmax').text) - 1
            ymin = int(bndbox_anno.find('ymin').text) - 1
            ymax = int(bndbox_anno.find('ymax').text) - 1

            if self.check_size(xmin, xmax) or self.check_size(ymin, ymax):
                root.remove(obj)
                ann_len -= 1
            elif int(obj.find('difficult').text) != 1:
                count += 1

        if ann_len == 0:
            os.remove(dest_file)

        ET.dump(root)
        anno.write(dest_file)
        return count


data_dir_voc = '/home/magda/Documents/orig/simple-faster-rcnn-pytorch/dataset/VOCdevkit/VOC2007/'

small_objects = VOCSmallObjects(32, data_dir_voc, 'trainval_small_32', 'trainval')
small_objects.extract_small_obj()
