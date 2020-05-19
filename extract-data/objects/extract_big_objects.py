import os
import xml.etree.ElementTree as ET
from shutil import copyfile
import numpy as np
import shutil
import cv2
import numpy as np

class VOCExtractBigObjects:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def extract_big_obj(self):
        print('Extracting big objects from VOC pascal dataset')

        big_obj_annotation_dir = os.path.join(self.data_dir, 'AnnotationsBig/')

        big_obj_images_dir = self.data_dir + 'JPEGImagesBigObjects/'
        if os.path.exists(big_obj_images_dir):
            print('Removing big annotation jpgs.')
            shutil.rmtree(big_obj_images_dir)
        os.makedirs(os.path.dirname(big_obj_images_dir), exist_ok=True)

        count = 0
        sum_x = 0
        sum_y = 0

        for anno_file in os.listdir(big_obj_annotation_dir):

            image_path = os.path.join(self.data_dir, 'JPEGImages/', anno_file.split('.')[0] + '.jpg')
            if not os.path.exists(image_path):
                print('Image ' + image_path + ' does not exists.')
                break

            image = cv2.imread(image_path)

            anno = ET.parse(os.path.join(self.data_dir, 'AnnotationsBig', anno_file))
            objects_dict = {}
            for obj in anno.findall('object'):
                bndbox_anno = obj.find('bndbox')
                name_anno = obj.find('name').text

                xmin = int(bndbox_anno.find('xmin').text) - 1
                xmax = int(bndbox_anno.find('xmax').text) - 1
                ymin = int(bndbox_anno.find('ymin').text) - 1
                ymax = int(bndbox_anno.find('ymax').text) - 1

                count += 1
                sum_x += (xmax - xmin)
                sum_y += (ymax - ymin)

                img_obj = image[ymin:ymax, xmin:xmax]
                name_anno_file = ""
                if name_anno in objects_dict:
                    objects_dict[name_anno] += 1
                    name_anno_file = name_anno + "_" + str(objects_dict[name_anno])
                else:
                    objects_dict[name_anno] = 0
                    name_anno_file = name_anno
                obj_img_id = anno_file.split('.')[0]+"_"+name_anno_file
                cv2.imwrite(big_obj_images_dir + obj_img_id + '.jpg', img_obj)
        print('Avg x: ' + str(sum_x/count))
        print('Avg y: ' + str(sum_y/count))
        print('count: ' + str(count))

# Avg x: 209.61747086374032
# Avg y: 207.86079947718113
# count: 9181


data_dir_voc = '/home/magda/Documents/simple-faster-rcnn-pytorch/dataset/VOCdevkit/VOC2007/'

big_objects = VOCExtractBigObjects(data_dir_voc)
big_objects.extract_big_obj()
