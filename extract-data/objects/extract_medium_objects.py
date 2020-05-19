import os
import xml.etree.ElementTree as ET
from shutil import copyfile
import numpy as np
import shutil
import cv2
import numpy as np

class VOCExtractMediumObjects:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def extract_medium_obj(self):
        print('Extracting medium objects from VOC pascal dataset')

        medium_obj_annotation_dir = os.path.join(self.data_dir, 'AnnotationsMedium/')

        medium_obj_images_dir = self.data_dir + 'JPEGImagesMediumObjects/'
        if os.path.exists(medium_obj_images_dir):
            print('Removing medium annotation jpgs.')
            shutil.rmtree(medium_obj_images_dir)
        os.makedirs(os.path.dirname(medium_obj_images_dir), exist_ok=True)

        count = 0
        sum_x = 0
        sum_y = 0

        for anno_file in os.listdir(medium_obj_annotation_dir):

            image_path = os.path.join(self.data_dir, 'JPEGImages/', anno_file.split('.')[0] + '.jpg')
            if not os.path.exists(image_path):
                print('Image ' + image_path + ' does not exists.')
                break

            image = cv2.imread(image_path)

            anno = ET.parse(os.path.join(self.data_dir, 'AnnotationsMedium', anno_file))

            for obj in anno.findall('object'):
                bndbox_anno = obj.find('bndbox')
                name_anno = obj.find('name')

                xmin = int(bndbox_anno.find('xmin').text) - 1
                xmax = int(bndbox_anno.find('xmax').text) - 1
                ymin = int(bndbox_anno.find('ymin').text) - 1
                ymax = int(bndbox_anno.find('ymax').text) - 1

                count += 1
                sum_x += (xmax - xmin)
                sum_y += (ymax - ymin)

                img_obj = image[ymin:ymax, xmin:xmax]
                cv2.imwrite(medium_obj_images_dir + anno_file.split('.')[0]+'_'+name_anno.text + '.jpg', img_obj)
        print('Avg x: ' + str(sum_x/count))
        print('Avg y: ' + str(sum_y/count))
        print('count: ' + str(count))

# Avg x: 45.55005055611729
# Avg y: 47.08493427704752
# count: 989

data_dir_voc = '/home/magda/Documents/GenerativeAdversarialNetwork/dataset/VOCdevkit/VOC2007/'

medium_objects = VOCExtractMediumObjects(data_dir_voc)
medium_objects.extract_medium_obj()
