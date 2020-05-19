from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    # voc_data_dir = '/media/magda/3B88651125331BFA/praca-magisterska/simple-faster-rcnn-pytorch/dataset/VOCdevkit/VOC2007/'
    # voc_data_dir = '/home/plgstachon/simple-faster-rcnn-pytorch/dataset/VOCdevkit/VOC2007/'
    # voc_data_dir = '/home/magda//simple-faster-rcnn-pytorch/dataset/VOCdevkit/VOC2007/'
    voc_data_dir = '/home/plgstachon/VOCdevkit/VOC2007/'

    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 1

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 10  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = 'checkpoints-class/fasterrcnn_04301447'

    caffe_pretrain = True # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'
    imdb_name_large = 'test_big_64'
    imdb_name_small = 'test_small_32'
    max_num_gt_boxes = 20
    gan_load_path = None
    disc_load_path = None
    LEARNING_RATE = 0.0005
    lr_decay_step = 5
    LEARNING_RATE_DECAY_GAMMA = 0.1
    gan_epoch = 20

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
