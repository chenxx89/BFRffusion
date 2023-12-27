import json
import cv2
import numpy as np
from torch.utils.data import Dataset

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class TrainDataset(data.Dataset):
    def __init__(self, opt):
        super(TrainDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        gt_bytes = cv2.imread(gt_path)
        lq_path = self.paths[index]['lq_path']
        lq_bytes = cv2.imread(lq_path)

        # Do not forget that OpenCV read images in BGR order.
        gt = cv2.cvtColor(gt_bytes, cv2.COLOR_BGR2RGB)
        lq = cv2.cvtColor(lq_bytes, cv2.COLOR_BGR2RGB)

        
        # Normalize source images to [0, 1].
        lq = lq.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        gt = (gt.astype(np.float32) / 127.5) - 1.0

        return dict(gt=gt, txt='', lq=lq)


    def __len__(self):
        return len(self.paths)


