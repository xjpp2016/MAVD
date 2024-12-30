import torch.utils.data as data
import numpy as np
import utils.utils as utils
 
class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False, return_name=False):
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
            self.flow_list_file = args.test_flow_list
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list
            self.flow_list_file = args.flow_list
        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = test_mode
        self.return_name = return_name
        self.normal_flag = '_label_A'
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file, encoding='utf-8'))
        self.audio_list = list(open(self.audio_list_file, encoding='utf-8'))
        self.flow_list = list(open(self.flow_list_file, encoding='utf-8'))

    def __getitem__(self, index):
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0
        f_v = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        f_f = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
        f_a = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
        if self.transform is not None:
            f_v = self.transform(f_v)
            f_f = self.transform(f_f)
            f_a = self.transform(f_a)
        if self.test_mode:
            if self.return_name == True:
                file_name = self.list[index].strip('\n').split('/')[-1][:-7]
                return f_v, f_a, f_f, file_name
            return f_v, f_a, f_f
        else:
            f_v = utils.process_feat(f_v, self.max_seqlen, is_random=False)
            f_a = utils.process_feat(f_a, self.max_seqlen, is_random=False)
            f_f = utils.process_feat(f_f, self.max_seqlen, is_random=False)
            if self.return_name == True:
                file_name = self.list[index].strip('\n').split('/')[-1][:-7]
                return f_v, f_a, f_f, file_name
            return f_v, f_a, f_f, label

    def __len__(self):
        return len(self.list)

    

