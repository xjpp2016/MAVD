import argparse
from random import seed
import os

def init_args(args):
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args

descript = 'Pytorch Implementation of VAF'
parser = argparse.ArgumentParser(description = descript)
parser.add_argument('--output_path', type = str, default = 'outputs/')
parser.add_argument('--root_dir', type = str, default = 'outputs/')
parser.add_argument('--log_path', type = str, default = 'logs/')
parser.add_argument('--model_path', type = str, default = 'saved_models/')
parser.add_argument('--init_path', type = str, default = 'saved_models/init_models/')
parser.add_argument('--model_file', type = str, default = "model_{}.pkl".format(seed), help = 'the path of pre-trained model file')

parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rates')
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--seed', type = int, default = 500, help = 'random seed (-1 for no manual seed)')
parser.add_argument('--workers', default=8, help='number of workers in dataloader')
parser.add_argument('--num_steps', default=1000, help='number of epochs to train for')

parser.add_argument('--rgb-list', default='./list/video_train.list', help='list of rgb features')
parser.add_argument('--flow-list', default='./list/flow_train.list', help='list of flow features')
parser.add_argument('--audio-list', default='./list/audio_train.list', help='list of audio features')
parser.add_argument('--test-rgb-list', default='./list/video_test.list', help='list of test rgb features ')
parser.add_argument('--test-flow-list', default='./list/flow_test.list', help='list of test flow features')
parser.add_argument('--test-audio-list', default='./list/audio_test.list', help='list of test audio features')

parser.add_argument('--max_seqlen', type=int, default=200, help='maximum sequence length during training')
parser.add_argument('--gt', default='./list/gt.npy', help='file of ground truth ')

parser.add_argument('--lamda1', type = float, default = 10.0)
parser.add_argument('--lamda2', type = float, default = 10.0)
parser.add_argument('--lamda3', type = float, default = 0.001)



