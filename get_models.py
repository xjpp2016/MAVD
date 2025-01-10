import os
import torch
import numpy as np
import torch.utils.data as data
import config.options as options

from train_test.train import *
from train_test.test import *
from utils.utils import *
from losses.losses import * 
from model.unimodal import *
from model.multimodal import *
from model.projection import *
from dataset.dataset_loader import *


if __name__ == "__main__":

    args = options.parser.parse_args()
    args.save_model_path = f'saved_models/seed_{args.seed}/'
    args = options.init_args(args)
    set_seed(args.seed)
    
    lamda1, lamda2, lamda3= args.lamda1, args.lamda2, args.lamda3

    train_loader = data.DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = data.DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    v_net = Unimodal(input_size=1024, h_dim=128, feature_dim=128)
    a_net = Unimodal(input_size=128, h_dim=64, feature_dim=32)
    f_net = Unimodal(input_size=1024, h_dim=128, feature_dim=64)
    v_net = v_net.cuda()
    a_net = a_net.cuda()
    f_net = f_net.cuda()

    va_net = Projection(32, 32, 32)
    vf_net = Projection(64, 64, 64)
    va_net = va_net.cuda()
    vf_net = vf_net.cuda()

    vaf_net = Multimodal(input_size=128+32+64, h_dim=128, feature_dim=64)
    vaf_net = vaf_net.cuda()
    
    optimizer = torch.optim.Adam(list(v_net.parameters())+list(a_net.parameters())+list(f_net.parameters())+list(va_net.parameters())+list(vf_net.parameters())+list(vaf_net.parameters()), 
                                 lr = args.lr, betas = (0.9, 0.999), weight_decay = 0.0005)

    criterion = AD_Loss()
    criterion_disl = DISL_Loss()
    
    best_ap = 0.0
    test_info = {"iteration": [], "m_ap":[]}

    gt = np.load(args.gt)

    for step in range(1, args.num_steps + 1):

        if (step-1) % len(train_loader) == 0:
            train_loader_iter = iter(train_loader)

        loss_dict_list, loss_dict_list_disl = train(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                    train_loader_iter, 
                    optimizer,
                    criterion, criterion_disl, step,
                    lamda1, lamda2, lamda3)
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]

        print(f'Step: {step}, '
            f'U_MIL_loss: {loss_dict_list["U_MIL_loss"]:.6f}, '
            f'MA_loss: {loss_dict_list_disl["MA_loss"]:.6f}, '
            f'M_MIL_loss: {loss_dict_list_disl["M_MIL_loss"]:.6f}, '
            f'Triplet_loss: {loss_dict_list_disl["Triplet_loss"]:.6f}, '
            f'LR: {current_lr:.6f} '
            )

        if step % 10 == 0: 
            test(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                            test_loader, gt, 
                            test_info, step)

            if test_info["m_ap"][-1] > best_ap:
                best_ap = test_info["m_ap"][-1]
                utils.save_best_record(test_info, 
                    os.path.join(args.output_path, "best_record_{}.txt".format(args.seed)))
                torch.save(v_net.state_dict(), os.path.join(args.save_model_path, "v_model.pth"))
                torch.save(a_net.state_dict(), os.path.join(args.save_model_path, "a_model.pth"))
                torch.save(f_net.state_dict(), os.path.join(args.save_model_path, "f_model.pth"))
                torch.save(va_net.state_dict(), os.path.join(args.save_model_path, "va_model.pth"))
                torch.save(vf_net.state_dict(), os.path.join(args.save_model_path, "vf_model.pth"))
                torch.save(vaf_net.state_dict(), os.path.join(args.save_model_path, "vaf_model.pth"))
            
