import torch
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def test(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, test_info, epoch):
    
    with torch.no_grad():

        v_net.eval()
        a_net.eval()
        f_net.eval()
        va_net.eval()
        vf_net.eval()
        vaf_net.eval()
        
        m_pred = torch.zeros(0).cuda()

        for i, (f_v, f_a, f_f) in tqdm(enumerate(test_loader)):
            
            v_data = f_v.cuda()
            a_data = f_a.cuda()
            f_data = f_f.cuda()

            v_res = v_net(v_data)
            a_res = a_net(a_data)
            f_res = f_net(f_data)

            mix_f = torch.cat([v_res["satt_f"], va_net(a_res["satt_f"]), vf_net(f_res["satt_f"])], dim=-1)
            m_out = vaf_net(mix_f)
            
            m_out = torch.mean(m_out["output"], 0)
            m_pred = torch.cat((m_pred, m_out))

        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["epoch"].append(epoch)
        test_info["m_ap"].append(m_ap)

        