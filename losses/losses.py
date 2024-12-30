import torch
import torch.nn as nn
import torch.nn.functional as F


def CosLoss(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].reshape(a[item].shape[0],-1),
                                      b[item].reshape(b[item].shape[0],-1)))
    return loss

def match_sparsify(V, A_, F_):
    
    batch_size = V.shape[0]
    time = V.shape[1]
    max_features = V.shape[-1]
    
    def pad_tensor(V, O, b, t, m):
        O_m = O.shape[-1]
        O_flat = O.view(b * t, O_m)  
        V_flat = V.view(b * t, m)  

        O_flat_norm = F.normalize(O_flat, p=2, dim=0)
        V_flat_norm = F.normalize(V_flat, p=2, dim=0)      

        similarity = torch.mm(O_flat_norm.t(), V_flat_norm)

        _, S = torch.topk(similarity, m, dim=1)

        I = torch.empty(O_m, dtype=torch.long)  
        used_values = set()  

        for i in range(O_m):
            value_found = False
            for j in range(m):  
                candidate = S[i, j].item() 
                if candidate not in used_values:  
                    I[i] = candidate  
                    used_values.add(candidate)  
                    value_found = True
                    break  
            if not value_found:
                print(f"Warning: No unique value found for index {i}.")

        extended_i = torch.empty(m, dtype=torch.long)
        extended_i[:O_m] = I 

        remaining_values = [x for x in range(m) if x not in used_values]
        extended_i[O_m:] = torch.tensor(remaining_values[:m-O_m])  

        padded_O = F.pad(O, (0, m-O_m))
        padded_O = padded_O[:, :, extended_i]

        return padded_O
    
    padded_A = pad_tensor(V, A_, batch_size, time ,max_features)
    padded_F = pad_tensor(V, F_, batch_size, time ,max_features)
    
    return V, padded_A, padded_F

class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def get_loss(self, result, label):

        output = result['output']
        output_loss = self.bce(output, label)

        return output_loss
        
    def forward(self, v_result, a_result, f_result, label):

        label = label.float()

        v_loss = self.get_loss(v_result, label)
        a_loss = self.get_loss(a_result, label)
        f_loss = self.get_loss(f_result, label)

        U_MIL_loss = v_loss + a_loss + f_loss

        loss_dict = {}
        loss_dict['U_MIL_loss'] = U_MIL_loss

        return U_MIL_loss, loss_dict
    

class DISL_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.triplet = nn.TripletMarginLoss(margin=5)

    def norm(self, data):
        l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
        return torch.div(data, l2)
    
    def get_seq_matrix(self, seq_len):
        N = seq_len.size(0)
        M = seq_len.max().item()
        seq_matrix = torch.zeros((N, M))
        for j, val in enumerate(seq_len):
            seq_matrix[j, :val] = 1
        seq_matrix = seq_matrix.cuda()
        return seq_matrix

    def get_mil_loss(self, result, label):

        output = result['output']
        output_loss = self.bce(output, label)

        return output_loss
    
    def cross_entropy_loss(self, q, p):
        epsilon = 1e-6
        p = torch.clamp(p, epsilon, 1 - epsilon)
        q = torch.clamp(q, epsilon, 1 - epsilon)
        cross_entropy_loss = -torch.mean(p * torch.log(q) + (1 - p) * torch.log(1 - q))
        return cross_entropy_loss

    def get_alignment_loss(self, v_result, va_result, vf_result, seq_len):
        def distance(x, y):
            return CosLoss(x, y)
        
        V = v_result["satt_f"].detach().clone()
        A = va_result["satt_f"]
        F = vf_result["satt_f"]

        batch_size = V.shape[0]

        V, A, F = match_sparsify(V, A, F)

        d_VA = distance(V, A)/batch_size
        d_VF = distance(V, F)/batch_size
        d_AF = distance(A, F)/batch_size

        seq_matrix = self.get_seq_matrix(seq_len)
        V = v_result["avf_out"].detach().clone() * seq_matrix
        A = va_result["avf_out"] * seq_matrix
        F = vf_result["avf_out"] * seq_matrix

        ce_VA = self.cross_entropy_loss(V, A)
        ce_VF = self.cross_entropy_loss(V, F)
        ce_AF = self.cross_entropy_loss(A, F)

        return d_VA + d_VF + d_AF + ce_VA + ce_VF + ce_AF

    def get_triplet_loss(self, vaf_result, label, seq_len):

        if torch.sum(label) == label.shape[0] or torch.sum(label) == 0:
            return 0.0

        N_label = (label==0)
        A_label = (label==1)

        sigout = vaf_result["avf_out"]
        feature = vaf_result["satt_f"]

        N_feature = feature[N_label]
        A_feature = feature[A_label]

        N_sigout = sigout[N_label]
        A_sigout = sigout[A_label]

        N_seq_len = seq_len[N_label]
        A_seq_len = seq_len[A_label]

        anchor = torch.zeros(N_feature.shape[0], N_feature.shape[-1]).cuda()
        for i in range(N_sigout.shape[0]):
            _, index = torch.topk(N_sigout[i][:N_seq_len[i]], k=int(N_seq_len[i]), largest=True)
            tmp = N_feature[i, index, :]
            anchor[i] = tmp.mean(dim=0)
        anchor = anchor.mean(dim=0)

        positivte = torch.zeros(A_feature.shape[0], A_feature.shape[-1]).cuda()
        negative = torch.zeros(A_feature.shape[0], A_feature.shape[-1]).cuda()

        for i in range(A_sigout.shape[0]):
            _, index = torch.topk(A_sigout[i][:A_seq_len[i]], k=int(A_seq_len[i] // 16 + 1), largest=False)
            tmp = A_feature[i, index, :]
            positivte[i] = tmp.mean(dim=0)

            _, index = torch.topk(A_sigout[i][:A_seq_len[i]], k=int(A_seq_len[i] // 16 + 1), largest=True)
            tmp = A_feature[i, index, :]
            negative[i] = tmp.mean(dim=0)

        positivte = positivte.mean(dim=0)
        negative = negative.mean(dim=0)
            
        triplet_margin_loss = self.triplet(self.norm(anchor), self.norm(positivte), self.norm(negative))

        return triplet_margin_loss
     
    def forward(self, v_result, va_result, vf_result, vaf_result, label, seq_len, lamda1, lamda2, lamda3, lamda4):

        label = label.float()
        a_loss = self.get_mil_loss(va_result, label)
        f_loss = self.get_mil_loss(vf_result, label)
        raf_loss = self.get_mil_loss(vaf_result, label)

        ma_loss = self.get_alignment_loss(v_result, va_result, vf_result, seq_len)
        triplet_loss = self.get_triplet_loss(vaf_result, label, seq_len)

        total_loss = lamda1*ma_loss + lamda2*(a_loss + f_loss) + lamda3*raf_loss + lamda4*triplet_loss

        loss_dict = {}
        loss_dict['ma_loss'] = ma_loss
        loss_dict['af_loss'] = a_loss + f_loss
        loss_dict['raf_loss'] = raf_loss
        loss_dict['triplet_loss'] = triplet_loss

        return total_loss, loss_dict