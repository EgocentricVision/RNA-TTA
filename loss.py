# list all the additional loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

################## IM loss ####################
def IM_loss(pred):
    softmax = torch.nn.Softmax(dim=1)
    softmax_out = softmax(pred)
    entropy_loss = torch.mean(Entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    im_loss = entropy_loss
    return 3 + im_loss

################## MCC loss ####################
def MCC_loss(outputs_target, class_num, bS):
    temperature = 1 #1.0 #3.0
    outputs_target_temp = outputs_target / temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = bS * target_entropy_weight / torch.sum(target_entropy_weight)

    cov_matrix_t_temp = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
        target_softmax_out_temp)
    cov_matrix_t_temp = cov_matrix_t_temp / torch.sum(cov_matrix_t_temp, dim=1)

    mcc_loss = (torch.sum(cov_matrix_t_temp) - torch.trace(cov_matrix_t_temp)) / class_num

    return mcc_loss

################## RNA_g loss ####################
def compute_RNA_g_loss(feat_frac_1=None, feat_frac_2=None, feat_frac_3=None):
    loss = 0
    '''
    feat_frac_1 is the ratio between the mean features norms of modality_1 and modality_2
    
    Example
    logits_1, feat_1 = model_1(input_modality_1)
    logits_2, feat_2 = model_2(input_modality_2)
    feat_frac_1 = feat_1.norm(p=2, dim=1).mean() / feat_2.norm(p=2, dim=1).mean()
     
    It can be used also with more then two modalities, 
    in this case you can use also feat_frac_2 and feat_frac_3
    '''
    

    if feat_frac_1 is not None:
        loss += get_L2norm_loss_self_driven(feat_frac_1,1)
    if feat_frac_2 is not None:
        loss += get_L2norm_loss_self_driven(feat_frac_2,1)
    if feat_frac_3 is not None:
        loss += get_L2norm_loss_self_driven(feat_frac_3,1)
    return loss

def get_L2norm_loss_self_driven(x, radius, modality=None):
    loss = (x - radius) ** 2
    return loss


################## CENT loss ####################
def CENT_loss(output, p_label, classes=8):

    batch_size = len(p_label)
    output = F.softmax(output, dim=1)
    Yg = torch.gather(output, 1, torch.unsqueeze(p_label, 1))
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = output / Yg_.view(len(output), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = torch.ones(batch_size, classes).scatter_(1, p_label.view(batch_size, 1).data.cpu(), 0)
    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output)
    loss /= float(batch_size)
    loss /= float(classes)
    return loss


