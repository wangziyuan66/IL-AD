import torch
import torch.nn.functional as F
def index_hanlder():
    can_list = list(reversed(list(range(40))))
    mod_list = list(reversed(list(range(40,60))))
    resorted_idx = []
    for i in range(6):
        for j in range(10):
            if j%5 ==2 or i == 2:
                resorted_idx.append(mod_list.pop())
            else:
                resorted_idx.append(can_list.pop())
    return resorted_idx


def mc_regularization(mod_outputs):
    c_related = mod_outputs[:,:,:].index_select(2,torch.tensor([1,5,9,13,8,9,42,10,11,12,13,43,14,15,17,21,25,29,33,37]).to("cuda:0")).to("cuda:0")
    m_related = mod_outputs[:,:,:].index_select(2,torch.arange(40,60).to("cuda:0")).to("cuda:0")
    return F.mse_loss(c_related,m_related)
