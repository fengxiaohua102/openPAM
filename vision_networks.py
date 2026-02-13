from __future__ import print_function
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
import warnings
warnings.filterwarnings("ignore")
import os
import torch.nn as nn

def load_model():
    model_path = os.getcwd() + '/model_zoo/ffdnet_gray.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------
    # load model
    n_channels = 1        # setting for grayscale image
    nc = 64               # setting for grayscale image
    nb = 15               # setting for grayscale image
    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model

def load_model_DRUNet():
    model_path = os.getcwd() + '/model_zoo/drunet_gray.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------
    # load model
    n_channels = 1        # setting for grayscale image
    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, 
                act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model

def load_model_Restormer():
    model_path = os.getcwd() + '/model_zoo/restormer/restormer_gaussian_gray_denoising_sigma25.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------
    # load model
    from models.restormer_arch import Restormer
    inp_channels=1 
    out_channels=1 
    dim = 48
    num_blocks = [4,6,6,8]
    num_refinement_blocks = 4
    heads = [1,2,4,8]
    ffn_expansion_factor = 2.66
    bias = False
    LayerNorm_type = 'BiasFree'   ## Other option 'BiasFree'
    dual_pixel_task = False   
    
    model = Restormer(inp_channels=inp_channels, out_channels= out_channels, dim=dim, num_blocks=num_blocks,
                      num_refinement_blocks=num_refinement_blocks, heads = heads, ffn_expansion_factor=ffn_expansion_factor,
                      bias = bias, LayerNorm_type=LayerNorm_type,dual_pixel_task=dual_pixel_task )
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model


