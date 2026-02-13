import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class conj_function_deblur_Debulr_DRP(nn.Module):
    """
    performs DC step
    """

    def __init__(self, kernel_forward, degration_kernel, lam, dataform):
        super(conj_function_deblur_Debulr_DRP, self).__init__()
        self.A = kernel_forward[0]
        self.At = kernel_forward[1]
        self.B = degration_kernel[0]
        self.Bt = degration_kernel[1]
        self.lam = lam
        self.dataform = dataform

    def forward(self, im):  # step for batch image
        """
        :im: input image (B x nrow x nrol)
        """
        # ax = ndimage.filters.convolve(im.squeeze().permute(1, 2, 0).cpu(), np.expand_dims(self.kernel_forward.cpu().squeeze().cpu().numpy(), axis=2), mode='wrap')
        # atax = ndimage.filters.convolve(ax, np.expand_dims(self.kernel_forward.cpu().squeeze().cpu().numpy()[::-1, ::-1], axis=2), mode='wrap')
        # atax = torch.tensor(atax).permute(2, 0, 1).unsqueeze(0)
        atax = self.At(self.A(im.squeeze(0).permute(1, 2, 0).contiguous()))
        atax = atax.permute(2, 0, 1).contiguous().unsqueeze(0)

        # hx = self.B(im.squeeze()) # Equivalent operator Hx is used here
        # hthx = self.Bt(hx)
        hthx = BtB_function(im,self.B,self.Bt,self.dataform)
        return atax + self.lam * hthx
    
# class conj_function_deblur_SR_DRP(nn.Module):
#     """
#     performs DC step
#     """

#     def __init__(self, kernel_forward, sr_prior, lam, H_func, Ht_func,dataform):
#         super(conj_function_deblur_SR_DRP, self).__init__()
#         self.A = kernel_forward[0]
#         self.At = kernel_forward[1]
#         self.sr_prior = sr_prior
#         self.lam = lam
#         self.H = H_func
#         self.Ht = Ht_func
#         self.dataform = dataform

#     def forward(self, im):  # step for batch image
#         """
#         :im: input image (B x nrow x nrol)
#         """
#         # ax = ndimage.filters.convolve(im.squeeze().permute(1, 2, 0).cpu(), np.expand_dims(self.kernel_forward.cpu().squeeze().cpu().numpy(), axis=2), mode='wrap')
#         # atax = ndimage.filters.convolve(ax, np.expand_dims(self.kernel_forward.cpu().squeeze().cpu().numpy()[::-1, ::-1], axis=2), mode='wrap')
#         # atax = torch.tensor(atax).permute(2, 0, 1).unsqueeze(0)
#         atax = self.At(self.A(im.squeeze(0).permute(1, 2, 0).contiguous()))
#         atax = atax.permute(2, 0, 1).contiguous().unsqueeze(0).cpu()

        # hx = imresize(im.squeeze(), 1 / self.sr_prior, True) # Equivalent operator Hx is used here
#         # hthx = imresize(hx, self.sr_prior, True).unsqueeze(0)
#         hthx = HtH_function(im,self.H,self.Ht,self.sr_prior,self.dataform)
#         return atax + self.lam * hthx

    
class B_Bt_function(nn.Module):
    """
    universal B kernel function class
    """

    def __init__(self, kernel, conv_method):
        super(B_Bt_function, self).__init__()
        self.B_kernel = kernel
        self.conv_method = conv_method

    def conv_spatial(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        B_kenerl_expand = self.B_kernel.repeat(C, 1, 1, 1)
        Bx = F.conv2d(
            x,
            B_kenerl_expand,
            padding="same",
            groups=C  # Key parameter
        )
        return Bx

    def conv_fft(sefl,x):
        return

    def forward(self, x):  # step for batch image
        if self.conv_method == "conv_spatial":
            return self.conv_spatial(x)
        elif self.conv_method == "conv_fft":
            return self.conv_fft(x)
        return

# retormer motion blur function 
def B_function(x, B_kenerl):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    B_kenerl_expand = B_kenerl.repeat(C, 1, 1, 1)
    Bx = F.conv2d(
        x,
        B_kenerl_expand,
        padding="same",
        groups=C  
    )
    return Bx

def Bt_function(x,Bt_kenerl):
    if x.dim == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    Bt_kenerl_expand = Bt_kenerl.repeat(C, 1, 1, 1)
    Btx = F.conv2d(
        x,
        Bt_kenerl_expand,
        padding="same",
        groups=C  
    )
    return Btx

# Montion Deblur restoration for btb operation in correct orientation
def BtB_function(x,B,Bt,dataform):
    x = einops.rearrange(x, f"nb nz nx ny->nb {dataform}")
    BtBx = Bt(B(x))
    BtBx = einops.rearrange(BtBx, f"nb {dataform}->nb nz nx ny")
    return BtBx

def load_motion_deblur_Restormer(model_path):
    # ----------------------------------------
    # load model
    from models.restormer_arch import Restormer
    inp_channels = 3
    out_channels = 3
    dim = 48
    num_blocks = [4,6,6,8]
    num_refinement_blocks = 4
    heads = [1,2,4,8]
    ffn_expansion_factor = 2.66
    bias = False
    LayerNorm_type = "WithBias"
    dual_pixel_task = False
    
    model = Restormer(inp_channels=inp_channels, out_channels= out_channels, dim=dim, num_blocks=num_blocks,
                      num_refinement_blocks=num_refinement_blocks, heads = heads, ffn_expansion_factor=ffn_expansion_factor,
                      bias = bias, LayerNorm_type=LayerNorm_type,dual_pixel_task=dual_pixel_task )
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    
    return model

def test(img_lq, model):
    model.eval()
    with torch.inference_mode():
        output = model(img_lq)
    return output
