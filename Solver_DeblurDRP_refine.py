import os.path
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from utils_DRP import utils_conj as conj
from utils_DRP.utils_DRP import *

import einops
import numpy as np
import torch
from vision_networks import *
from utils_module import pad2modulo, norm1, numpy2cuda, cuda2numpy
import os
import tiffile

# restormer motion deblurring
from utils_DRP.util_motion_deblur import *
# extra denoising
from Solver_FISTA_PnP import denoise_net_drp, proxl1

# os.environ["CUDA_VISIBLE_DEVICES"]  = "3"
device = torch.device('cuda')

def power_iter(A, At, input_shape):
    max_iter = 100
    b0 = torch.rand(input_shape).float().cuda()
    b0 = b0/np.sqrt(normNdMatrix(b0))
    u_old = 0
    iter=0
    cond = True
    while(cond):
        iter=iter+1
        if(iter==1):
            b_old = b0
            u_old = 1.0
        
        b_new = At(A(b_old)) 
        b_old_temp = b_old.flatten()
        u_new = torch.dot(b_old_temp, b_new.flatten() )
        b_new = b_new/np.sqrt( normNdMatrix(b_new) )
        if(iter>1):
            if( (u_new-u_old)/u_new<1e-3 or iter>=max_iter):
                cond = False

        b_old = b_new
        u_old = u_new

    max_egival = u_new
    return max_egival.cpu().item()

def save_tensor_as_tiff(tensor, filename, normalize=False):
    # Input validation
    assert tensor.dim() == 4 and tensor.shape[0] == 1, "Input must be [1,C,H,W] shape"
    
    # Dimension conversion
    hwc_tensor = einops.rearrange(tensor.squeeze(0),"c h w->h w c")
    
    # Convert to NumPy array
    np_array = hwc_tensor.detach().cpu().numpy()
    
    # Normalization processing
    if normalize:
        np_array = (np_array-np_array.min())/(np_array.max()-np_array.min())
    # Save TIFF
    tiffile.imwrite(
        filename,
        data=np_array,
        dtype=np.float32,
        compression=None
    )
    return

def normNdMatrix(x):
    norm_val_temp = torch.square(torch.abs(x))
    norm_val = torch.sum(norm_val_temp).cpu().item()
    return norm_val

def func(xk, b, A):
    e = b.cuda() - torch.reshape(A(xk), b.shape)
    Fx = 0.5*normNdMatrix(e)
    return Fx

def norm01_3d(x):
    min_val = torch.min(x)   # for addressing the nonnegativity of the denoiser
    x = x - min_val  
    max_val = torch.max(x)
    x = x/max_val
    return x, max_val, min_val

def denorm01_3d(x,max_val,min_val):
    return x * max_val + min_val

def norm_mean_scale(x, img_range=1.0):
    mean_val = torch.mean(x)          # Dynamically calculate the mean of the current tensor
    x_norm = x - mean_val            # Centralization: subtract the mean
    x_norm = x_norm * img_range      # Scaling: multiply by a preset coefficient
    return x_norm, mean_val, img_range 

def denorm_mean_scale(x_norm, mean_val, img_range):
    x_restore = x_norm / img_range    # Inverse scaling: divide by the scaling coefficient
    x_restore = x_restore + mean_val # Inverse centralization: add the mean
    return x_restore

def Solver_Deblur_DRP_refine(A, At, b, opt):

    print(' - Running DRP_Deblur_refine\n')
    vis = opt['vis']
    POSCOND= opt['POScond']

    # Read parameters from opt
    step = opt.get('step', 0.1)           # step size (γ in paper)
    reg_lam = opt.get('reg_lam', 1.0)     # regularization lambda (τ in paper)
    data_form = opt['dataform']           # restoration orientation
    maxiter = opt.get('maxiter', 60)      # max iterations
    sigma_n = opt.get('sigma_n', 25/255)  # noise level for denoising
    save_optimal = opt.get('save_optimal', True)  # whether to save optimal obj result 

    obj_val = []
    if save_optimal:
        obj_value_min = 1e10
    # --------------------------------
    # get img_L
    # --------------------------------
    img_L = b.clone() # key step for accurately calculating the obj_val !!!!

    # --------------------------------
    # motion deblurring prior path
    # --------------------------------
    model_path = './model_zoo/restormer/motion_deblurring.pth'

    # --------------------------------
    # initialize x, and pre-calculate Blur operator
    # --------------------------------
    #aty
    aty = At(img_L)
    aty = einops.rearrange(aty, 'nx ny nz->nz nx ny').unsqueeze(0).cuda()

    x = einops.rearrange(img_L, f'nx ny nz->{data_form}').unsqueeze(0).clone()
    # K_xy_tensor = einops.rearrange(K_xy, 'nx ny nz->nz nx ny').unsqueeze(0)

    # h, w = x.shape[-2], x.shape[-1]

    from dataclasses import dataclass 
 
    @dataclass 
    class KernelParams:
        r_angle: int = 0 
        thick: int = 1
        rescale: float = 1.0 
        bk_len: int = 5
    
    # Read kernel parameters from opt and convert dict to dataclass
    kernel_paras_config = opt.get('kernel_paras', [
        {'bk_len': 3, 'r_angle': 90, 'thick': 1, 'rescale': 1.0},
        {'bk_len': 1, 'r_angle': 90, 'thick': 1, 'rescale': 1.0}
    ])
    kernel_paras = [KernelParams(**config) for config in kernel_paras_config]
    
    # blur kernels definition and refinement
    B_list = []
    for k_para in kernel_paras:
        # degratioin blur kernel definition
        B_kernel = np.ascontiguousarray(generate_motion_blur_kernel(k_para.r_angle, k_para.bk_len, k_para.thick))
        Bt_kernel = np.ascontiguousarray(generate_adjoint_kernel(B_kernel))

        B_kernel = torch.from_numpy(B_kernel).float().cuda().unsqueeze(0).unsqueeze(0)
        Bt_kernel = torch.from_numpy(Bt_kernel).float().cuda().unsqueeze(0).unsqueeze(0)

        # degration operator
        B = B_Bt_function(B_kernel, "conv_spatial")
        Bt = B_Bt_function(Bt_kernel, "conv_spatial")
        B_list.append((B,Bt))

    #DEFINE motion deblur model
    deblur_model = load_motion_deblur_Restormer(model_path)
    deblur_model = deblur_model.to(device)   

    # DEFINE extra denoising method based on opt['denoiser']
    denoiser_type = opt.get('denoiser', 'proxl1')
    
    if denoiser_type == 'drunet':
        network = load_model_DRUNet()
        opt_temp = {'denoiser': 'drunet'}
        denoise = lambda noise_x: denoise_net_drp(network, opt_temp, noise_x, sigma_n)
    elif denoiser_type == 'ffdnet':
        network = load_model()
        opt_temp = {'denoiser': 'ffdnet'}
        denoise = lambda noise_x: denoise_net_drp(network, opt_temp, noise_x, sigma_n)
    elif denoiser_type == 'restormer':
        network = load_model_Restormer()
        opt_temp = {'denoiser': 'restormer'}
        denoise = lambda noise_x: denoise_net_drp(network, opt_temp, noise_x, sigma_n)
    elif denoiser_type == 'proxl1':
        denoise = lambda noise_x: proxl1(noise_x, sigma_n)
    elif denoiser_type == 'proxtv':
        denoise = lambda noise_x: denoise_ProxTV3D(noise_x, sigma_n)
    elif denoiser_type == 'none':
        denoise = None
    else:
        raise ValueError(f"Unknown denoiser type: {denoiser_type}")
    # --------------------------------
    # main iterations
    # --------------------------------
    for i in range(maxiter):

        x_pre = einops.rearrange(x, f'nb {data_form}->nb nz nx ny').clone()

        if i < maxiter // 2:
            B,Bt = B_list[0]
        else:
            B,Bt = B_list[1]

        z_lq = B(x).to(device)
        z_input = z_lq.clone()

        # extra denoising opt xy denoising
        if denoise != None:
            z_lq = einops.rearrange(z_lq, f'nb {data_form}->nb nz nx ny')
            z_lq = denoise(z_lq.squeeze(0)).unsqueeze(0)
            z_lq = einops.rearrange(z_lq, f'nb nz nx ny->nb {data_form}')

        # slice-wise drp process
        with torch.no_grad():
            z_lq_temp, mean_val, img_range = norm_mean_scale(z_lq)
            # Pre-allocate RGB buffer to avoid repeated creation
            z_lq_slice_RGB = torch.empty((1, 3, z_lq_temp.shape[2], z_lq_temp.shape[3]), 
                                          dtype=z_lq_temp.dtype, device=z_lq_temp.device)
            # get channels mean from deblur result
            for z_id in range(z_lq_temp.shape[1]):
                # Expand single channel to 3 channels, then average and compress back to single channel
                z_lq_slice_RGB.copy_(z_lq_temp[:,z_id,:,:].unsqueeze(0).expand(-1, 3, -1, -1))
                x_slice_RGB = test(z_lq_slice_RGB, deblur_model)
                x_slice_GRAY = torch.mean(x_slice_RGB, dim=1, keepdim=True)
                x[:,z_id,:,:] = x_slice_GRAY
            # get individual channel from deblur result
            # for z_id in range(z_lq.shape[1]):
            #     z_lq_slice_RGB = z_lq[:,z_id,:,:].unsqueeze(1)
            #     z_lq_slice_RGB = torch.cat([z_lq_slice_RGB]+[torch.zeros_like(z_lq_slice_RGB)]*2, dim =1)
            #     x_slice_RGB = test(z_lq_slice_RGB, deblur_model)
            #     x_slice_GRAY = x_slice_RGB[:,0,:,:].unsqueeze(1)
            #     x[:,z_id,:,:] = x_slice_GRAY
            x = denorm_mean_scale(x, mean_val, img_range)
         
        # Clean up temporary tensors
        if vis > 0:
            del z_input, z_lq
            
        x = einops.rearrange(x, f'nb {data_form}->nb nz nx ny')

        # refinement step
        if i < maxiter // 2:
            z = step*reg_lam * x + (1 - step*reg_lam) * x_pre
        else:
            z = 0.5*step*reg_lam * x + (1 - 0.5*step*reg_lam) * x_pre

        # z = x
        #CG method implementing scaled proximal operator
        function = conj_function_deblur_Debulr_DRP(kernel_forward = [A,At], degration_kernel=[B,Bt], lam=step, dataform=data_form) 
        resb = step * BtB_function(z,Bt,B,data_form) + aty
        conj_cal = conj.ConjGrad(resb, function, max_iter=3, l2lam=0, verbose=False)
        x = conj_cal(z)

        if(POSCOND):
            x[x<0]=0 # positiveness constraint

        if vis > 0 :
            obj_value = func(x.squeeze(0).permute(1, 2, 0).contiguous(),b,A)
            
            if save_optimal and obj_value_min and (obj_value_min > obj_value):
                obj_value_min = obj_value
                out_min_obj = x.clone()
                obj_val_min_step = i
            
            print('\nIter:', str(i), f'obj_val is: {obj_value:.8f}',end=' ')
            obj_val.append(obj_value)

        x = einops.rearrange(x, f'nb nz nx ny->nb {data_form}')
        
        # early stop
        if(i>=maxiter*0.8 and abs(obj_val[-1]-obj_val[-3])/obj_val[-1]<=opt['tol'] ):
            print('Solution stagnates, exit ...')
            break

    out_final = einops.rearrange(x, f'nb {data_form}->nb nz nx ny')

    if save_optimal:
        out_save = [out_final, out_min_obj]
        print(f'\nOptimal Obj_Val: Iter= {obj_val_min_step}  Obj_val= {obj_value_min:8f}')
    else:
        out_save = [out_final]
    
    return [x.squeeze(0).permute(1, 2, 0).contiguous() for x in out_save], obj_val

class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image, reg_param):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        # self.regularization_term = tv_loss()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image
        self.reg_param = reg_param
        
    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + self.reg_param * tv3d_loss(self.clean_image) \
              + self.reg_param*2e-1 * torch.mean(torch.abs(self.clean_image))

    def get_clean_image(self):
        return self.clean_image
 
def denoise_ProxTV3D(noisy_image, sigma_hat):
    # define the total variation denoising network
    # noisy_image input is in shape of Nz, Nx, Ny, 
    tv_denoiser = TVDenoise(noisy_image, sigma_hat)
    
    # define the optimizer to optimize the 1 parameter of tv_denoiser
    # optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr = 0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(tv_denoiser.parameters(), lr= 1e-4)
    # num_iters = 50
    optimizer = torch.optim.Adam(tv_denoiser.parameters(), lr= 1e-3)
    num_iters = 80
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = tv_denoiser()
        # if i % 25 == 0:
        #     print("Loss in iteration {} of {}: {:.6f}".format(i, num_iters, loss.item()))
        loss.backward()
        optimizer.step()
    # print("TV Loss: {:.6f}".format(loss.item()))

    img_clean = tv_denoiser.get_clean_image().detach()
    return img_clean

def tv3d_loss(img):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (C, H, W) holding an input image.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
    """
    w_variance = torch.sum(torch.pow(img[:,:,:-1] - img[:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:-1,:] - img[:,1:,:], 2))
    z_variance = torch.sum(torch.pow(img[:-1,:,:] - img[1:,:,:], 2))
    loss = torch.sqrt(h_variance + w_variance + 2*z_variance) 
    return loss
