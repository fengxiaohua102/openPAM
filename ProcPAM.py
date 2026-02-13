#!/usr/bin/env python
# coding: utf-8
'''
# ** This script performs jitter correction for PA raw signals using PD measurment signals (if any)**
# ** and optimize the jitter further via unsupervised learning *****************************
# **    1: The raw ADC reading is normalized to the range [-1, 1] by subtracting 2048 and dividing 2048 
# **    2: Jitter correction (using PD if any)
# **    3: Store the processed signal and hilbert_transformed signals/images
'''

import torch
import numpy as np
from tifffile import imread
import os
import einops
from matplotlib import pyplot as plt
import time
from utils_module import numpy2cuda, cuda2numpy
import torch.nn as nn

torch.cuda.set_device(0)

def roll_col(mat, shifts):
    # function that roll the 2D cuda tensor on col direction with different shifts for each col.
    h, w = mat.shape    
    rows, cols = torch.arange(h), torch.arange(w)
    rows, cols = rows.cuda().long(), cols.cuda().long()
    shifted = torch.sub(rows.unsqueeze(1).repeat(1,w), shifts.repeat(h,1)) % h    
    return  mat[shifted.long(), cols.long()]

# **Perform hilbert transform on the aligned PA signals**
def hilbert_transform(cuda_tensor):
    # computes the hilbert transform of input tensor (cuda) shaped as [Nt, Nx, Ny]
    Nt = cuda_tensor.shape[0]
    cuda_tensor_fft = torch.fft.fft( torch.fft.fftshift(cuda_tensor, dim = 0), dim = 0, n = Nt)
    cuda_tensor_fft[Nt//2+1:,:,:] = 0.0
    cuda_tensor_fft[1:Nt//2,:,:] = 2.0 * cuda_tensor_fft[1:Nt//2,:,:]
    cuda_tensor_hilbert = torch.fft.ifftshift( torch.fft.ifft(cuda_tensor_fft, dim = 0, n = Nt), dim = 0)
    return torch.abs(cuda_tensor_hilbert)

def xcorr_func(PD_sig_cuda):
    # using conv1d to find the shifts, the input should be in [Nt, Nx]
    Nz, Nx = PD_sig_cuda.shape
    weight_kernel = PD_sig_cuda[:,10].unsqueeze(0)  # the reference PD signal, shaped into [1, Nt]
    weight_kernel = torch.mean(PD_sig_cuda, dim=1).unsqueeze(0)  # the reference PD signal, shaped into [1, Nt]
    weight_kernel = weight_kernel - torch.min(weight_kernel)
    PD_sig_cuda = einops.rearrange(PD_sig_cuda,'nt nxny -> nxny nt').unsqueeze(0) # shaped into [1, Nt, nxny]
    PD_sig_cuda = PD_sig_cuda - torch.min(PD_sig_cuda)
    corr_res = torch.nn.functional.conv1d(PD_sig_cuda, weight_kernel.repeat(Nx,1,1), padding = (Nz-1)//1, stride = 1, groups = Nx)
    max_PD = torch.argmax(corr_res.squeeze(0), dim = 1)
    N_shift = torch.max(max_PD) - max_PD     # the amount of shift w.r.t. the reference (max)
    # PD_sig_cuda = einops.rearrange(PD_sig_cuda.squeeze(0), 'nxny nt -> nt nxny')
    return N_shift, corr_res

def jitter_correc(PA_file, PD_file):    
    # store_fname = '/jitter_test/openPAM_sig' + str(n_f) + '_test.tif'
    st = time.perf_counter()
    PA_sig = imread(PA_file)
    PD_sig = imread(PD_file)
    stop_t = time.perf_counter()
    print('Reading raw data takes:', stop_t - st)
    
    Nz, Nx, Ny = PA_sig.shape # Shape determined by actual measurement
    ax = plt.figure(1,figsize = (15,5))
    ax.add_subplot(1,2,1)
    plt.imshow(PA_sig[100:,:,800])
    plt.title('Raw PA signal')
    ax.add_subplot(1,2,2)
    plt.imshow(PD_sig[100:,:,800])
    plt.title('Raw PD signal')
    
    # **Signal alignment of the PD signals using cross correlations**
    PD_sig_cuda = einops.rearrange( numpy2cuda(PD_sig*1.0), 'nz nx ny -> nz (nx ny)')
    PA_sig_cuda = einops.rearrange( numpy2cuda(PA_sig*1.0), 'nz nx ny -> nz (nx ny)')
    PD_sig_cuda = PD_sig_cuda - torch.min(PD_sig_cuda)
    PD_sig_cuda = PD_sig_cuda / torch.max(PD_sig_cuda, dim = 0)[0]
    st = time.perf_counter()
    N_shift_PD2, corr_res = xcorr_func(PD_sig_cuda)
    # max_PD = torch.argmax(PD_sig_cuda, dim = 0)
    # N_shift_PD2 = torch.max(max_PD) - max_PD     # the amount of shift w.r.t. the reference (max)
    stop_t = time.perf_counter()
    print('cross correlatoin time is:', stop_t - st)
    
    # In[9]:
    st = time.perf_counter()
    PA_sig_align2 = roll_col(PA_sig_cuda, torch.Tensor(N_shift_PD2).view(1, -1).cuda() )
    PD_sig_align2 = roll_col(PD_sig_cuda, torch.Tensor(N_shift_PD2).view(1, -1).cuda() )  
    stop_t = time.perf_counter()
    print('Jitter correction time using indexing is:', stop_t - st)
    
    ax3 = plt.figure(2,figsize = (15,5))
    ax3.add_subplot(1,2,1)
    plt.imshow(cuda2numpy(einops.rearrange(PA_sig_align2, 'nz (nx ny) -> nz nx ny', ny = Ny)[100:,:,800]))
    plt.title('Jitter corrected Raw PA signal')
    ax3.add_subplot(1,2,2)
    plt.imshow(cuda2numpy(einops.rearrange(PD_sig_align2, 'nz (nx ny) -> nz nx ny', ny = Ny)[100:,:,800]))
    plt.title('Jitter corrected Raw PD signal')
    # fig = px.imshow(cuda2numpy(PD_sig_align2[200:-150,:,800]), color_continuous_scale='hot')
    # fig.show()
 
    # store the aligned results
    PA_sig_align2 = einops.rearrange(PA_sig_align2, 'nz (nx ny) -> nz nx ny', ny = Ny)
    PD_sig_align2 = einops.rearrange(PD_sig_align2, 'nz (nx ny) -> nz nx ny', ny = Ny)
    
    torch.cuda.empty_cache()
    return PA_sig, PA_sig_align2
      

def tv1d_loss(img):
    # loss = torch.mean(torch.abs(img[:-1,:] - img[1:,:]))
    loss = torch.mean(torch.abs(img[:-2,:] - 2*img[1:-1,:] +img[2:,:]))
    return loss

def tv2d_loss(img):
    hloss = torch.mean(torch.abs(img[:,:,:-1] - img[:,:,1:]))
    vloss = torch.mean(torch.abs(img[:,:-1,:] - img[:,1:,:]))
    loss = hloss+vloss
    return loss

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=200):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.25**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def jitter_correct3D(PA_file, store_file, PD_file=None):
    # performs jitter correction for 3D PAM measurment data in the format of x-y-t
    # Note that 2048 here is the zero for the 12-bit DAQ sampling value.
    bd_pix = 10   # boundary pixels for x-y-t data
    store_fname = store_file
    # process the raw DAQ data into the range of [-1, 1]
    if(PD_file is not None):# set initial value = jitter corrected PA_sig using PD_sig
        PA_sig, PA_sig_align_small = jitter_correc(PA_file, PD_file)
        PA_sig = (PA_sig - 2048.0)/2048.0
        PA_sig_align_small = (PA_sig_align_small - 2048.0)/2048.0  # Apply Gaussian convolution shift method for jitter correcting. No learning here, just using PD signal as convolution kernel parameter
    else: # set initial value = original PA_sig
        st = time.perf_counter()
        PA_sig = imread(PA_file)
        # PA_sig = PA_sig # crop
        # PA_sig = PA_sig[:,250:-250,250:-250]
        PA_sig = PA_sig[:,500:-500,500:-500]
        stop_t = time.perf_counter()
        print('Reading raw data takes:', stop_t - st)
        PA_sig_align_small = numpy2cuda( (PA_sig - 2048.0)/2048.0 )
       
    nt, nx, ny = PA_sig.shape
    
    # formulating the jitter correction problem as a optimizaiton problem 
    # now optimize the u_array
    # ****************************
    # PA_sig_align_small = numpy2cuda(PA_sig)  # directly optmizing from the raw PA signal (without PD jitter correction) selection of training initial values
    PA_sig_align = einops.rearrange(PA_sig_align_small, 'nt nx ny -> nt (nx ny)')
    u_array = torch.zeros((nx*ny,1), dtype=torch.float32, device='cuda')
    u_array_param = nn.Parameter(u_array)
    x = (torch.linspace(0,nt-1,nt)-nt/2).view(1,-1).float().cuda()# referred center-posed time idx 
    PA_sig_align = einops.rearrange(PA_sig_align,'nt nxny -> nxny nt').unsqueeze(0) # shaped into [1, nxny, nt]
    
    LR = 1e-0
    sigma = 0.8
    optimizer_r = torch.optim.Adam([u_array_param], lr= LR) # Optimize the mean (shift) parameter of the Gaussian convolution kernel
    num_iter = 100
    loss_trend = np.zeros((num_iter, 1))
    st = time.perf_counter()
    for j in range(num_iter):
        torch.cuda.empty_cache()
        # optimizer_r = exp_lr_scheduler(optimizer_r, j, init_lr= LR, lr_decay_epoch=50)
        
        # the computational graph
        # ************************************
        # 1: generation of the convolution kernel
        kernel = torch.exp(-0.5*( x - u_array_param)**2/sigma**2) #* 1/(np.sqrt(2*np.pi) *sigma)  # shape: [nx, nt]
        kernel = kernel.unsqueeze(1)           # shaped into [nx, 1, nt]  for batch-wise(nx) group convolution
        kernel = kernel / torch.sum(kernel, dim=2).unsqueeze(-1) # nn.functional.softmax(kernel, dim=2) 替代方法
        
        # 2: channel-wise convolution
        conv_res = torch.nn.functional.conv1d(PA_sig_align, kernel, padding = 'same', stride = 1, groups = nx*ny)
        conv_res = conv_res.squeeze(0)# [nx nt]
        
        # 3: compute loss (objective)
        loss_r = torch.mean( torch.abs( conv_res - PA_sig_align ) ) # data fidelity term
        loss_tv = 1.0 * tv2d_loss( einops.rearrange(conv_res, '(nx ny) nt->nt nx ny', nx = nx) ) # regulation term
        
        # 4: backprop
        optimizer_r.zero_grad()
        (loss_tv).backward()   
        if(np.mod(j,20) == 0):# verbose
            print ('Iteration %05d Loss %f reg_loss %f' % (j, loss_r.item(), loss_tv.item()),'\n', end='')
        loss_trend[j] = loss_r.item() + loss_tv.item() #可以计算数据保证项和显式正则作为obj_val
        optimizer_r.step()
        
    stop_t = time.perf_counter()
    print('Jitter optimization time is:', stop_t -st)
    
    #
    PA_sig_opt = cuda2numpy( einops.rearrange(conv_res, '(nx ny) nt->nt nx ny', nx = nx).detach() )
    ax3 = plt.figure(3, figsize = (20,20))
    ax3.add_subplot(2,3,1)
    plt.imshow((PA_sig[bd_pix:-bd_pix,:,ny//2]))
    plt.title('Org PAM')
    ax3.add_subplot(2,3,2)
    PA_sig_align_np = cuda2numpy( einops.rearrange(PA_sig_align.squeeze(0), '(nx ny) nt->nt nx ny', nx = nx) )
    plt.imshow(PA_sig_align_np[bd_pix:-bd_pix,:,ny//2]) 
    plt.title('Jitter corrected Raw PA signal')
    ax3.add_subplot(2,3,3)
    plt.imshow(PA_sig_opt[bd_pix:-bd_pix,:,ny//2])
    plt.title('Optimally + Raw PA signal')
    
    # make the sigma smooth to a dirac delta function
    sigma_opt = 0.1
    kernel_opt = torch.exp(-0.5*( x - u_array_param)**2/sigma_opt**2) * 1/(np.sqrt(2*np.pi) *sigma_opt)   # shape: [nx, nt]
    kernel_opt = kernel_opt.unsqueeze(1)           # shaped into [nx, 1, nt]
    kernel_opt = kernel_opt / torch.sum(kernel_opt, dim=2).unsqueeze(-1)
    
    # channel-wise convolution
    conv_res_opt = torch.nn.functional.conv1d(PA_sig_align, kernel_opt, padding = 'same', stride = 1, groups = nx*ny)
    conv_res_opt = conv_res_opt.squeeze(0) # [nx nt]kernel = kernel / torch.sum(kernel, dim=2).unsqueeze(-1)
    PA_sig_opt = cuda2numpy(einops.rearrange(conv_res_opt, '(nx ny) nt->nt nx ny', nx = nx).detach())
    
    # ax3.add_subplot(2,3,5)
    # plt.imshow(cuda2numpy(kernel_opt.squeeze(1).detach()[::200,:]))
    # plt.title('Optimally correction kernel')
    
    ax3.add_subplot(2,3,6)
    plt.imshow(PA_sig_opt[bd_pix:-bd_pix,:,ny//2])
    plt.title('Optimally corrected Raw PA signal')
    
    from tifffile import imwrite
    imwrite(store_fname + '.tif', PA_sig_opt)
    PA_sig_hil = hilbert_transform( (numpy2cuda(PA_sig_opt)) )
    imwrite(store_fname + '_hilbert.tif', cuda2numpy( PA_sig_hil) ) # What is the result after Hilbert transform?

    # plt.figure()
    # plt.plot(np.log10(loss_trend))
    # plt.figure()
    # plt.imshow(cuda2numpy(conv_res.detach()).T - cuda2numpy(PA_sig_test))
    # plt.title('Error PAM')
            
    u_opt = cuda2numpy(u_array_param.detach())
    print('u_opt range is:', np.max(u_opt), np.min(u_opt))
    torch.cuda.empty_cache()
    # plt.figure()
    # plt.imshow(cuda2numpy(kernel_fix.squeeze(1).detach()[::200,:]))

if __name__ == '__main__':
    dataset_dir = os.getcwd() + r"/ExpData_DH/"
    imagingsession_dir = dataset_dir + r"rawdata_ear"     # folder of the data acquired in an imaging session
    raw_dataset_dir = imagingsession_dir + r"/raw"         # dataset dir that contains all the experimental data

    N_samples = 2 # number of raw measurements to be processed
   
    for K in range(N_samples):
        save_dir = imagingsession_dir + '/jitter_corrected_results/'+ str(K+1) + '/'        # specify where to store the results
       
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        PA_file = raw_dataset_dir + str(K+1) +f'/ears2r5wp{str(K+1)}PA.bin.tif'   # the name of the openPAM measurement (if any)
        # PD_file = raw_dataset_dir + str(K+1) +f'/ears2r5wp{str(K+1)}PD.bin.tif' # the name of the photodiode measurement (if any)
        store_file_name = save_dir + 'openPAM_sig' + str(K+1) 
        jitter_correct3D(PA_file, store_file_name, PD_file=None)
        
