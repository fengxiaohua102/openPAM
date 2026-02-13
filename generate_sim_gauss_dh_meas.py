# -*- coding: utf-8 -*-
"""
Neural deconvolution of openPAM
This script performs neural deconvolution of openPAM based on PnP-FISTA framework, which uses neural denoisers for regularization
Note that the psf_stack is positive, while ht and PAM signals are zero-centered
Please read the comments of the code and adjust the settings accordingly to suit your own needs
"""
import torch
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import skimage
import os
import scipy.io as sio
import tifffile
import time
import matplotlib as mpl
from openPAM_Operators import *
from Solver_FISTA_PnP import *
from utils_module import pad2modulo, norm1, norm01, hilbert_transform, numpy2cuda, cuda2numpy
from einops import rearrange
from autoAlign import PSF_scale_z
import einops
import torch.nn.functional as F

from scipy.stats import poisson
from scipy.interpolate import interp1d

def np_imresize(array_in, r_scale):
    # resize the image along the first two dimensions
    nx_resize = np.int32(np.round(r_scale * array_in.shape[0]))
    ny_resize = np.int32(np.round(r_scale * array_in.shape[1]))
    array_torch = numpy2cuda(array_in)
    array_torch = einops.rearrange(array_torch, '(c h) w nb ->nb c h w', c=1)
    array_out = F.interpolate(array_torch, (nx_resize, ny_resize), mode='bicubic')
    array_out = cuda2numpy( einops.rearrange(array_out, 'nb c h w ->(c h) w nb', c= 1) )
    return array_out

def np_imresize1D(array_in, r_scale_z):
    # resize the image along z dimension
    nz_resize = np.int32(np.round(r_scale_z * array_in.shape[2]))
    nxy_resize = array_in.shape[0] * array_in.shape[1]
    array_torch = numpy2cuda(array_in)
    array_torch = einops.rearrange(array_torch, '(nx nb c) ny nz ->nb c (nx ny) nz', c=1, nb = 1)
    array_out = F.interpolate(array_torch, (nxy_resize, nz_resize), mode='bicubic') #interpolate
    array_out = cuda2numpy( einops.rearrange(array_out, 'nb c (nx ny) nz ->(nb c nx) ny nz', nx = array_in.shape[0] ) )

    # adjust the rescaled z slice num to even number
    cnt_slice = array_out.shape[2]//2+1
    z_num = cnt_slice*2
    array_out = array_out[:,:, cnt_slice - z_num//2: cnt_slice - z_num//2 + z_num]#???

    return array_out

def resize_1d_pytorch(array_in: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Use PyTorch's interpolation function to resize a 1D NumPy array.

    This method mimics the process of interpolating images (multidimensional data) by temporarily
    expanding the 1D data to 4D, using 'bicubic' or 'bilinear' interpolation, and then restoring it to 1D.

    Args:
        array_in (np.ndarray): Input 1D NumPy array with shape [Nt].
        scale_factor (float): Scaling factor. Greater than 1 for upsampling, less than 1 for downsampling.

    Returns:
        np.ndarray: Resized 1D NumPy array.
    """

    if not isinstance(array_in, np.ndarray) or array_in.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array.")
    
    nt_original = array_in.shape[0]

    array_torch = torch.from_numpy(array_in).float()

    array_torch = array_torch.view(1, 1, 1, nt_original)
    
    nt_resize = int(round(nt_original * scale_factor))

    array_out_torch = F.interpolate(array_torch, size=(1, nt_resize), mode='bicubic', align_corners=False)

    array_out = array_out_torch.squeeze().numpy()
    
    return array_out

def get_dh_gauss_PSF_ht(dataset_dir, conf, psf_name_list):
    psf_stack_list = []

    for psf_name in psf_name_list:
        # note that tifffile read different from matlab: depth first here
        # read the experimental psf: psf stack and the transducer ht
        psf_stack = tifffile.imread( dataset_dir + psf_name + '_psf_stack_z2um_0.2NA.tif')
        psf_stack = psf_stack.transpose(1,2,0) # for different view of saved psf_stack
        psf_stack = norm01(psf_stack)

        # psf_stack = psf_stack[:,:,::3]
        # interplote z dimension to enlarge psf z range or reduce dz
        # psf_stack = np_imresize1D(psf_stack, 2)

        # psf_stack = PSF_scale_z(psf_stack, conf)
        
        # psf_stack and ht normliazation 
        psf_stack = torch.from_numpy( psf_stack ).float()
        psf_stack[psf_stack<0.0] = 0.0
    
        psf_stack = psf_stack ** conf['nonlinear_pow']
        for K in range(psf_stack.shape[2]):
            psf_stack[:,:,K] = psf_stack[:,:,K] / torch.sum(psf_stack[:,:,K])  #[Nx, Ny, Nz]

        if(conf['transform'] =='lr'):
            psf_stack = torch.fliplr( psf_stack )  # FLIP the psf upsidedown: axis-0 for up, axis-1 for lr
        elif(conf['transform'] =='ud'):
            psf_stack = torch.flipud(psf_stack)  # FLIP the psf upsidedown: axis-0 for up, axis-1 for lr
        elif(conf['transform'] =='inv'):
            psf_stack = torch.fliplr( torch.flipud(psf_stack) )  # FLIP the psf upsidedown: axis-0 for up, axis-1 for lr
        
        psf_stack_list.append(psf_stack)
    
    #******************************
    # the transducer psf (withut evenlop extraction)
    ht = tifffile.imread( dataset_dir + 'ht.tif')
    # ht = resize_1d_pytorch(ht,2.0)
    # tifffile.imwrite('ht_2.0.tif',ht)
    ht = np.reshape(ht, (1, -1))
    ht = torch.from_numpy(ht.copy()).float()
    #********* normalized the ht
    ht = ht - torch.mean(ht)
    ht = ht/torch.sum(ht**2) 

    return psf_stack_list, ht 

def add_noise_flexible(data: np.ndarray, gaussian_std: float=0.0/255, poisson_intensity: float=0) -> np.ndarray:
    """
    Flexibly add Poisson and/or Gaussian noise to a NumPy array.
    
    This function first adds Poisson noise (if specified), then adds Gaussian noise.
    This order is typically used to simulate image sensor noise (photon shot noise precedes readout noise).

    :param data: Input np.ndarray data, range should be [-1, 1].
    :param gaussian_std: Standard deviation of Gaussian noise. If 0, no Gaussian noise is added.
    :param poisson_intensity: Intensity factor for Poisson noise. If 0, no Poisson noise is added.
                             This value is used to map the [-1, 1] signal to the expected event count (lambda).
                             Larger values mean higher SNR, so Poisson noise impact is relatively smaller.
    :return: Data with added noise, clipped to [-1, 1] range.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy ndarray.")

    # Use .copy() to avoid modifying the original input array
    noisy_data = data.copy()

    # --- Step 1: Optionally add Poisson noise ---
    if poisson_intensity > 0.0:
        # Poisson noise requires non-negative event rates (lambda), so we need to shift data from [-1, 1] to [0, 2]
        # 1. Shift data from [-1, 1] to [0, 2]
        data_shifted = noisy_data + 1.0
        
        # 2. Multiply by intensity factor to get expected lambda for Poisson distribution.
        # This lambda now represents the average number of "events" corresponding to signal intensity.
        lam = data_shifted * poisson_intensity
        
        # 3. Generate random numbers from Poisson distribution based on lambda (noisy "event" counts).
        # np.random.poisson generates a corresponding random integer for each element in the lam array.
        noisy_counts = np.random.poisson(lam)
        
        # 4. Convert noisy "event" counts back to original signal range.
        # First divide by intensity factor, then shift back to [-1, 1] range.
        noisy_data = (noisy_counts / poisson_intensity) - 1.0

    # --- Step 2: Optionally add Gaussian noise ---
    # This step is executed if Gaussian std is > 0, applied to original or Poisson-noised data.
    if gaussian_std > 0.0:
        # Generate Gaussian noise with mean 0 and std dev gaussian_std
        noise = np.random.normal(loc=0.0, scale=gaussian_std, size=noisy_data.shape)
        # Add Gaussian noise to data
        noisy_data = noisy_data + noise
    
    # --- Step 3: Clip final result to [-1, 1] range ---
    noisy_data = np.clip(noisy_data, -1.0, 1.0)
    
    return noisy_data.astype(np.float32)

def downsample_ht_array(data: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return data
    # Calculate sampling step
    # Use round to handle floating point precision issues, ensuring closest integer step
    step = int(round(1.0 / factor))
    
    # Ensure step is at least 1
    if step < 1:
        step = 1

    # Use slicing for downsampling
    # '...' means selecting all preceding dimensions, ':' means from start to end, step is the step size
    # This way gracefully handles multiple shapes like (nt,), (1, nt) etc.
    resampled_data = data[..., ::step]

    return resampled_data

def generate_measurement(openPAM_gt, psf_stack, ht, shift_opt, opts):
    # Input
    # openPAM_gt: the openPAM raw signal, should be in the range [-1, 1]     
    save_filename = opts['save_filename']
    
    # %% Now perform ROI extraction to reduce memory burden.
    # extract the region for faster deconv results
    PAM_GT = openPAM_gt #[:, 0:1000, 0:1000], we used all the meausrement data by default
    
    # extract the z ROI
    # PAM_proj2D_x = np.sum(np.abs(PAM_new_ROI), axis = 1)
    # PAM_proj2D_y = np.sum(np.abs(PAM_new_ROI), axis = 2)
    # # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(PAM_proj2D_x, cmap = mpl.colormaps['hot'])
    # plt.subplot(1,2,2)
    # plt.imshow(PAM_proj2D_y, cmap = mpl.colormaps['hot'])
    
    # %% Extract the corresponding optical psfs for the PAM image
    PAM_GT = einops.rearrange(PAM_GT, 'nz nx ny->nx ny nz' )      # change to the shape of Nx, Ny, Nt/Nz
    
    # pad the input data
    pad_nx, pad_ny = psf_stack.shape[0], psf_stack.shape[1]
    PAM_GT_pad = np.pad(PAM_GT, ((pad_nx, pad_nx), (pad_ny, pad_ny), (0,0)) )

    # pad nz if there is signal in the edge of the z dimension
    # pad_nz = ht.shape[-1] + 1
    pad_nz = int(np.ceil(ht.shape[-1]/2))
    PAM_GT_pad = np.pad(PAM_GT_pad,((0, 0), (0, 0), (pad_nz,pad_nz))) 

    z_start, z_end = 0, PAM_GT_pad.shape[2]      # similarly, extract the ROI in z dimension
    z_num_ROI = PAM_GT_pad.shape[2]
    PAM_GT_pad = PAM_GT_pad[: , :, z_start:z_end]

    num_focal_left =  z_num_ROI // 2 - z_start   # PAM_GT focal plane
    z_num_ROI = z_end-z_start

    PAM_GT_pad = torch.from_numpy(PAM_GT_pad).float().cuda()    
    # PAM_GT, pad_right, pad_bottom = pad2modulo(einops.rearrange(PAM_GT, 'nx ny nz->nz nx ny' ), 8) 
    # PAM_GT = einops.rearrange(PAM_GT, 'nz nx ny->nx ny nz')
    
    # PAM_2D_ROI = np.max(np.abs(cuda2numpy(PAM_GT)), axis = 2)
    # plt.figure()
    # plt.imshow(PAM_2D_ROI[2:-2,2:-2], cmap = mpl.colormaps['hot'])
    # plt.title('PAM ROI')
    # plt.show()   
    
    # %% 1: model setup    
    psf_cnt = psf_stack.shape[2]//2-1 + shift_opt
    # psf_cnt = psf_stack.shape[2]//2 + shift_opt
    idx = np.arange(psf_cnt-num_focal_left, psf_cnt-num_focal_left + z_num_ROI)
    psf_stack_align = psf_stack[:,:, np.mod( idx,  psf_stack.shape[2] ) ]
    # tifffile.imwrite('./psf_stack_align.tif',psf_stack_align.cpu().numpy().transpose(2,0,1))
    # tifffile.imwrite('./PAM_GT_pad'+opts['psf_name']+'.tif',PAM_GT_pad.cpu().numpy().transpose(2,0,1))

    #%% generating the measurement
    measurement = openPAM_Forward_FFT(PAM_GT_pad, psf_stack_align, ht)
    if 'pad_nz' in locals():
        measurement = measurement[pad_nx:-pad_nx, pad_ny:-pad_ny, pad_nz:-pad_nz].cpu().numpy()
    else:
        measurement = measurement[pad_nx:-pad_nx, pad_ny:-pad_ny, :].cpu().numpy()   
    # norm
    measurement = measurement / np.max(np.abs(measurement.flatten()))
    measurement = einops.rearrange(measurement, 'nx ny nz->nz nx ny')
    # # 3: Add noise
    noisy_measurement = add_noise_flexible(measurement, gaussian_std= 10/255)
    # Save measurement data
    tifffile.imwrite( save_filename + '_0.2NA.tif', measurement)
    tifffile.imwrite( save_filename + '_noisy_0.2NA.tif', noisy_measurement)
    print(f"Successfully saved {save_filename + '.tif'}")

if __name__ == '__main__':
    dataset_dir = os.getcwd() + '/ExpData_DH/'           # dataset dir that contains all the experimental data
    GT_dir = dataset_dir + 'simulation/GT/'    # folder of the data acquired in an imaging session
    save_dir = dataset_dir + 'simulation/measurement/'          # specify where to store the results
    
    # Configuration data for the imaging session
    config = {}                       # imaging session configuration
    config['n_medium'] = 1.5         # refractive index of the medium where signals are generated for simulation  
    # config['n_medium'] = 1.37         # refractive index of the medium where signals are generated
    config['sos'] = 1.5               # in um/ns, speed of sound in medium
    config['dz_air'] = 2.0            # in um: sampling interval in z dimesion (measured in air) of the 3D PSF
    config['dt'] = 2.0                # ToF sampling rate in ns == one time bin in tdc/tcspc of spad
    config['dxy'] = None              # psf_stack lateral sampling interval (default 3 um), specify value if different
    config['transform'] = 'none'      # 'lr', 'ud', 'inv', transforms the psf_stack: default is 'none'
    config['nonlinear_pow'] = 2.0     # camera gamma correction for PSF measurement
    
    # save dir
    f_n = 1 # single sample number, you can change it to a loop for multiple samples
    save_dir += str(f_n)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #%%
    # 1: scaling the PSF z dimension first
    psf_name_list = ["dh","gauss"]
    psf_stack_list, ht = get_dh_gauss_PSF_ht(dataset_dir, config, psf_name_list)
        
    # shift_array = np.int32( np.load(imagingsession_dir + 'shift_opt.npy') ) # shift_opt.npy is generated by autoAlign.py
    shift_array = [0]
    # N_file = len(shift_array)  # you can change it to a single file (large objects are stored in multiple files)
    st_time = time.perf_counter()
    
    # 3: perform neural deconvolution (wrapped in PAM_recon function)
    # Options for running the neural deconvolution algorithm
    opts = {}
    for i, psf_stack in enumerate(psf_stack_list):
        opts['psf_name'] = '_gauss' if i else '_dh'

        PAM_pupil = tifffile.imread( GT_dir + 'openPAM_GT_' + str(f_n) + '.tif')
        PAM_pupil = PAM_pupil / np.max(np.abs(PAM_pupil.flatten()))
        opts['save_filename'] = save_dir + f'/openPAM_measurement_' + str(f_n) + opts['psf_name']
        generate_measurement(PAM_pupil, psf_stack, ht, shift_array[0], opts) # 

    stop_time = time.perf_counter()
    print('Dataset reconstruction time:', stop_time - st_time)