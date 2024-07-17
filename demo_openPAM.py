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
from utils_module import pad2modulo, norm1, hilbert_transform, numpy2cuda, cuda2numpy
from einops import rearrange
from autoAlign import get_PSF_ht
import einops

def PAM_recon(openPAM_sig, psf_stack, ht, shift_opt, opts):
    # Input
    # openPAM_sig: the openPAM raw signal, should be in the range [-1, 1]     
    save_filename = opts['save_filename']
    denoiser = opts['denoiser']
    
    # %% Now perform ROI extraction to reduce memory burden.
    # extract the region for faster deconv results
    PAM_ROI = openPAM_sig #[:, 0:1000, 0:1000], we used all the meausrement data by default
    
    # extract the z ROI
    # PAM_proj2D_x = np.sum(np.abs(PAM_new_ROI), axis = 1)
    # PAM_proj2D_y = np.sum(np.abs(PAM_new_ROI), axis = 2)
    # # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(PAM_proj2D_x, cmap = mpl.colormaps['hot'])
    # plt.subplot(1,2,2)
    # plt.imshow(PAM_proj2D_y, cmap = mpl.colormaps['hot'])
    
    # %% Extract the corresponding optical psfs for the PAM image
    z_start, z_end = 0, 512      # similarly, extract the ROI in z dimension
    PAM_ROI = PAM_ROI[z_start:z_end, :, :]

    num_focal_left = 255 - z_start     # 255 of ToF corresponds to focal plane
    z_num_ROI = PAM_ROI.shape[0]
    
    PAM_ROI = einops.rearrange(PAM_ROI, 'nz nx ny->nx ny nz' )      # change to the shape of Nx, Ny, Nt/Nz
    
    # pad the input data
    pad_nx, pad_ny = psf_stack.shape[0], psf_stack.shape[1]
    PAM_ROI = np.pad(PAM_ROI, ((pad_nx, pad_nx), (pad_ny, pad_ny), (0,0)) )
    
    PAM_ROI = torch.from_numpy(PAM_ROI).float().cuda()    
    PAM_ROI, pad_right, pad_bottom = pad2modulo(einops.rearrange(PAM_ROI, 'nx ny nz->nz nx ny' ), 8) 
    PAM_ROI = einops.rearrange(PAM_ROI, 'nz nx ny->nx ny nz')
    
    # PAM_2D_ROI = np.max(np.abs(cuda2numpy(PAM_ROI)), axis = 2)
    # plt.figure()
    # plt.imshow(PAM_2D_ROI[2:-2,2:-2], cmap = mpl.colormaps['hot'])
    # plt.title('PAM ROI')
    # plt.show()   
    
    # %% 1: model setup    
    psf_cnt = psf_stack.shape[2]//2-1 + shift_opt
    idx = np.arange(psf_cnt-num_focal_left, psf_cnt-num_focal_left + z_num_ROI)
    psf_stack_align = psf_stack[:,:, np.mod( idx,  psf_stack.shape[2] ) ]
    A = lambda x: openPAM_Forward_FFT(x, psf_stack_align, ht)
    AT = lambda x: openPAM_Adjoint_FFT(x, psf_stack_align, ht)

    max_egival = 0.8 #  power_iter(A, AT, PAM_ROI.shape)
    print('max_egival is:', max_egival)
    
    # %% 2.FSITA recon
    opt = {}
    opt['tol'] = 1e-9
    opt['maxiter'] = 60              # param for max iteration
    opt['lambda'] = 5e-3             # param for regularizing param 5e-3 for drunet3D
    opt['vis'] = 1
    opt['denoiser'] = denoiser       # option of denoiser: proxtv:2e-2 ffdnet fastdvd
    opt['POScond'] = False           # positiveness contraint on the solution
    opt['monotone'] = True
    opt['step'] = 1.0*max_egival     # step size
    
    start_t = time.time()
    recon_PAM_HD, convergence = Solver_PnP_FISTA(A, AT, PAM_ROI, PAM_ROI, opt)
    stop_t = time.time()
    print('GPU reconstruction time is:', stop_t - start_t)
    
    #%% Postproessing the reconstruction    
    recon_PAM_HD = recon_PAM_HD[pad_nx:-pad_nx-pad_bottom, pad_ny:-pad_ny-pad_right, :]
    recon_PAM_HD = einops.rearrange(recon_PAM_HD, 'nx ny nz->nz nx ny')
    recon_PAM_HD = cuda2numpy(recon_PAM_HD)
    
    # Eliminating reflections that show inverse polarity w.r.t. the calibrated
    # transducer impulse response.
    recon_PAM_HD_n = recon_PAM_HD.copy()
    recon_PAM_HD_n[recon_PAM_HD_n>0] = 0.0      # the negtive part
    recon_PAM_HD_n = norm1(np.abs(recon_PAM_HD_n))
    tifffile.imwrite( save_filename + '_neg.tif', recon_PAM_HD_n)
    
    PAM_2D_deconv = np.sum(recon_PAM_HD_n, axis=0)
    
    #%
    plt.figure
    plt.imshow(PAM_2D_deconv[2:-2,2:-2], cmap = mpl.colormaps['hot'] )
    plt.title('Deconv 2D res')
    plt.show()
    # release cuda memory
    torch.cuda.empty_cache()

    # plt.figure
    # plt.plot(np.log(convergence))
    # plt.title('Deconv convergence')

if __name__ == '__main__':
    dataset_dir = os.getcwd() + '/ExpData_DH/'           # dataset dir that contains all the experimental data
    imagingsession_dir = dataset_dir + 'rawdata_eye/'    # folder of the data acquired in an imaging session
    save_dir = imagingsession_dir + '/results/'          # specify where to store the results
    
    # Configuration data for the imaging session
    config = {}                       # imaging session configuration
    config['n_medium'] = 1.37         # refractive index of the medium where signals are generated
    config['sos'] = 1.5               # in um/ns, speed of sound in medium
    config['dz_air'] = 2.0            # in um: sampling interval in z dimesion (measured in air) of the 3D PSF
    config['dt'] = 2.0                # ToF sampling rate in ns
    config['dxy'] = None              # psf_stack lateral sampling interval (default 3 um), specify value if different
    config['transform'] = 'none'      # 'lr', 'ud', 'inv', transforms the psf_stack: default is 'none'
    config['nonlinear_pow'] = 2.0     # camera gamma correction for PSF measurement
    
    #%%
    # 1: scaling the PSF z dimension first
    psf_stack, ht = get_PSF_ht(dataset_dir, config)
    
    # 2: psf alignment, locating NFP, we have done it for the same dataset in script: autoAlign.py
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    shift_array = np.int32( np.load(imagingsession_dir + 'shift_opt.npy') ) # shift_opt.npy is generated by autoAlign.py
    N_file = len(shift_array)  # you can change it to a single file (large objects are stored in multiple files)
    st_time = time.perf_counter()
    
    # 3: perform neural deconvolution (wrapped in PAM_recon function)
    # Options for running the neural deconvolution algorithm
    opts = {}
    opts['denoiser'] = 'drunet'
    for K in range(N_file):
        f_n = K+1
        PAM_pupil = tifffile.imread( imagingsession_dir + 'openPAM_sig' + str(f_n) + '.tif')
        PAM_pupil = PAM_pupil / np.max(np.abs(PAM_pupil.flatten()))
        opts['save_filename'] = save_dir + '/openPAM_deconv_'+ opts['denoiser'] + str(f_n) 
        PAM_recon(PAM_pupil, psf_stack, ht, shift_array[K], opts) # 

    stop_time = time.perf_counter()
    print('Dataset reconstruction time:', stop_time - st_time)