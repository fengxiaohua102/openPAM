# -*- coding: utf-8 -*-
"""
Neural deconvolution of openPAM
This script performs neural deconvolution of openPAM based on PnP-FISTA and DRP framework, which uses neural denoisers and restoration models for regularization
Note that the psf_stack is positive, while ht and PAM signals are zero-centered
Please read the comments of the code and adjust the settings accordingly to suit your own needs
"""
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import tifffile
import time
import matplotlib as mpl
from openPAM_Operators import *
from Solver_FISTA_PnP import *
from Solver_DeblurDRP_refine import Solver_Deblur_DRP_refine
from utils_module import pad2modulo, norm1, numpy2cuda, cuda2numpy, postproc
from autoAlign import get_PSF_ht
import einops
from datetime import datetime

torch.cuda.set_device(0)

def save_list_to_file(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def PAM_recon(openPAM_sig, psf_stack, ht, shift_opt, save_filename, opt):
    # Input
    # openPAM_sig: the openPAM raw signal, should be in the range [-1, 1]
    # save_filename: path to save the results
    # opt: dictionary containing all algorithm parameters including 'Solver' and 'denoiser'
    
    # Now perform ROI extraction to reduce memory burden.
    # extract the region for faster deconv results
    PAM_ROI = openPAM_sig
    # PAM_ROI = openPAM_sig[:, 125:-125, 125:-125]
        
    # Extract the corresponding optical psfs for the PAM image
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
    
    # 1: model setup    
    psf_cnt = psf_stack.shape[2]//2-1 + shift_opt
    idx = np.arange(psf_cnt-num_focal_left, psf_cnt-num_focal_left + z_num_ROI)
    psf_stack_align = psf_stack[:,:, np.mod( idx,  psf_stack.shape[2] ) ]
    A = lambda x: openPAM_Forward_FFT(x, psf_stack_align, ht)
    AT = lambda x: openPAM_Adjoint_FFT(x, psf_stack_align, ht)
    
    # 2. Configure algorithm parameters based on Solver type
    # Common parameters
    opt['ht'] = ht                   # transducer impulse response (for DRP kernel usage)
    opt['vis'] = 1                   # verbose output
    opt['POScond'] = False           # positiveness constraint on the solution
    opt['tol'] = 1e-9                # convergence tolerance
    
    # Solver-specific parameters
    if opt['Solver'] == 'Solver_FISTA_PnP':

        max_egival = power_iter(A, AT, PAM_ROI.shape)
        # max_egival = 0.8
        print('max_egival is:', max_egival)
        
        # FISTA-PnP specific parameters
        # default, eye
        opt['maxiter'] = 60              # max iterations
        opt['lambda'] = 5e-3             # regularization parameter (5e-3 for drunet3D)
        opt['monotone'] = True           # monotone FISTA
        opt['step'] = 1.0*max_egival     # step size based on max eigenvalue
        
    elif opt['Solver'] == 'Solver_Deblur_DRP_refine':
        # DRP-refine specific parameters
        # # ear
        # opt['maxiter'] = 60              # max iterations
        # opt['step'] = 0.3                # step size (γ in paper)
        # opt['reg_lam'] = 0.6             # regularization lambda (τ in paper)
        # opt['sigma_n'] = 10/255          # noise level for denoising
        
        # fiber
        opt['maxiter'] = 60              # max iterations
        opt['step'] = 0.4               # step size (γ in paper)
        opt['reg_lam'] = 0.6             # regularization lambda (τ in paper)
        opt['sigma_n'] = 25/255          # noise level for denoising

        opt['save_optimal'] = False
        # Adjust deblur restoration orientation C H W
        # opt['dataform'] = 'nz nx ny'   # xy restoration 
        opt['dataform'] = 'nx ny nz'     # yz restoration 
        # opt['dataform'] = 'ny nz nx'   # xz restoration

        # Define kernel parameters for motion blur
        # kernel_paras: list of kernel configurations for different stages
        # First half iterations use kernel_paras[0], second half use kernel_paras[1]
        # Each kernel config is a dict with keys: r_angle, thick, rescale, bk_len
        opt['kernel_paras'] = [
            {'bk_len': 3, 'r_angle': 90, 'thick': 1, 'rescale': 1.0},
            {'bk_len': 1, 'r_angle': 90, 'thick': 1, 'rescale': 1.0}
        ]
    
    # convert odd num of z layers into even for restormer 
    if opt['Solver'] == 'Solver_Deblur_DRP_refine' or opt['denoiser'] == 'restormer':
        if PAM_ROI.shape[-1] % 2 != 0 and psf_stack_align.shape[-1] % 2 != 0:
            psf_stack_align = psf_stack_align[:,:,:-1]
            PAM_ROI = PAM_ROI[:,:,:-1]

    start_t = time.time()

    # Run reconstruction based on selected solver
    if opt['Solver'] == 'Solver_FISTA_PnP':
        recon_PAM_HD, convergence = Solver_PnP_FISTA(A, AT, PAM_ROI, PAM_ROI, opt)
        recon_PAM_HD_list = [recon_PAM_HD]  # Wrap in list for consistent processing
    elif opt['Solver'] == 'Solver_Deblur_DRP_refine':
        recon_PAM_HD_list, convergence = Solver_Deblur_DRP_refine(A, AT, PAM_ROI, opt)
    else:
        raise ValueError(f"Unknown solver: {opt['Solver']}")

    stop_t = time.time()
    print('\nGPU reconstruction time is:', stop_t - start_t)

    # release cuda memory
    torch.cuda.empty_cache()
    
    # Postprocessing the reconstruction
    for i, recon_PAM_HD in enumerate(recon_PAM_HD_list):    
        recon_PAM_HD = recon_PAM_HD[pad_nx:-pad_nx-pad_bottom, pad_ny:-pad_ny-pad_right, :]
        recon_PAM_HD = einops.rearrange(recon_PAM_HD, 'nx ny nz->nz nx ny')
        recon_PAM_HD = cuda2numpy(recon_PAM_HD)
        
        # Eliminating reflections that show inverse polarity w.r.t. the calibrated
        # transducer impulse response.
        recon_PAM_HD_n = recon_PAM_HD.copy()
        recon_PAM_HD_n[recon_PAM_HD_n>0] = 0.0      # get the negative part
        recon_PAM_HD_n = norm1(np.abs(recon_PAM_HD_n))

        # Post-processing: thresholding and gamma correction if needed, set gamma = 0.5 for better visualization in a few DRP reconstruction cases
        recon_PAM_HD_n = postproc(recon_PAM_HD_n, threshold=opt.get('threshold', 0.02), gamma=opt.get('gamma', 1.0))

        now = datetime.now().replace(second=0, microsecond=0)
        datetime_str = now.strftime('%Y%m%d %H_%M')
        
        # Save files based on solver type and result index
        if opt['Solver'] == 'Solver_FISTA_PnP':
            # For FISTA, only one result (default nz nx ny format)
            tifffile.imwrite(save_filename +'_'+datetime_str +'_iter'+str(opt['maxiter'])+ '_norm_neg.tif', recon_PAM_HD_n, dtype=np.float32)
        elif opt['Solver'] == 'Solver_Deblur_DRP_refine':
            # For DRP, final result first, then optional optimal result
            orient_name = {'nz nx ny':'xy','nx ny nz':'yz','ny nz nx':'zx'}  # Use orient_name from DRP params
            if i == 0:
                tifffile.imwrite(save_filename +'_'+'{:s}'.format(orient_name[opt['dataform']])+'_'+datetime_str +'_iter'+str(opt['maxiter'])+ '_norm_neg_final.tif', recon_PAM_HD_n, dtype=np.float32)
            elif i == 1:
                tifffile.imwrite(save_filename +'_'+'{:s}'.format(orient_name[opt['dataform']])+'_'+datetime_str +'_iter'+str(opt['maxiter'])+ '_norm_neg_optimal_obj.tif', recon_PAM_HD_n, dtype=np.float32)
        recon_PAM_HD_list[i] = recon_PAM_HD_n
    
    # Save convergence results
    save_list_to_file(convergence, save_filename+'_obj_val.txt')

    # # release cuda memory
    torch.cuda.empty_cache()
    
    # PAM_2D_deconv = np.sum(recon_PAM_HD_n, axis=0) 
    recon_PAM_HD_list_xy = [np.sum(x, axis=0) for x in recon_PAM_HD_list] # to compare yz
    recon_PAM_HD_list_yz = [np.sum(x, axis=1) for x in recon_PAM_HD_list] # to compare yz

    # visualization
    if len(recon_PAM_HD_list) == 1:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(recon_PAM_HD_list_xy[0][2:-2, 2:-2], cmap=mpl.colormaps['hot'])
        plt.title('Deconv_Final 2D XY')
        
        plt.subplot(1, 2, 2)
        plt.imshow(recon_PAM_HD_list_yz[0][150:-150, 1:-1], cmap=mpl.colormaps['hot'])
        plt.title('Deconv_Final 2D YZ')
        
        plt.tight_layout()
        plt.show()
    elif len(recon_PAM_HD_list) == 2:

        plt.figure(figsize=(10,  10))

        plt.subplot(2,  2, 1)  
        plt.imshow(recon_PAM_HD_list_xy[0][2:-2,  2:-2], cmap=mpl.colormaps['hot']) 
        plt.title('Deconv_Final 2D XY')
        
        plt.subplot(2,  2, 2) 
        plt.imshow(recon_PAM_HD_list_xy[1][2:-2,  2:-2], cmap=mpl.colormaps['hot'])  
        plt.title('Deconv_Optimal_Obj (Min) 2D XY')
        
        plt.subplot(2,  2, 3)  
        plt.imshow(recon_PAM_HD_list_yz[0][150:-150,  1:-1], cmap=mpl.colormaps['hot']) 
        plt.title('Deconv_Final 2D YZ')
        
        plt.subplot(2,  2, 4)  
        plt.imshow(recon_PAM_HD_list_yz[1][150:-150,  1:-1], cmap=mpl.colormaps['hot'])   
        plt.title('Deconv_Optimal_Obj (Min) 2D YZ')

        plt.tight_layout() 
        plt.show()

if __name__ == '__main__':
    # dataset_dir = os.getcwd() + r"/ExpData_DH/rawdata_ear"            # dataset dir that contains all the experimental data
    dataset_dir = os.getcwd() + r"/ExpData_DH/rawdata_fiber"            # dataset dir that contains all the experimental data
    imagingsession_dir = dataset_dir + '/jitter_corrected_results/'    # folder of the data acquired in an imaging session
    save_dir = dataset_dir + '/recon_results/'          # specify where to store the results
    ht_psf_dir = os.getcwd() + r"/ExpData_DH/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Configuration data for the imaging session
    config = {}                       # imaging session configuration
    config['n_medium'] = 1.37         # refractive index of the medium where signals are generated
    config['sos'] = 1.5               # in um/ns, speed of sound in medium
    config['dz_air'] = 2.0            # in um: sampling interval in z dimesion (measured in air) of the 3D PSF
    config['dt'] = 2.0                # ToF sampling rate in ns
    config['dxy'] = None              # psf_stack lateral sampling interval (default 3 um), specify value if different
    config['transform'] = 'none'      # 'lr', 'ud', 'inv', transforms the psf_stack: default is 'none'
    config['nonlinear_pow'] = 2.0     # camera gamma correction for PSF measurement

    # Algorithm configuration
    # Choose reconstruction algorithm and denoiser
    opt = {}
    # Solver options: 'Solver_FISTA_PnP' or 'Solver_Deblur_DRP_refine'
    opt['Solver'] = 'Solver_Deblur_DRP_refine'
    # Denoiser options: 'drunet', 'ffdnet', 'restormer', 'proxl1', 'proxtv' and 'none' for no denoising in DRP
    opt['denoiser'] = 'drunet'
    
    # 1: scaling the PSF z dimension first
    psf_stack, ht = get_PSF_ht(ht_psf_dir, config)
    
    # 2: psf alignment, locating NFP, we have done it for the same dataset in script: autoAlign.py
    # Get all sample subdirectories
    subdirs = [d for d in os.listdir(imagingsession_dir) if os.path.isdir(os.path.join(imagingsession_dir, d)) and d.isdigit()]
    subdirs.sort(key=lambda x: int(x)) # Sort numerically
    
    st_time = time.perf_counter()
    # 3: perform neural deconvolution (wrapped in PAM_recon function)
    # Options for running the neural deconvolution algorithm
    for f_n_str in subdirs:
        f_n = int(f_n_str)
        sample_subdir = os.path.join(imagingsession_dir, f_n_str)
        
        # Load the shift value for this sample
        shift_file = os.path.join(sample_subdir, 'shift_opt.npy')
        if not os.path.exists(shift_file):
            print(f"Shift file not found for sample {f_n}: {shift_file}")
            continue
        shift_opt = int(np.load(shift_file))

        # Load the PAM signal
        pam_sig_file = os.path.join(sample_subdir, 'openPAM_sig' + str(f_n) + '.tif')
        if not os.path.exists(pam_sig_file):
            print(f"PAM signal file not found for sample {f_n}: {pam_sig_file}")
            continue
            
        PAM_pupil = tifffile.imread(pam_sig_file)
        PAM_pupil = PAM_pupil / np.max(np.abs(PAM_pupil.flatten()))
        
        save_filename = save_dir + '/openPAM_deconv_'+ opt['Solver'] + "_" + opt['denoiser'] + "_" + str(f_n)
        print(f"Processing sample {f_n} with shift {shift_opt}")
        PAM_recon(PAM_pupil, psf_stack, ht, shift_opt, save_filename, opt) 

    stop_time = time.perf_counter()
    print('Dataset reconstruction time:', stop_time - st_time)