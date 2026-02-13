# -*- coding: utf-8 -*-
"""
Neural deconvolution of openPAM
This script performs openPAM neural deconvolution on comparison between Gaussian and DH PSF simulated openPAM measurements (generated from generate_sim_gauss_dh_meas.py), 
using FISTA-PnP as default algorithm.
"""
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import tifffile
import time
import matplotlib as mpl
from openPAM_Operators import *
from Solver_FISTA_PnP import Solver_PnP_FISTA
from Solver_DeblurDRP_refine import Solver_Deblur_DRP_refine, power_iter
from utils_module import pad2modulo, norm1, norm01, numpy2cuda, cuda2numpy, postproc
from autoAlign import PSF_scale_z
import einops
from datetime import datetime
import torch.nn.functional as F

torch.cuda.set_device(0)

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
    array_out = F.interpolate(array_torch, (nxy_resize, nz_resize), mode='bicubic') #interpolate default input shape is (N, C, H, W) 4D tensor, here using 4D tensor interpolation instead of 5D, for speed? 
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
    # 1. Ensure input is a NumPy array and get original length
    if not isinstance(array_in, np.ndarray) or array_in.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array.")
    
    nt_original = array_in.shape[0]

    array_torch = torch.from_numpy(array_in).float()

    array_torch = array_torch.view(1, 1, 1, nt_original)
    
    nt_resize = int(round(nt_original * scale_factor))
    # For 1D data, 'bicubic' and 'bilinear' are effectively equivalent to cubic/linear interpolation on the last dimension
    array_out_torch = F.interpolate(array_torch, size=(1, nt_resize), mode='bicubic', align_corners=False)

    array_out = array_out_torch.squeeze().numpy()
    
    return array_out

def save_list_to_file(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

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
        # psf_stack = np_imresize1D(psf_stack, 20)

        # if psf_name == 'gauss':
        #     psf_stack = np_imresize(psf_stack,30/36)
        
        # visualize all the psfs in a montage
        # psf_montage = skimage.util.montage( np.transpose(psf_stack, (2,0,1)) )
        # plt.figure()
        # plt.imshow(psf_montage)
        
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

    # # xy_imresize in psf stacks alignment
    # psf_stack_list[1] = torch_imresize(psf_stack_list[1],30/36)
    
    #******************************
    # the transducer psf (withut evenlop extraction)
    ht = tifffile.imread( dataset_dir + 'ht.tif')
    # upsampling
    # ht = resize_1d_pytorch(ht,2.0)
    ht = np.reshape(ht, (1, -1))
    ht = torch.from_numpy(ht.copy()).float()
    #********* normalized the ht
    ht = ht - torch.mean(ht)
    ht = ht/torch.sum(ht**2) 

    return psf_stack_list, ht

def PAM_recon(openPAM_sig, psf_stack, ht, shift_opt, save_filename, opt):
    # Input
    # openPAM_sig: the openPAM raw signal, should be in the range [-1, 1]
    # save_filename: path to save the results
    # opt: dictionary containing all algorithm parameters
    
    #  Now perform ROI extraction to reduce memory burden.
    # extract the region for faster deconv results
    PAM_ROI = openPAM_sig #[:, 0:1000, 0:1000], we used all the meausrement data by default
    PAM_ROI = openPAM_sig[:, 125:-125, 125:-125]
    
    #  Extract the corresponding optical psfs for the PAM image
    PAM_ROI = einops.rearrange(PAM_ROI, 'nz nx ny->nx ny nz' )      # change to the shape of Nx, Ny, Nt/Nz

    # pad the input data
    pad_nx, pad_ny = psf_stack.shape[0], psf_stack.shape[1]
    PAM_ROI = np.pad(PAM_ROI, ((pad_nx, pad_nx), (pad_ny, pad_ny), (0,0)) )

    # pad nz if there is signal in the edge of the z dimension
    # pad_nz = ht.shape[-1] + 1
    pad_nz = int(np.ceil(ht.shape[-1]/2))
    PAM_ROI = np.pad(PAM_ROI,((0, 0), (0, 0), (pad_nz,pad_nz))) 

    z_start, z_end = 0, PAM_ROI.shape[2]      # similarly, extract the ROI in z dimension
    z_num_ROI = PAM_ROI.shape[2]
    PAM_ROI = PAM_ROI[: , :, z_start:z_end]

    num_focal_left =  z_num_ROI // 2 - z_start   # PAM_ROI focal plane
    z_num_ROI = z_end-z_start
    
    PAM_ROI = torch.from_numpy(PAM_ROI).float().cuda()    
    PAM_ROI, pad_right, pad_bottom = pad2modulo(einops.rearrange(PAM_ROI, 'nx ny nz->nz nx ny' ), 8) 
    PAM_ROI = einops.rearrange(PAM_ROI, 'nz nx ny->nx ny nz')
    
    #  1: model setup    
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
    opt['monotone'] = True
    # opt['step'] = 1.0*max_egival     # step size

    # Solver-specific parameters
    if opt['Solver'] == 'Solver_FISTA_PnP':
        # FISTA-PnP specific parameters
        max_egival = power_iter(A, AT, PAM_ROI.shape)
        # max_egival = 0.8
        print('max_egival is:', max_egival)

        opt['maxiter'] = 60 
        # Previous code had opt['maxiter'] = 60 inside PAM_recon_drp (which was used for PnP?)
        opt['step'] = 1.0*max_egival     # step size
        opt['lambda'] = 5e-3             # regularization parameter
        
    elif opt['Solver'] == 'Solver_Deblur_DRP_refine':

        # DRP-refine specific parameters
        opt['maxiter'] = 60              # max iterations
        opt['step'] = 0.1                # step size (γ in paper)
        opt['reg_lam'] = 1.0             # regularization lambda (τ in paper)
        opt['sigma_n'] = 20/255          # noise level for denoising
        opt['save_optimal'] = False      # save optimal obj result
        # opt['dataform'] = 'nz nx ny'   # xy restoration 
        opt['dataform'] = 'nx ny nz'     # yz restoration 
        # opt['dataform'] = 'ny nz nx'   # xz restoration
        
        # Define kernel parameters for motion blur
        opt['kernel_paras'] = [
            {'bk_len': 3, 'r_angle': 90, 'thick': 1, 'rescale': 1.0},
            {'bk_len': 1, 'r_angle': 90, 'thick': 1, 'rescale': 1.0}
        ]

    start_t = time.time()
    
    # convert odd num of z layers into even for restormer 
    if opt['Solver'] == 'Solver_Deblur_DRP_refine' or opt['denoiser'] == 'restormer':
        if PAM_ROI.shape[-1] % 2 != 0 and psf_stack_align.shape[-1] % 2 != 0:
            psf_stack_align = psf_stack_align[:,:,:-1]
            PAM_ROI = PAM_ROI[:,:,:-1]
    
    # Run reconstruction based on selected solver
    if opt['Solver'] == 'Solver_FISTA_PnP':
        recon_PAM_HD, convergence = Solver_PnP_FISTA(A, AT, PAM_ROI, PAM_ROI, opt)
        recon_PAM_HD_list = [recon_PAM_HD]
    elif opt['Solver'] == 'Solver_Deblur_DRP_refine':
        recon_PAM_HD_list, convergence = Solver_Deblur_DRP_refine(A, AT, PAM_ROI, opt)
    else:
        raise ValueError(f"Unknown solver: {opt['Solver']}")

    stop_t = time.time()
    print('\nGPU reconstruction time is:', stop_t - start_t)
    
    #  Postproessing the reconstruction    
    for i, recon_PAM_HD in enumerate(recon_PAM_HD_list):
        if 'pad_nz' in locals():
            recon_PAM_HD = recon_PAM_HD[pad_nx:-pad_nx-pad_bottom, pad_ny:-pad_ny-pad_right, pad_nz:-pad_nz]
        else:
            recon_PAM_HD = recon_PAM_HD[pad_nx:-pad_nx-pad_bottom, pad_ny:-pad_ny-pad_right, :]

        recon_PAM_HD = einops.rearrange(recon_PAM_HD, 'nx ny nz->nz nx ny')
        recon_PAM_HD = cuda2numpy(recon_PAM_HD)
        
        # Eliminating reflections 
        recon_PAM_HD_n = recon_PAM_HD.copy()
        
        # Original code kept Positive part (recon < 0 => 0.0)
        recon_PAM_HD_n[recon_PAM_HD_n<0] = 0.0      # the positive part for no reflection
        
        # NOTE: Using norm1 (from previous script)
        recon_PAM_HD_n = norm1(np.abs(recon_PAM_HD_n))

        # Post-processing: thresholding and gamma correction if needed, set gamma = 0.5 for better visualization in a few DRP reconstruction cases
        recon_PAM_HD_n = postproc(recon_PAM_HD_n, threshold=opt.get('threshold', 0.02), gamma=opt.get('gamma', 1.0))

        now = datetime.now().replace(second=0, microsecond=0)
        datetime_str = now.strftime('%Y%m%d %H_%M')

        # Construct final filename based on solver
        # Original logic: save_filename += '_openPAM_DRP_deconv_' + denoiser + psf_name
        # New logic: save_filename passed in, append suffix
        
        # Note: 'openPAM_deconv' is already in save_filename passed from main
        
        if opt['Solver'] == 'Solver_FISTA_PnP':
             tifffile.imwrite( save_filename + '_'+ datetime_str + \
                '_iter_' + str(opt['maxiter']) + opt['is_noisy'] + '_norm_pos_0.2NA.tif', recon_PAM_HD_n, dtype=np.float32)
        elif opt['Solver'] == 'Solver_Deblur_DRP_refine':
            orient_name = {'nz nx ny':'xy','nx ny nz':'yz','ny nz nx':'zx'}  # Use orient_name from DRP params
            if i == 0:
                 tifffile.imwrite( save_filename + '_'+'{:s}_'.format(orient_name[opt['dataform']]) + datetime_str + \
                    '_iter_' + str(opt['maxiter']) + opt['is_noisy'] + '_norm_pos_0.2NA_final.tif', recon_PAM_HD_n, dtype=np.float32)
            elif i == 1:
                tifffile.imwrite( save_filename + '_'+'{:s}_'.format(orient_name[opt['dataform']]) + datetime_str + \
                    '_iter_' + str(opt['maxiter']) + opt['is_noisy'] + '_norm_pos_0.2NA_optimal_obj.tif', recon_PAM_HD_n, dtype=np.float32)
        
        recon_PAM_HD_list[i] = recon_PAM_HD_n

    save_list_to_file(convergence, save_filename+'_obj_val.txt') # obj_val after regulation step
    
    # release cuda memory
    torch.cuda.empty_cache()


if __name__ == '__main__':
    dataset_dir = os.getcwd() + r"/ExpData_DH/rawdata_particles"            # dataset dir that contains all the experimental data
    imagingsession_dir = dataset_dir + '/simulation_results/'    # folder of the data acquired in an imaging session
    save_dir = dataset_dir + '/recon_results/'          # specify where to store the results
    ht_psf_dir = os.getcwd() + r"/ExpData_DH/"

    # Configuration data for the imaging session
    config = {}                       # imaging session configuration
    # config['n_medium'] = 1.37         # refractive index of the medium where signals are generated
    config['n_medium'] = 1.5         # refractive index of the medium where signals are generated for simulation   
    config['sos'] = 1.5               # in um/ns, speed of sound in medium
    config['dz_air'] = 2.0            # in um: sampling interval in z dimesion (measured in air) of the 3D PSF
    config['dt'] = 2.0               # ToF sampling rate in ns == one time bin in tdc/tcspc of spad
    config['dxy'] = None              # psf_stack lateral sampling interval (default 3 um), specify value if different
    config['transform'] = 'none'      # 'lr', 'ud', 'inv', transforms the psf_stack: default is 'none'
    config['nonlinear_pow'] = 2.0     # camera gamma correction for PSF measurement
    
    
    # 1: scaling the PSF z dimension first
    psf_name_list = ["dh", "gauss"] # specify the psf names to be used for deconvolution, should match the saved psf file names
    psf_stack_list, ht = get_dh_gauss_PSF_ht(ht_psf_dir, config, psf_name_list)
    # ht rescale
    
    # 2: psf alignment, locating NFP, we have done it for the same dataset in script: autoAlign.py
    # Get all sample subdirectories
    subdirs = [d for d in os.listdir(imagingsession_dir) if os.path.isdir(os.path.join(imagingsession_dir, d)) and d.isdigit()]
    subdirs.sort(key=lambda x: int(x)) # Sort numerically

    st_time = time.perf_counter()
    
    # 3: perform neural deconvolution (wrapped in PAM_recon function)
    
    # Define Algorithm Options (Unified)
    opt = {}
    # Options: 'Solver_FISTA_PnP', 'Solver_Deblur_DRP_refine'
    opt['Solver'] = 'Solver_FISTA_PnP' 
    # Denoiser options: 'drunet', 'ffdnet', 'restormer', 'proxl1', 'proxtv' and 'none' for no denoising in DRP
    opt['denoiser'] = 'drunet'

    for f_n_str in subdirs:
        f_n = int(f_n_str)
        sample_subdir = os.path.join(imagingsession_dir, f_n_str)
        
        # Setup save directory for this sample
        sample_save_dir = os.path.join(save_dir, str(f_n))
        if not os.path.exists(sample_save_dir):
            os.makedirs(sample_save_dir)

        # Load the shift value for this sample
        shift_file = os.path.join(sample_subdir, 'shift_opt.npy')
        if os.path.exists(shift_file):
            shift_opt = int(np.load(shift_file))
        else:
            shift_opt = 0 # Default if not found
            
        print(f"Processing sample {f_n} with shift {shift_opt}")

        for is_noisy in [0]:
            opt['is_noisy'] = '_noisy' if is_noisy else ''
            opt['hilbert_devconv'] = ''
            opt['xy_devconv'] = ''
            
            for i, psf_stack in enumerate(psf_stack_list):
                if i == 0: # dh kernel 
                    opt['psf_name'] = '_dh'
                    # Construct input filename
                    pam_file_name = f'openPAM_measurement_{f_n}{opt["psf_name"]}{opt["is_noisy"]}_0.2NA.tif'
                    pam_file_path = os.path.join(sample_subdir, pam_file_name)
                    
                    if not os.path.exists(pam_file_path):
                         print(f"File not found: {pam_file_path}")
                         continue

                    PAM_pupil_openPAM = tifffile.imread(pam_file_path)
                    PAM_pupil_openPAM = PAM_pupil_openPAM / np.max(np.abs(PAM_pupil_openPAM.flatten()))
                    
                    # Construct Save Filename
                    save_filename = sample_save_dir + '/' + str(f_n) + '_openPAM_' + ('DRP' if 'DRP' in opt['Solver'] else 'PnP') + '_deconv_' + opt['denoiser'] + opt['psf_name']

                    PAM_recon(PAM_pupil_openPAM, psf_stack, ht, shift_opt, save_filename, opt) 
                
                elif i: # gaussain kernel
                    opt['psf_name'] = '_gauss'
                    # Construct input filename
                    pam_file_name = f'openPAM_measurement_{f_n}{opt["psf_name"]}{opt["is_noisy"]}_0.2NA.tif'
                    pam_file_path = os.path.join(sample_subdir, pam_file_name)
                    
                    if not os.path.exists(pam_file_path):
                         print(f"File not found: {pam_file_path}")
                         continue

                    PAM_pupil_openPAM = tifffile.imread(pam_file_path)
                    PAM_pupil_openPAM = PAM_pupil_openPAM / np.max(np.abs(PAM_pupil_openPAM.flatten()))
                    
                    # Construct Save Filename
                    save_filename = sample_save_dir + '/' + str(f_n) + '_openPAM_' + ('DRP' if 'DRP' in opt['Solver'] else 'PnP') + '_deconv_' + opt['denoiser'] + opt['psf_name']

                    PAM_recon(PAM_pupil_openPAM, psf_stack, ht, shift_opt, save_filename, opt) 

    stop_time = time.perf_counter()
    print('Dataset reconstruction time:', stop_time - st_time)
