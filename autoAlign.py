# -*- coding: utf-8 -*-
"""
This script performs the oepnPAM psf stack alignment (that is align the measured psf_stack to its actual physical
detph inside media as detailed in the accompanied supplementary information of the manuscript "High resolution
volumetrtic imaging with optically encoded photoacoustic microscopy (openPAM)"
Note that the psf_stack is positive, while ht (transducer impulse response) and PAM signals are zero-centered
and should be in the range of [-1, 1]
"""
import torch
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from matplotlib import cm
import skimage 
import os
import scipy.io as sio
import tifffile
import time
import matplotlib as mpl
from openPAM_Operators import *
from Solver_FISTA_PnP import *
from utils_module import pad2modulo, norm01, numpy2cuda, cuda2numpy, torch_imresize1D, hilbert_transform, torch_imresize
import einops

torch.cuda.set_device(0)

def npxyt_hilbert(p_xyt):
    # a utility function that perform hilbert transform on numpy array of [nx ny nt]
    hlb_res = numpy2cuda(einops.rearrange(p_xyt, 'nx ny nt->nt nx ny'))
    hlb_res = hilbert_transform(hlb_res)
    return einops.rearrange(cuda2numpy(hlb_res), 'nt nx ny->nx ny nt' )

def PSF_scale_z(psf_stack_raw, conf):
# perform re-scaling along the z dimension
# air 2 um sampling becomes 2*n_medium um in tissue, PAM z sampling (tof: 2ns*1500)=3 um in tissue
# therefore, need to resample 2*n um in tissue to 3 um
# ********** See derivation in the supplementary file for details *****************************

# input: 
#       psf_stack_raw: [nx ny nz], preprocessed psf_stack from camera
#       conf: system configuration for the imaging session
    dz_air = conf['dz_air']         # in um
    c = conf['sos']                 # in um/ns
    n_medium = conf['n_medium']
    dt = conf['dt']                 # in ns
    dxy = conf['dxy']               # sampling in xy lateral dimension
    dz_tissue = dz_air*n_medium
    dz_tof = c*dt 
    r_scale_z = dz_tissue/dz_tof 
    psf_stack_rz = torch_imresize1D(psf_stack_raw, r_scale_z)

    # print('Shape and Range of psf_stack is:',psf_stack_rz.shape, np.max(psf_stack_rz), np.min(psf_stack_rz))
    cnt_slice = psf_stack_rz.shape[2]//2+1
    z_num = cnt_slice*2
    psf_stack = psf_stack_rz[:,:, cnt_slice - z_num//2: cnt_slice - z_num//2 + z_num]
    if dxy is not None:
        psf_stack = torch_imresize(psf_stack, 3.0/dxy)   # to interpolate to dxy um in lateral xy dimension
    return psf_stack
    
def get_PSF_ht(dataset_dir, conf):
    # note that tifffile read different from matlab: depth first here
    # read the experimental psf: psf stack and the transducer ht
    psf_stack = tifffile.imread( dataset_dir + 'psf_stack_z2um.tif')
    psf_stack = psf_stack.transpose(1,2,0) # for different view of saved psf_stack
    psf_stack = norm01(psf_stack)
    
    # visualize all the psfs in a montage
    # psf_montage = skimage.util.montage( np.transpose(psf_stack, (2,0,1)) )
    # plt.figure()
    # plt.imshow(psf_montage)
    
    # the transducer psf (withut evenlop extraction)
    ht = tifffile.imread( dataset_dir + 'ht.tif')
    ht = np.reshape(ht, (1, -1))
    
    psf_stack = PSF_scale_z(psf_stack, conf)
    
    # psf_stack and ht normliazation 
    psf_stack = torch.from_numpy( psf_stack ).float()
    psf_stack[psf_stack<0.0] = 0.0
    ht = torch.from_numpy(ht.copy()).float()
    #********* normalized the psfs 
    ht = ht - torch.mean(ht)
    ht = ht/torch.sum(ht**2)  
    psf_stack = psf_stack ** conf['nonlinear_pow']
    for K in range(psf_stack.shape[2]):
        psf_stack[:,:,K] = psf_stack[:,:,K] / torch.sum(psf_stack[:,:,K])  #[Nx, Ny, Nz]

    if(conf['transform'] =='lr'):
        psf_stack = torch.fliplr( psf_stack )  # FLIP the psf upsidedown: axis-0 for up, axis-1 for lr
    elif(conf['transform'] =='ud'):
        psf_stack = torch.flipud(psf_stack)  # FLIP the psf upsidedown: axis-0 for up, axis-1 for lr
    elif(conf['transform'] =='inv'):
        psf_stack = torch.fliplr( torch.flipud(psf_stack) )  # FLIP the psf upsidedown: axis-0 for up, axis-1 for lr
    #******************************
    
    return psf_stack, ht 


def autoAlign_FISTA(psf_stack, ht, PAM_ROI, configs ):
    '''
    Input:
        psf_stack: torch tensor
        ht: torch tensor
        PAM_ROI: the PAM measurement, torch tensor
        configs: configuration parameters for psf aligntment
        shift_opt: the optimized shift from coarse adjustment
    Output:
        shift: the identified shift for the psf_stack
    '''
    num_focal_left = configs['num_focal_left']
    shift_array = configs['shift_array']
    save_filename = configs['save_filename']
    reg_param = configs['reg_param']
    ROI = configs['ROI']
    
    # Solver options
    opt = {}
    opt['tol'] = 1e-6
    opt['maxiter'] = 30               # param for max iteration
    opt['lambda'] = reg_param         # param for regularizing param 2e-3 for drunet3D
    opt['vis'] = 1
    opt['denoiser'] = configs['denoiser']       # option of denoiser: ProxTV,ProxTV3D:2e-2 ffdnet fastdvd
    opt['POScond'] = False            # positiveness contraint on the solution
    opt['monotone'] = True
    opt['step'] = 1.0*0.8             # step size
    
    z_num_ROI = PAM_ROI.shape[2]
     
    # 2: Fine alignment
    start_t = time.time()     
    
    # extract a small ROI for fast computation ***********
    pad_nx, pad_ny = psf_stack.shape[0], psf_stack.shape[1]
    # comment the following two lines to use full xyt data
    if(ROI):
        nx, ny = PAM_ROI.shape[0], PAM_ROI.shape[1]
        PAM_ROI = (PAM_ROI[1*nx//16:15*nx//16, 1*ny//16:15*ny//16,:])  #   1*nx//4:3*nx//4, 1*ny//4:3*ny//4,:
    if(save_filename is not None):
        tifffile.imwrite(save_filename + '_org.tif', np.sum( npxyt_hilbert(PAM_ROI), axis=2) ) #  
    
    nx, ny = PAM_ROI.shape[0],  PAM_ROI.shape[1]
    PAM_ROI = np.pad(PAM_ROI, ((pad_nx, pad_nx), (pad_ny, pad_ny), (0,0)) )
    PAM_ROI = numpy2cuda(PAM_ROI)    
    PAM_ROI, pad_right, pad_bottom = pad2modulo(einops.rearrange(PAM_ROI, 'nx ny nz->nz nx ny' ), 8) 
    PAM_ROI = einops.rearrange(PAM_ROI, 'nz nx ny->nx ny nz')
    
    # ****************************************************
    N_step = len(shift_array)
    Phys_loss_fine = np.zeros((N_step,))
    if(save_filename is not None):
        PAM_2D_deconv = np.zeros((N_step, nx, ny), dtype=np.float32)
        # PAM_xz_deconv = np.zeros((N_step, nx, PAM_ROI.shape[2]), dtype=np.float32)
        
    for K, shift in enumerate( shift_array):    
        print('Solving K: ', K)
        psf_cnt = psf_stack.shape[2]//2-1 + shift
        idx = np.arange(psf_cnt-num_focal_left, psf_cnt-num_focal_left + z_num_ROI)
        psf_stack_shift = psf_stack[:,:, np.mod( idx,  psf_stack.shape[2] ) ]

        A = lambda x: openPAM_Forward_FFT(x, psf_stack_shift, ht)
        AT = lambda x: openPAM_Adjoint_FFT(x, psf_stack_shift, ht)

        recon_PAM_HD_cuda, convergence = Solver_PnP_FISTA(A, AT, PAM_ROI, PAM_ROI, opt)
        Phys_loss_fine[K] = convergence[-1]
        if(save_filename is not None): 
            recon_PAM_HD = cuda2numpy( recon_PAM_HD_cuda[pad_nx:-pad_nx-pad_bottom, pad_ny:-pad_ny-pad_right, :] )
            recon_PAM_HD[recon_PAM_HD>0] = 0.0      # the negtive part       
            recon_PAM_HD = (np.abs(recon_PAM_HD))
            PAM_2D_deconv[K, :, :] = np.sum(recon_PAM_HD, axis =2)
            # PAM_xz_deconv[K, :, :] = np.sum(recon_PAM_HD, axis =1)
    
    # ************************************************************************************
    # NOTE: a 2D projection results of the deconvolution process are stored to validate or manually
    # fine-tune the automatic alignment of the psf-stack with the actual depth.
    if(save_filename is not None):    
        tifffile.imwrite(save_filename + '.tif', PAM_2D_deconv ) #
        # tifffile.imwrite(save_filename + '_xz.tif', PAM_xz_deconv ) #
            
    stop_t = time.time()  
    print('GPU reconstruction time is:', stop_t - start_t)

    return np.argmin(Phys_loss_fine), Phys_loss_fine

def read_dataset(data_dir, f_n):
    # Note the openPAM_sig and PAMsig should be in the range [-1, 1]    
    PAM_pupil = tifffile.imread( data_dir + 'openPAM_sig' + str(f_n) + '.tif')  # [nz nx ny]

    # Now perform ROI extraction to reduce memory burden.
    # extract the region for fast deconv results
    z_start, z_end = 0, 512      # 240, 450
    PAM_ROI = PAM_pupil[z_start:z_end, :, :]

    num_focal_left = 255 - z_start     # 255 of ToF corresponds to focal plane
    
    PAM_ROI = einops.rearrange(PAM_ROI, 'nz nx ny -> nx ny nz' )      # change to the shape of Nx, Ny, Nt/Nz
    return PAM_ROI, num_focal_left


if __name__ == '__main__':
    dataset_dir = os.getcwd() + r"/ExpData_DH/rawdata_ear"           # dataset dir that contains all the experimental data
    imagingsession_dir = dataset_dir + '/jitter_corrected_results/'     # folder of the data acquired in an imaging session
    ht_psf_dir = os.getcwd() + r"/ExpData_DH/"

    N_samples = 2   # number of raw measurements to be processed
    
    # Configuration data for the imaging session
    config = {}                       # imaging session configuration
    config['n_medium'] = 1.37         # refractive index of the medium where signals are generated
    config['sos'] = 1.5               # in um/ns, speed of sound in medium
    config['dz_air'] = 2.0            # in um
    config['dt'] = 2.0                # ToF sampling rate in ns
    config['dxy'] = None              # psf_stack lateral sampling interval (default 3 um), specify value if different
    config['transform'] = 'none'      # 'lr', 'ud', 'inv', transforms the psf_stack
    config['nonlinear_pow'] = 2.2  
    # Options for the solvers
    configs_sol = {}
    configs_sol['reg_param'] = 1e-3       # regularization of the solver strength
    configs_sol['denoiser'] = 'proxl1'    # regularization type of the solver (FISTA)
    configs_sol['ROI'] = True             # whether use ROI to speed up calculation
       
    # 1: scaling the PSF z dimension first
    psf_stack, ht = get_PSF_ht(ht_psf_dir, config)
        
    # shift_opt_array = np.zeros((N_file,), dtype=np.int32)
    st_time = time.perf_counter()
    shift_array = np.int32( np.arange(-80, 180, step=4) )  # -90, 186
   
    for K in range(N_samples):
        f_n = K + 1
        sample_dir = imagingsession_dir + f'{str(f_n)}/'
        save_dir = sample_dir + 'shift_recon/'

        if not os.path.exists(sample_dir):
            print(f"Directory {sample_dir} does not exist. Skipping.")
            continue

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        PAM_ROI, num_focal_left = read_dataset(sample_dir, f_n)
        configs_sol['num_focal_left'] = num_focal_left
        save_fname = save_dir + 'Fista_coarse_' + config['transform'] + str(f_n) # _MIP_
        
        #%1: Coarse search of the NFP using FISTA
        shift_opt = shift_array[len(shift_array)//2]
        BD = shift_array[-1] - shift_opt
        step_val = np.int32(np.sqrt(BD))
        shift_array_coarse = np.int32( np.arange(shift_opt-BD, shift_opt+BD, step= step_val) )
        configs_sol['shift_array'] = shift_array_coarse
        configs_sol['save_filename']= save_fname
        K_idx, Phys_loss_fista = autoAlign_FISTA(psf_stack, ht, PAM_ROI, configs_sol)
        
        shift_opt_coarse = shift_array_coarse[np.int32(K_idx)]
                    
        # 2: Fine search of the NFP using FISTA
        save_fname = None # save_dir + 'Fista_fine' + str(K)            
        shift_array_fine = np.int32( np.arange(shift_opt_coarse-step_val, shift_opt_coarse+step_val+1, step= 1) )
        configs_sol['shift_array'] = shift_array_fine
        configs_sol['save_filename']= save_fname
        K_idx2, Phys_loss_fista2 = autoAlign_FISTA(psf_stack, ht, PAM_ROI, configs_sol)

        shift_opt_fine = shift_array_fine[np.int32(K_idx2)]
        print(f'Sample {f_n} - FISTA located coarse index and shift is:', np.int32(K_idx), shift_opt_coarse)
        print(f'Sample {f_n} - FISTA located fine index and shift is:', np.int32(K_idx2), shift_opt_fine)
        
        # Save individual shift_opt for this sample
        np.save(sample_dir + 'shift_opt.npy', shift_opt_fine)
        
        plt.figure()
        plt.plot(Phys_loss_fista) 
        plt.title(f'Coarse search via objective loss - Sample {f_n}')
        
        plt.figure()
        plt.plot(Phys_loss_fista2) 
        plt.title(f'Fine search via objective loss - Sample {f_n}')

    stop_time = time.perf_counter()
    print('Total processing time is:', stop_time - st_time)
    
