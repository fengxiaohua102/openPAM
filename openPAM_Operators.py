# -*- coding: utf-8 -*-
"""
Implement the openPAM operators using pytorch to take full advantage
of GPU and deep neural denoising networks.
"""

import torch
import numpy as np
from torch.autograd import gradcheck


def openPAM_Forward_FFT(PAM_HD, psf_stack, psf_transducer):
#  The forward model for openPAM
#  1 PAM_HD: the postive high resolution (z) 3D PAM image in the shape of Nx, Ny, Nt, a torch.tensor
#  2 psf_stack: the normalized point spread function or simply the kernel of the depth dependent psf, similar as PAM_HD
#  3 psf_transducer: the temporal kernel (envelop) of the transucer, a one-D torch.tensor: 1, Nt
#   ******** The forward model of the image formation is:   *******
#   PAM_3D = psf_transducer *t (PAM_HD(:,:,K) *xy psf_stack(:,:,K))
#  where *t is conv in time while *xy is conv in space

    psf_cuda = psf_stack.float().cuda()            # H, W, Nz
    nx, ny, nz = psf_cuda.shape

    PAM_HD = PAM_HD.float().cuda()                 # H, W, Nz    
    Nx, Ny, Nz = PAM_HD.shape
    
    psf_stack_cuda = torch.zeros_like(PAM_HD, dtype = torch.float32, device = 'cuda')    
    psf_stack_cuda[np.int32(np.round((Nx-nx)/2.0)): np.int32(np.round((Nx-nx)/2.0))+ nx, 
                   np.int32(np.round((Ny-ny)/2.0)): np.int32(np.round((Ny-ny)/2.0))+ ny, :]  = psf_cuda
    
    # depthwise convolution
    PAM_FFT = torch.fft.rfft2( torch.fft.ifftshift(PAM_HD, dim=(0,1)) , dim=(0,1) )   
    PSF_FFT = torch.fft.rfft2( torch.fft.ifftshift(psf_stack_cuda, dim=(0,1)), dim=(0,1) )  
    PAM_FFT = PAM_FFT * PSF_FFT                      # H, W, Nz
    PAM_FFT = torch.fft.fftshift( torch.real( torch.fft.irfft2( PAM_FFT, s =(Nx,Ny), dim=(0,1) )), dim=(0,1) )  # H, W, Nz
    
    # Process temporal convolution in parallel   
    idx_max = torch.argmax(psf_transducer).item()
    # print('idx_max is', idx_max)
    
    psf_trans = psf_transducer.float().view(1, -1).cuda() # 1, Nz
    psf_trans_cuda = torch.zeros((1, PAM_FFT.shape[2]), dtype = torch.float32, device = 'cuda')
    psf_trans_cuda[0, np.int32( np.round( PAM_FFT.shape[2]/2.0) ) - idx_max 
                   :  np.int32( np.round( PAM_FFT.shape[2]/2.0) ) - idx_max + psf_trans.shape[1] ] = psf_trans
    trans_FFT = torch.fft.rfft( torch.fft.ifftshift( psf_trans_cuda, dim = 1), dim = 1 )                                # 1, Nz

    PAM_FFT = torch.fft.rfft( torch.fft.ifftshift( PAM_FFT, dim = 2 ), dim = 2 )                 # H, W, Nz
    PAM_FFT = PAM_FFT * trans_FFT   # (H, W, Nz) .* (1, Nz)
    PAM_FFT = torch.fft.fftshift( torch.real( torch.fft.irfft(PAM_FFT, n = Nz, dim = 2 ) ), dim = 2 )     # H, W, Nz

    return PAM_FFT  # Nx,Ny,Nz

def openPAM_Adjoint_FFT(PAM_3D, psf_stack, psf_transducer):
#  The forward model for openPAM
#  1 PAM_3D: the measured 3D PAM image in the shape of Nx, Ny, Nt, a torch.tensor
#  2 psf_stack: the normalized point spread function or simply the kernel of the depth dependent psf similar as PAM_3D
#  3 psf_transducer: the temporal kernel (envelop) of the transucer, a one-D torch.tensor: 1, Nt
#   ******** The forward model of the image formation is:   *******
#   PAM_3D = psf_transducer *t (PAM_HD(:,:,K) *xy psf_stack(:,:,K))
#  where *t is conv in time while *xy is conv in space

    PAM_3D = PAM_3D.float().cuda()                 # H, W, Nz
    Nx, Ny, Nz = PAM_3D.shape
    
    # Process temporal convolution in parallel   
    idx_max = torch.argmax(psf_transducer).item()
    # print('idx_max is', idx_max)
    psf_trans = psf_transducer.float().view(1, -1).cuda() # 1, Nz
    psf_trans_cuda = torch.zeros((1, PAM_3D.shape[2]), dtype = torch.float32, device = 'cuda') # 1, Nz
    psf_trans_cuda[0, np.int32( np.round(PAM_3D.shape[2]/2.0) ) - idx_max 
                    : np.int32( np.round(PAM_3D.shape[2]/2.0) ) - idx_max + psf_trans.shape[1] ] = psf_trans
    trans_FFT = torch.fft.rfft( torch.fft.ifftshift( psf_trans_cuda, 1), dim = 1 )                                # 1, Nz
    PAM_FFT = torch.fft.rfft( torch.fft.ifftshift(PAM_3D , dim = 2), dim = 2 )                 # H, W, Nz
    PAM_FFT = PAM_FFT * torch.conj(trans_FFT)                  # H, W, Nz
    PAM_FFT = torch.fft.fftshift( torch.real( torch.fft.irfft(PAM_FFT, n = Nz,  dim = 2 ) ), dim=2 )   # H, W, Nz

    psf_cuda = psf_stack.float().cuda()            # H, W, Nz    
    nx, ny, nz = psf_cuda.shape    
    Nx, Ny, Nz = PAM_3D.shape
    psf_stack_cuda = torch.zeros_like(PAM_3D, dtype = torch.float32, device = 'cuda')
    psf_stack_cuda[np.int32(np.round((Nx-nx)/2.0)): np.int32(np.round((Nx-nx)/2.0))+ nx, 
                   np.int32(np.round((Ny-ny)/2.0)): np.int32(np.round((Ny-ny)/2.0))+ ny, :]  = psf_cuda
        
    # depthwise convolution
    PAM_FFT = torch.fft.rfft2( torch.fft.ifftshift(PAM_FFT, dim=(0,1)), dim =(0,1) )  
    PSF_FFT = torch.fft.rfft2( torch.fft.ifftshift(psf_stack_cuda, dim=(0,1)), dim= (0,1) )  
    PAM_FFT = PAM_FFT * torch.conj(PSF_FFT)                    # H, W, Nz
    PAM_FFT = torch.fft.fftshift( torch.real( torch.fft.irfft2( PAM_FFT, s =(Nx,Ny), dim=(0,1) ) ), dim=(0,1) ) # H, W, Nz  

    return PAM_FFT  # Nx,Ny,Nz
  

class openPAM_FFT_Module(torch.nn.Module):
#  The forward model for openPAM using autograd Module (can be grad w.r.t. 
#  both the psf or the unknow PAM image)
#  1 PAM_HD: the postive high resolution (z) 3D PAM image in the shape of Nt, Nx, Ny a torch.tensor
#  2 psf_stack: the normalized point spread function or simply the kernel of the depth dependent psf, similar as PAM_HD
#  3 psf_transducer: the temporal kernel (envelop) of the transucer, a one-D torch.tensor: 1, Nt
#   ******** The forward model of the image formation is:   *******
#   PAM_3D = psf_transducer *t (PAM_3D(K,:,:) *xy psf_stack(K,:,:))
#  where *t is conv in time while *xy is conv in space
    def __init__(self, measure3Dshape, psf_stack, psf_transducer):
        # psf_stack should be in [Nx, Ny, Nz]             
        super(openPAM_FFT_Module, self).__init__()    
        psf_cuda = psf_stack            # H, W, Nz
        nx, ny, nz = psf_cuda.shape
    
        self.Nx, self.Ny, self.Nz = measure3Dshape[0], measure3Dshape[1], measure3Dshape[2]
        nx_s, ny_s = np.int32(np.round((self.Nx-nx)/2.0)), np.int32(np.round((self.Ny-ny)/2.0))
        self.psf_stack_cuda = torch.zeros(measure3Dshape, dtype = torch.float32, device = 'cuda')    
        self.psf_stack_cuda[nx_s: nx_s+ nx, ny_s: ny_s + ny, :]  = psf_cuda
        # Process temporal convolution in parallel   
        idx_max = torch.argmax(psf_transducer).item()
        
        psf_trans = psf_transducer.float().view(1, -1).cuda() # 1, Nz
        self.psf_trans_cuda = torch.zeros((1, self.Nz), dtype = torch.float32, device = 'cuda')
        self.psf_trans_cuda[0, np.int32( np.round( self.Nz/2.0) ) - idx_max 
                       :  np.int32( np.round( self.Nz/2.0) ) - idx_max + psf_trans.shape[1] ] = psf_trans
        self.PSF_FFT = torch.fft.rfft2( torch.fft.ifftshift(self.psf_stack_cuda, dim=(0,1)), dim=(0,1) )  
        self.trans_FFT = torch.fft.rfft( torch.fft.ifftshift( self.psf_trans_cuda, dim = 1), dim = 1 )                                # 1, Nz

    def forward(self, PAM_HD):   
        # depthwise convolution
        PAM_FFT = torch.fft.rfft2( torch.fft.ifftshift(PAM_HD, dim=(0,1)) , dim=(0,1) )   
        PAM_FFT = PAM_FFT * self.PSF_FFT                      # H, W, Nz
        PAM_FFT = torch.fft.fftshift( torch.real( torch.fft.irfft2( PAM_FFT, s =(self.Nx, self.Ny), dim=(0,1) )), dim=(0,1) )  # H, W, Nz
            
        PAM_FFT = torch.fft.rfft( torch.fft.ifftshift( PAM_FFT, dim = 2 ), dim = 2 )                 # H, W, Nz
        PAM_FFT = PAM_FFT * self.trans_FFT   # (H, W, Nz) .* (1, Nz)
        PAM_FFT = torch.fft.fftshift( torch.real( torch.fft.irfft(PAM_FFT, n = self.Nz, dim = 2 ) ), dim = 2 )     # H, W, Nz
    
        return PAM_FFT  # Nx,Ny,Nz
    
    def adjoint(self, PAM_LQ):
        PAM_FFT = torch.fft.rfft( torch.fft.ifftshift(PAM_LQ , dim = 2), dim = 2 )                 # H, W, Nz
        PAM_FFT = PAM_FFT * torch.conj( self.trans_FFT)                  # H, W, Nz
        PAM_FFT = torch.fft.fftshift( torch.real( torch.fft.irfft(PAM_FFT, n = self.Nz,  dim = 2 ) ), dim=2 )   # H, W, Nz
    
        # depthwise convolution
        PAM_FFT = torch.fft.rfft2( torch.fft.ifftshift(PAM_FFT, dim=(0,1)), dim =(0,1) )  
        PAM_FFT = PAM_FFT * torch.conj(self.PSF_FFT)                    # H, W, Nz
        PAM_FFT = torch.fft.fftshift( torch.real( torch.fft.irfft2( PAM_FFT, s =(self.Nx, self.Ny), dim=(0,1) ) ), dim=(0,1) ) # H, W, Nz  
    
        return PAM_FFT  # Nx,Ny,Nz
    
        
