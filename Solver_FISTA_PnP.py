#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural deoncovolution algorithm based on Plug-and-Play FISTA
"""
import numpy as np
import torch
import torch.nn as nn
import os
import kornia.filters as kf
from vision_networks import *
from utils_module import pad2modulo
import os
import tifffile
from utils_module import numpy2cuda, cuda2numpy

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

def Solver_PnP_FISTA(A, At, b, x0, opt):
    # The objective function: F(x) = 1/2 ||y - hx||^2 + lambda |x|
    # Input: all except opt are torch tensors
    #   A: the forward operator function
    #   At: backward operator function
    #   b: measurment data of the forward model: b = Ax
    #   x0: initialization of the solution
    #   opt.lambda: weight constant for the regularization term
    # Output:
    #   x: output solution
    # Nothe: the regularziation is implemented by denoising
    # Author: Xiaohua Feng

    print(' - Running FISTA-RED with fixed step size')
    reg_lambda = opt['lambda']
    maxiter = opt['maxiter']
    vis = opt['vis']
    xk = x0.clone()
    yk = xk.clone()
    tk = 1
    L0 = opt['step']
    POSCOND= opt['POScond']
    # k-th (k=0) function, gradient, hessian
    # objk  = func(x0, b, A, reg_lambda)

    if (opt['denoiser'] == 'proxtv'):
        denoise = lambda noise_x,sigma_n: denoise_ProxTV3D(noise_x, sigma_n)
    elif(opt['denoiser'] == 'proxl1'):
        denoise = lambda noise_x,sigma_n: proxl1(noise_x, sigma_n)
    else:
        if(opt['denoiser'] == 'drunet'):
            network = load_model_DRUNet()
        elif(opt['denoiser'] == 'restormer'):
            network = load_model_Restormer()
        else: 
            network = load_model()

        denoise = lambda noise_x,sigma_n: denoise_net(network, opt, noise_x, sigma_n)

    convergence = []
    
    for iter_num in np.arange(maxiter):
        # Uncomment this block to use the vanilla FISTA algorithm
        x_old = xk 
        # y_old = yk
        t_old = tk
        # L0 = L0 * 0.999
        
        # Memory optimization: compute gradient with explicit cleanup
        with torch.no_grad():
            A_yk = A(yk)
            grad = At(A_yk - b)
            del A_yk  # Explicitly delete intermediate tensor
            yk = yk - (1/L0) * grad
            del grad  # Clean up gradient tensor
        
        # Denoising step
        xk = denoise(yk, reg_lambda/L0)  # torch.reshape(yg, x0.shape)
        # if(iter_num>10 and np.mod(iter_num, 5) ==0):
        #     zk = deblur(model_db, zk)
            
        fx = func(x_old, b, A)            
        if(POSCOND):
            xk[xk<0]=0 # positiveness constraint
        
        tk = (1 + np.sqrt(1+4*t_old**2) )/2
        
        # Memory optimization: reuse yk buffer
        with torch.no_grad():
            yk = xk + (t_old-1)/tk * (xk-x_old) # + (t_old)/tk*(zk-xk)
        
        if(POSCOND):
            yk[yk<0]=0 # positiveness constraint
            
        convergence.append(fx)
        if vis > 0 :
            print('\nIter:', iter_num, f'obj_val is: {func(xk,b,A):.8f}',end=' ')
            # print(convergence)
        
        # # Clean up memory periodically
        # if iter_num % 10 == 0:
        #     torch.cuda.empty_cache()
            
        if(iter_num>=5 and -(convergence[-1]-convergence[-3])/convergence[-1]<=opt['tol'] ):
            print('Solution stagnates, exit ...')
            break
    
    # Final cleanup
    torch.cuda.empty_cache()
                
    return xk, convergence


def normNdMatrix(x):
    norm_val_temp = torch.square(torch.abs(x))
    norm_val = torch.sum(norm_val_temp).cpu().item()
    return norm_val

def func(xk, b, A):
    e = b - torch.reshape(A(xk), b.shape)
    Fx = 0.5*normNdMatrix(e)
    return Fx

def denoise_net(net, opt, noisy, sigma_hat):
    try:
        Nx, Ny, Nz = noisy.shape
        Flag_2D = False
    except:
        Flag_2D = True
        Nx, Ny = noisy.shape
        noisy = noisy.unsqueeze(2)
        
    min_val = torch.min(noisy)   # for addressing the nonnegativity of the denoiser
    noisy = noisy - min_val  
    max_val = torch.max(noisy)
    noisy = noisy/max_val
    
    net.eval()
    
    img = torch.empty_like(noisy)  # Pre-allocate for better memory efficiency
    if(opt['denoiser'] == 'ffdnet'):
        sigma = torch.full((1,1,1,1), sigma_hat).type_as(noisy)  # noise power relative to 255 for 8-bit images
        with torch.inference_mode():
            for K in range(noisy.shape[2]):
                img[:,:,K] = net(noisy[:,:,K].unsqueeze(0).unsqueeze(0), sigma).squeeze(0).squeeze(0)
        
            # for P in range(img.shape[1]):
            #     img[:,P,:] = net(img[:,P,:].unsqueeze(0).unsqueeze(0), sigma*1.0).squeeze(0).squeeze(0)
            # for Q in range(img.shape[0]):
            #     img[Q,:,:] = net(img[Q,:,:].unsqueeze(0).unsqueeze(0), sigma*1.0).squeeze(0).squeeze(0)
            
    elif(opt['denoiser'] == 'drunet'):
        noise_map = torch.ones([1,1,Nx,Ny], dtype=torch.float32, device = 'cuda') * sigma_hat
        with torch.inference_mode():
            for K in range(noisy.shape[2]):
                img[:,:,K] = net(torch.cat([noisy[:,:,K].unsqueeze(0).unsqueeze(0), noise_map], dim =1) ).squeeze(0).squeeze(0)
            
            # Note that for 2e-3 regparam, 0.1 is used in yz and xz while 0.05 is used in 4e-3, and 0.05 for 2e-2
            # noise_map = torch.ones([1,1,Nx,Nz], dtype=torch.float32, device = 'cuda') * sigma_hat *0.1/2
            # for P in range(img.shape[1]):
            #     img[:,P,:] = net(torch.cat([img[:,P,:].unsqueeze(0).unsqueeze(0), noise_map], dim =1) ).squeeze(0).squeeze(0)
            
            # noise_map = torch.ones([1,1,Ny,Nz], dtype=torch.float32, device = 'cuda') * sigma_hat *0.1/2
            # for Q in range(img.shape[0]):
            #     img[Q,:,:] = net(torch.cat([img[Q,:,:].unsqueeze(0).unsqueeze(0), noise_map], dim =1) ).squeeze(0).squeeze(0)  
    else:
        ''' This case can deal with restormer (~10 times slower), KBNet (~10 times slower), swinir (~100 times slower than drunet)
            Refers to the RED by fixed point projection paper (RED-PRO) for using denoisiers with fixed sigma
        '''
        with torch.inference_mode():
            for K in range(noisy.shape[2]):
                denoised_tmp = net(noisy[:,:,K].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                img[:,:,K] = denoised_tmp * sigma_hat + noisy[:,:,K] * (1-sigma_hat)
                
            # for P in range(img.shape[1]):
            #     denoised_tmp = net(img[:,P,:].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            #     img[:,P,:] = denoised_tmp * sigma_hat + img[:,P,:] * (1-sigma_hat)
                
            # for Q in range(img.shape[0]):
            #     denoised_tmp = net(img[Q,:,:].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            #     img[Q,:,:] = denoised_tmp * sigma_hat + img[Q,:,:] * (1-sigma_hat)
                
    img = img*max_val + min_val
    img = img.squeeze(2) if Flag_2D else img
    img = proxl1(img, 3e-3)
    
    return img

def denoise_net_drp(net, opt, noisy, sigma_hat):
    try:
        Nz, Nx, Ny  = noisy.shape
        Flag_2D = False
    except:
        Flag_2D = True
        Nx, Ny = noisy.shape
        noisy = noisy.unsqueeze(0)
        
    min_val = torch.min(noisy)   # for addressing the nonnegativity of the denoiser
    noisy = noisy - min_val  
    max_val = torch.max(noisy)
    noisy = noisy/max_val

    net.eval()
    
    img = torch.empty_like(noisy)  # Pre-allocate for better memory efficiency
    if(opt['denoiser'] == 'ffdnet'):
        sigma = torch.full((1,1,1,1), sigma_hat).type_as(noisy)  # noise power relative to 255 for 8-bit images
        with torch.inference_mode():
            for K in range(noisy.shape[0]):
                img[K,:,:] = net(noisy[K,:,:].unsqueeze(0).unsqueeze(0), sigma).squeeze(0).squeeze(0)
        
            # for P in range(img.shape[1]):
            #     img[:,P,:] = net(img[:,P,:].unsqueeze(0).unsqueeze(0), sigma*1.0).squeeze(0).squeeze(0)
            # for Q in range(img.shape[0]):
            #     img[Q,:,:] = net(img[Q,:,:].unsqueeze(0).unsqueeze(0), sigma*1.0).squeeze(0).squeeze(0)
                
    elif(opt['denoiser'] == 'drunet'):
        noise_map = torch.ones([1,1,Nx,Ny], dtype=torch.float32, device = 'cuda') * sigma_hat
        with torch.inference_mode():
            for K in range(noisy.shape[0]):
                img[K,:,:] = net(torch.cat([noisy[K,:,:].unsqueeze(0).unsqueeze(0), noise_map], dim =1) ).squeeze(0).squeeze(0)
            
            # Note that for 2e-3 regparam, 0.1 is used in yz and xz while 0.05 is used in 4e-3, and 0.05 for 2e-2
            # noise_map = torch.ones([1,1,Nx,Nz], dtype=torch.float32, device = 'cuda') * sigma_hat *0.1/2
            # for P in range(img.shape[1]):
            #     img[:,P,:] = net(torch.cat([img[:,P,:].unsqueeze(0).unsqueeze(0), noise_map], dim =1) ).squeeze(0).squeeze(0)
            
            # noise_map = torch.ones([1,1,Ny,Nz], dtype=torch.float32, device = 'cuda') * sigma_hat *0.1/2
            # for Q in range(img.shape[0]):
            #     img[Q,:,:] = net(torch.cat([img[Q,:,:].unsqueeze(0).unsqueeze(0), noise_map], dim =1) ).squeeze(0).squeeze(0)  
    else:
        ''' This case can deal with restormer (~10 times slower), KBNet (~10 times slower), swinir (~100 times slower than drunet)
            Refers to the RED by fixed point projection paper (RED-PRO) for using denoisiers with fixed sigma
        '''
        with torch.inference_mode():
            for K in range(noisy.shape[0]):
                denoised_tmp = net(noisy[K,:,:].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                img[K,:,:] = denoised_tmp * sigma_hat + noisy[K,:,:] * (1-sigma_hat)
                
            # for P in range(img.shape[1]):
            #     denoised_tmp = net(img[:,P,:].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            #     img[:,P,:] = denoised_tmp * sigma_hat + img[:,P,:] * (1-sigma_hat)
                
            # for Q in range(img.shape[0]):
            #     denoised_tmp = net(img[Q,:,:].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            #     img[Q,:,:] = denoised_tmp * sigma_hat + img[Q,:,:] * (1-sigma_hat)
                
    img = img*max_val + min_val
    img = img.squeeze(0) if Flag_2D else img
    img = proxl1(img, 3e-3)
    
    return img


# **********************************************************************
''' Using ProxTV (3D or a single z) or ProxHessian for denoising a volumetric image
'''
# **********************************************************************
# **********************************************************************
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
    # z_variance = torch.sum(torch.pow(img[:-1,:,:] - img[1:,:,:], 2))
    loss = torch.sqrt(h_variance + w_variance ) # + z_variance
    return loss

def tv2d_loss(img):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (C, H, W) holding an input image.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
    """
    w_variance = torch.sum(torch.pow(img[:,:-1] - img[:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:-1,:] - img[1:,:], 2))
    # z_variance = torch.sum(torch.pow(img[:-1,:,:] - img[1:,:,:], 2))
    loss = torch.sqrt(h_variance + w_variance ) # + z_variance
    return loss

def hessian_loss(img):
    """
    Compute hessian loss.
    Inputs:
    - img: PyTorch Variable of shape (C, H, W) holding an input image.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the hessian loss
    """
    dx = img[:,:,:-1] - img[:,:,1:]
    dy = img[:,:-1,:] - img[:,1:,:]
    # dz = img[:-1,:,:] - img[1:,:,:]
    
    # second order
    dxx = 0.5*(img[:,:,:-2] - 2*img[:,:,1:-1] +img[:,:,2:])
    dyy = 0.5*(img[:,:-2,:] - 2*img[:,1:-1,:] +img[:,2:,:])
    dzz = 0.5*(img[:-2,:,:] - 2*img[1:-1,:,:] +img[2:,:,:])
    dxy = dx[:,:-1,:] - dx[:,1:,:]
    dxz = dx[:-1,:,:] - dx[1:,:,:]
    dyz = dy[:-1,:,:] - dy[1:,:,:]
    
    loss = torch.sum(torch.abs(dxx)) + torch.sum(torch.abs(dyy)) + 0.2*torch.sum(torch.abs(dzz))   \
          + torch.sum(torch.abs(dxy)) + 0.2*torch.sum(torch.abs(dxz)) + 0.2*torch.sum(torch.abs(dyz))

    return loss

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
        return self.l2_term(self.clean_image, self.noisy_image) + self.reg_param * tv2d_loss(self.clean_image) \
              + self.reg_param*2e-1 * torch.mean(torch.abs(self.clean_image))

    def get_clean_image(self):
        return self.clean_image
 
def denoise_ProxTV3D(noisy_image, sigma_hat):
    # define the total variation denoising network
    # noisy_image input is in shape of Nx, Ny, Nz
    tv_denoiser = TVDenoise(noisy_image, sigma_hat)
    
    # define the optimizer to optimize the 1 parameter of tv_denoiser
    # optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr = 0.01, momentum=0.9)
    optimizer = torch.optim.Adam(tv_denoiser.parameters(), lr= 1e-4)
    num_iters = 50
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

''' Proxl1 denoising with pytorch
'''
def proxl1(z, sigma):
    # z: the signal to be denoised
    # sigma: the noise threshold
    sgn_val = torch.sign(z) 
    sz = torch.abs(z)-sigma
    sz[sz<0] = 0
    sz = sgn_val * sz
    
    # 2: This soft thresholding function supports complex numbers
    # sz = max(abs(z)-T,0)./(max(abs(z)-T,0)+T).*z
    # # Handle the size
    # sz = reshape(sz,size_z)
    return sz
