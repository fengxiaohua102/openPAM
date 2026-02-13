import numpy as np
import torch
import skimage
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import einops
import time
import scipy.ndimage

def norm1(array_in):
    max_val = np.amax(array_in.flatten())
    min_val = np.amin(array_in.flatten())
    return (array_in)/(max_val-min_val)

def norm01(array_in):
    min_val = np.amin(array_in.flatten())
    array_out = array_in - min_val
    max_val = np.max(array_out.flatten())
    return (array_out)/(max_val)

def torch_norm01(array_in):
    min_val = torch.amin(array_in)
    array_out = array_in - min_val
    max_val = torch.max(array_out)
    return (array_out)/(max_val)

def util_imresize(array_in, r_scale):
    # resize the image along the first two dimensions
    nx_resize = np.int32(np.round(r_scale * array_in.shape[0]))
    ny_resize = np.int32(np.round(r_scale * array_in.shape[1]))
    array_out = skimage.transform.resize(array_in, (nx_resize, ny_resize, array_in.shape[2]))
    return array_out

def torch_imresize(array_in, r_scale):
    # resize the image along the first two dimensions
    nx_resize = np.int32(np.round(r_scale * array_in.shape[0]))
    ny_resize = np.int32(np.round(r_scale * array_in.shape[1]))
    array_torch = numpy2cuda(array_in)
    array_torch = einops.rearrange(array_torch, '(c h) w nb ->nb c h w', c=1)
    array_out = F.interpolate(array_torch, (nx_resize, ny_resize), mode='bicubic')
    array_out = cuda2numpy( einops.rearrange(array_out, 'nb c h w ->(c h) w nb', c= 1) )
    return array_out

def torch_imresize1D(array_in, r_scale_z):
    # resize the image along z dimension
    nz_resize = np.int32(np.round(r_scale_z * array_in.shape[2]))
    nxy_resize = array_in.shape[0] * array_in.shape[1]
    array_torch = numpy2cuda(array_in)
    array_torch = einops.rearrange(array_torch, '(nx nb c) ny nz ->nb c (nx ny) nz', c=1, nb = 1)
    array_out = F.interpolate(array_torch, (nxy_resize, nz_resize), mode='bicubic')
    array_out = cuda2numpy( einops.rearrange(array_out, 'nb c (nx ny) nz ->(nb c nx) ny nz', nx = array_in.shape[0] ) )
    return array_out

def hilbert_transform(cuda_tensor):
    # computes the hilbert transform of input tensor (cuda) shaped as [Nt, Nx, Ny]
    Nt = cuda_tensor.shape[0]
    cuda_tensor_fft = torch.fft.fft( torch.fft.fftshift(cuda_tensor, dim = 0), dim = 0, n = Nt)
    cuda_tensor_fft[Nt//2+1:,:,:] = 0.0
    cuda_tensor_fft[1:Nt//2,:,:] = 2.0 * cuda_tensor_fft[1:Nt//2,:,:]
    cuda_tensor_hilbert = torch.fft.ifftshift( torch.fft.ifft(cuda_tensor_fft, dim = 0, n = Nt), dim = 0)
    return torch.abs(cuda_tensor_hilbert)

def numpy2cuda(ndarray_in):
    return torch.from_numpy(ndarray_in).float().cuda()
def cuda2numpy(tensor_in):
    return tensor_in.cpu().numpy()

def pad2modulo(in_tensor, modulo=16):
    # pad the last two dimensions as multiples of modulo
    h, w = in_tensor.size()[-2:]
    paddingBottom = int(np.ceil(h/modulo)*modulo-h)
    paddingRight = int(np.ceil(w/modulo)*modulo-w)
    out_tensor = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(in_tensor)
    return out_tensor, paddingRight, paddingBottom



def tv1d(array2D):
    # array_dif = np.sum( np.abs(array2D[:,1:]- array2D[:,:-1]), axis= 1)
    array_dif = np.sum( np.abs(scipy.ndimage.gaussian_laplace(array2D, (0.1, 1.0))), axis = 1)
    return array_dif

def tv2d(array2D):
    # array_dif = np.sum( np.abs(array2D[:,1:]- array2D[:,:-1]), axis= 1)
    array_dif = np.sum( np.abs(scipy.ndimage.gaussian_laplace(array2D, (0, 1.5, 1.5))), axis = (1,2))
    return array_dif/np.std(array2D, axis =(1,2))


#%% Loss utilities
def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (N, C, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * torch.sqrt(h_variance + w_variance)
    return loss

def tv3d_loss(img, tv_weight, weight_z):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (N, C, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    - weight_z: Scalar giving the weight of TV along the z direction.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    z_variance = torch.sum(torch.pow(img[:,:-1,:,:] - img[:,1:,:,:], 2))
    loss = tv_weight * torch.sqrt(h_variance + w_variance + weight_z * z_variance)
    return loss

def l1_loss(img, l1_weight):
    """
    Compute l1 loss.
    Inputs:
    - img: PyTorch Variable of shape (N, C, H, W) holding an input image.
    - l1_weight: Scalar giving the weight w_t to use for the l1 loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by l1_weight.
    """
    loss = l1_weight * torch.mean(torch.abs(img))
    return loss

def postproc(img, threshold=0.02, gamma=1.0):
    """
    Post-processing for reconstruction results.
    1. Thresholding: values below threshold are set to 0.
    2. Gamma correction: apply gamma power.
    """
    img_out = img.copy()
    img_out[img_out < threshold] = 0.0
    if gamma != 1.0:
        img_out = np.power(img_out, gamma)
    return img_out



