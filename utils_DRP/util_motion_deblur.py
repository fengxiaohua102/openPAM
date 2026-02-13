import math
import torch
import torch.nn.functional as F
import numpy as np

from scipy import ndimage
import matplotlib.pyplot as plt 
from PIL import Image 

import matplotlib as plot

def generate_motion_blur_kernel(angle_deg, length=15, thickness=1, kernel_size=None):
    """
    Generate linear motion blur kernel (based on explicit angle)
    
    Parameters:
        angle_deg (float): Blur direction angle (0° is horizontal right, 90° is vertical up)
        length (int): Blur trajectory length (pixels)
        thickness (int): Trajectory line width (pixels)
        kernel_size (int): Kernel matrix size
    
    Returns:
        kernel (np.ndarray): Normalized floating-point kernel matrix
    """
    if kernel_size is None:
        kernel_size = max(length, thickness) + 2
    
    # Step 1: Create basic horizontal line segment kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    start = center - length // 2
    end = start + length  # Ensure length is accurate
    start_row = center - (thickness // 2)
    end_row = start_row + thickness
    kernel[start_row:end_row, start:end] = 1.0
    
    # # Step 2: Calculate rotation angle
    # angle_rad = np.deg2rad(-angle_deg)  # SciPy rotation direction is opposite to OpenCV
    
    # Step 3: Execute rotation transform
    kernel = ndimage.rotate(
        kernel, 
        angle_deg, 
        reshape=False,  # Maintain original size
        order=0,        # Nearest neighbor interpolation
        mode='constant', 
        cval=0.0
    )
    
    # Step 4: Binarize processing
    kernel = (kernel > 0.1).astype(np.float32)
    
    # Step 5: Energy normalization
    if np.sum(kernel) == 0:
        return np.zeros_like(kernel)
    return kernel / np.sum(kernel)

def generate_adjoint_kernel(kernel):
    # Flip kernel
    adjoint = np.flip(kernel,  axis=(0, 1))
    # Take transpose and conjugate (handle complex cases)
    # adjoint = adjoint.T.conj() 
    return adjoint 

def hilbert_transform(cuda_tensor):
    # computes the hilbert transform of input tensor (cuda) shaped as [Nt, Nx, Ny]
    Nt = cuda_tensor.shape[-1]
    cuda_tensor_fft = torch.fft.fft( torch.fft.fftshift(cuda_tensor, dim = -1), dim = -1, n = Nt)
    cuda_tensor_fft[:,Nt//2+1] = 0.0
    cuda_tensor_fft[:,1:Nt//2] = 2.0 * cuda_tensor_fft[:,1:Nt//2]
    cuda_tensor_hilbert = torch.fft.ifftshift( torch.fft.ifft(cuda_tensor_fft, dim = -1, n = Nt), dim = -1)
    return torch.abs(cuda_tensor_hilbert)

def generate_ht_hilbert_kernel(ht, angle=0, scale=1.0, thickness=1): 
    """ 
    Generate Hilbert transform convolution kernel matrix, supporting affine transformation
    Parameters:
        ht : Input signal (1, nt) Tensor
        angle : Rotation angle (degrees)
        scale : Scaling factor
        thickness : Kernel thickness coefficient (implemented via numerical scaling)
    Returns:
        (kernel_size, kernel_size) convolution kernel matrix
    """ 
    nt = ht.shape[-1]  
    
    # Core calculation (avoid division by zero)
    ht_hilbert = hilbert_transform(ht)

    # check ht_hilbert
    # t = torch.linspace(0,  1, nt)
    # plt.figure()
    # plt.plot(t, ht.cpu().squeeze().numpy(), color='blue')
    # plt.title(f"original ht")

    # plt.figure()
    # plt.plot(t, ht_hilbert.cpu().squeeze().numpy(), color='red')
    # plt.title(f"ht_hilbert")
    # plt.show()

    # Rescale
    if scale!=1.0:
        nt = int(np.round(nt*scale))
        ht_hilbert = F.interpolate(ht_hilbert.unsqueeze(0).unsqueeze(0), (1,nt), mode='bicubic')
        ht_hilbert = ht_hilbert.squeeze(0).squeeze(0) 
    ht_hilbert = ht_hilbert.cpu().numpy()  
    
    # 1. Thickness control
    ht_hilbert = np.repeat(ht_hilbert,  thickness, axis=0)

    # 2. Create extended matrix
    kernel_size = nt+3
    full_kernel = np.zeros((kernel_size,  kernel_size), dtype=np.float32)  
    center = (kernel_size) // 2 
    start = center - nt // 2
    end = start + nt  # Ensure length is accurate
    start_col = center - (thickness // 2)
    end_col = start_col + thickness    
    full_kernel[start:end,start_col:end_col] = ht_hilbert.transpose(1,0) # Default vertical placement
 
    # 4. Construct affine transformation matrix
    kernel = ndimage.rotate(  
        full_kernel, 
        angle, 
        reshape=False,  # Maintain original size
        order=0,        # Nearest neighbor interpolation
        mode='constant', 
        cval=0.0 
    ) 
 
    # Step 5: Energy normalization
    if np.sum(kernel)  == 0: 
        return np.zeros_like(kernel)  
    
    return kernel / np.sum(kernel)

def generate_ht_norm_kernel(ht, angle=0, scale=1.0, thickness=1): 
    """ 
    Generate Hilbert transform convolution kernel matrix, supporting affine transformation
    Parameters:
        ht : Input signal (1, nt) Tensor
        angle : Rotation angle (degrees)
        scale : Scaling factor
        thickness : Kernel thickness coefficient (implemented via numerical scaling)
    Returns:
        (kernel_size, kernel_size) convolution kernel matrix
    """ 
    nt = ht.shape[-1]  
    
    # Core calculation norm
    ht_kernel = (ht-ht.min())/(ht.max()-ht.min())

    # check ht_hilbert
    # t = torch.linspace(0,  1, nt)
    # plt.figure()
    # plt.plot(t, ht.cpu().squeeze().numpy(), color='blue')
    # plt.title(f"original ht")

    # plt.figure()
    # plt.plot(t, ht_hilbert.cpu().squeeze().numpy(), color='red')
    # plt.title(f"ht_hilbert")
    # plt.show()

    # Rescale
    if scale!=1.0:
        nt = int(np.round(nt*scale))
        ht_hilbert = ht_hilbert.unsqueeze(0).unsqueeze(0)
        ht_hilbert = F.interpolate(ht_hilbert, (1,nt), mode='bicubic')
        ht_hilbert = ht_hilbert.squeeze(0).squeeze(0)  
    ht_kernel = ht_kernel.cpu().numpy()  
    
    # 1. Thickness control
    ht_kernel = np.repeat(ht_kernel,  thickness, axis=0)

    # 2. Create extended matrix 
    kernel_size = nt+3
    full_kernel = np.zeros((kernel_size,  kernel_size), dtype=np.float32)  
    center = (kernel_size) // 2 
    start = center - nt // 2
    end = start + nt  # Ensure length is accurate
    start_col = center - (thickness // 2)
    end_col = start_col + thickness    
    full_kernel[start:end,start_col:end_col] = ht_kernel.transpose(1,0) # Default vertical placement
 
    # 4. Construct affine transformation matrix
    kernel = ndimage.rotate(  
        full_kernel, 
        angle, 
        reshape=False,  # Maintain original size
        order=0,        # Nearest neighbor interpolation
        mode='constant', 
        cval=0.0 
    ) 
 
    # Step 5: Energy normalization
    if np.sum(kernel)  == 0: 
        return np.zeros_like(kernel)  
    
    return kernel / np.sum(kernel) 

def generate_adjoint_ht_hilbert_kernerl(kernel):
    # Flip kernel
    adjoint = np.flip(kernel,  axis=(0, 1))
    # Take transpose and conjugate (handle complex cases)
    # adjoint = adjoint.T.conj() 
    return adjoint 


# def validate_adjoint(kernel, adj_kernel, test_size=1000, use_normal=True):

#     if use_normal:
#         x = np.random.randn(test_size, test_size)
#         y = np.random.randn(test_size, test_size)
#     else:
#         x = np.random.rand(test_size, test_size)
#         y = np.random.rand(test_size, test_size)
    

#     # Ax = convolve2d(x, kernel, mode='same')
#     # A_star_y = convolve2d(y, adj_kernel, mode='same')
#     Ax = ndimage.filters.convolve(x, kernel, mode='wrap')
#     A_star_y = ndimage.filters.convolve(y, adj_kernel, mode='wrap')
#     left = np.vdot(y, Ax)
#     right = np.vdot(x, A_star_y)
    

#     denominator = 0.5 * (np.abs(left) + np.abs(right)) + 1e-15  # Avoid division by zero
#     relative_error = np.abs(left - right) / denominator
#     return relative_error < 1e-7

if __name__ == "__main__":

    k_len = 31
    r_angle = 15
    thick = 2

    # # cv2
    # image = cv2.imread("input.jpg")
    # kernel = get_motion_blur_kernel(x=-25, y=15, thickness=3, ksize=48)
    # blurred = cv2.filter2D(image, -1, kernel)
    # cv2.imwrite("blurred.png",blurred)

    # ndimage
    image = Image.open("input.jpg") 
    image = np.array(image)
    kernel = generate_motion_blur_kernel(angle_deg=r_angle, length=k_len, thickness=thick)
    kernel_adj = generate_adjoint_kernel(kernel)

    # Visualize kernel matrix
    plt.figure()
    plt.imshow(kernel, cmap='gray')
    plt.title(f"Motion Kernel (Angle={r_angle}°, Length={k_len})")

    plt.figure()
    plt.imshow(kernel_adj, cmap='gray')
    plt.title(f"Adjoint Motion Kernel")
    plt.show()

    blurred = ndimage.filters.convolve(image, np.expand_dims(kernel, axis=2), mode='wrap')
    Image.fromarray(blurred).save("output.jpg")
    # corroborate the result of the adjoint operator
    # flag = validate_adjoint(kernel,kernel_adj)

    print()