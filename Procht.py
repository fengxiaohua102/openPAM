import torch
import numpy as np
from tifffile import imread, imwrite
import os
import einops
from matplotlib import pyplot as plt
import time
from utils_module import numpy2cuda, cuda2numpy

def roll_col(mat, shifts):
    # function that roll the 2D cuda tensor on col direction with different shifts for each col.
    h, w = mat.shape
    rows, cols = torch.arange(h), torch.arange(w)
    rows, cols = rows.cuda().long(), cols.cuda().long()   #long integer type
    shifted = torch.sub(rows.unsqueeze(1).repeat(1,w), shifts.repeat(h,1)) % h #shifted time idx = referred idx - jitter correcting shift value
    return  mat[shifted.long(), cols.long()]   #broadcasting in cols(dim()=1) direction
def xcorr_func(PD_sig_cuda):
    # using conv1d to find the shifts, the input should be in [Nt, Nx]
    Nz, Nx = PD_sig_cuda.shape
    weight_kernel = PD_sig_cuda[:,0].unsqueeze(0)  # the reference PD signal, shaped into [1, 1, Nt]
    weight_kernel = weight_kernel - torch.mean(weight_kernel)
    PD_sig_cuda = einops.rearrange(PD_sig_cuda,'nt nxny -> nxny nt').unsqueeze(0) # shaped into [1, Nt, nxny]
    PD_sig_cuda = PD_sig_cuda - torch.min(PD_sig_cuda)
    corr_res = torch.nn.functional.conv1d(PD_sig_cuda, weight_kernel.repeat(Nx,1,1), padding = (Nz-1)//1, stride = 1, groups = Nx)
    max_PD = torch.argmax(corr_res.squeeze(0), dim = 1)
    N_shift = torch.max(max_PD) - max_PD     # the amount of shift w.r.t. the reference (max)
    # PD_sig_cuda = einops.rearrange(PD_sig_cuda.squeeze(0), 'nxny nt -> nt nxny')
    return N_shift, corr_res
def freadu12bit(filename):
    fileread = open(filename, 'rb')
    datastream = fileread.read()
    data = read_uint12(datastream)
    return data
def read_uint12(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = ((mid_uint8 & 0x0F) << 8) | fst_uint8
    snd_uint12 = (lst_uint8 << 4) | ((mid_uint8 & 0xF0) >> 4)
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])

root_dir = os.getcwd() + r"/ExpData_DH/"
data_fol = root_dir + '/rawdata_ht_extraction'
st = time.perf_counter()
ht = freadu12bit(root_dir + '/rawdata_ht_extraction/0.3na-ht_20250306-075348_2_1000_1000_512_PA1.bin')
PD = freadu12bit(root_dir + '/rawdata_ht_extraction/0.3na-ht_20250306-075348_2_1000_1000_512_PD1.bin')
stop_t = time.perf_counter()
print('reading PA and PD data time is:', stop_t - st)
ht.shape

Nz, Nx, Ny = 512, 1000, 1000 # Shape defined by ht_extraction script, although it is also measurement?
PA_sig_cuda = numpy2cuda( np.float32( np.reshape(ht,(Nx, Ny,Nz)) ) )
PA_sig_cuda = einops.rearrange(PA_sig_cuda, 'Nx Ny Nz -> Nz (Nx Ny)')
PD = np.reshape(PD,(Nx, Ny,Nz))
PD_sig_cuda = numpy2cuda(np.float32(PD))
PD_sig_cuda = einops.rearrange(PD_sig_cuda, 'Nx Ny Nz -> Nz (Nx Ny)')

PA_sig_cuda=PA_sig_cuda[:,0:100000] # Take only 1e5/1e6 = 1/10 of xy scan points as raw reference points for ht extraction N_test=1e5
PD_sig_cuda=PD_sig_cuda[:,0:100000] # More computation if larger
N_shift, corr_res = xcorr_func(PD_sig_cuda)
st = time.perf_counter()
PA_sig_align = roll_col(PA_sig_cuda, torch.Tensor(N_shift).view(1, -1).cuda() )
PD_sig_align = roll_col(PD_sig_cuda, torch.Tensor(N_shift).view(1, -1).cuda() )
stop_t = time.perf_counter()
print('Jitter correction time is:', stop_t - st)
PA_sig_align = einops.rearrange(PA_sig_align, 'nz (nx ny) -> nz nx ny', ny = Ny)
PD_sig_align = einops.rearrange(PD_sig_align, 'nz (nx ny) -> nz nx ny', ny = Ny)

ax3 = plt.figure(figsize = (15,5))
ax3.add_subplot(1,2,1)
plt.imshow(cuda2numpy(PA_sig_align[:,:,400]))
plt.title('Raw PA signal with jitter correction')
ax3.add_subplot(1,2,2)
plt.imshow(cuda2numpy(PD_sig_align[:,:,400]))
plt.title('Raw PD signal')
PAM_MIP = torch.max(torch.abs(PA_sig_align), dim=0)[0] #max intensity projection in t(z) direction
plt.figure()
plt.imshow( cuda2numpy(PAM_MIP) )
plt.title('PA MIP')
plt.show()

val, idx = torch.sort(PAM_MIP.view(-1), descending = True)
idxx, idxy = np.unravel_index(idx[14].item(), (Nx,Ny))  # Convert 1D index to 2D
PAM_MIP[idxx, idxy] - torch.max(PAM_MIP)
ht_1D = PA_sig_align[:,idxx, idxy]
print('ht shape is:',ht_1D.shape)
plt.figure()
plt.plot(cuda2numpy(ht_1D) )
N_test = 20   # number of ht to store
ht_array = torch.zeros((Nz,N_test), dtype=torch.float32, device='cuda')
for K in range(N_test): # Take Ntest points (with high SNR) around the 3D volume source point O[:,idxx, idxy] as jitter correction reference points
    idxx, idxy = np.unravel_index(idx[K].item(), (Nx,Ny))
    ht_1D = PA_sig_align[:,idxx, idxy]
    ht_array[:,K] = ht_1D
# The signal from PA_sig_align is used to extract ht, why do we need another jitter correction?
N_shift, corr_res = xcorr_func(ht_array)
ht_array_align = roll_col(ht_array, N_shift)
ax = plt.figure(figsize=(6,6))
ax.add_subplot(1,2,1)
plt.imshow(cuda2numpy(ht_array) )
plt.title('Unaligned ht_array')
ax.add_subplot(1,2,2)
plt.imshow(cuda2numpy(ht_array_align) )
plt.title('Aligned ht_array')
plt.show()

st = time.perf_counter()
ht_array_align = ht_array_align - torch.mean(ht_array_align)
U, S, Vh = torch.linalg.svd(ht_array_align, full_matrices=False)
stop_t = time.perf_counter()
print('SVD time is:', stop_t - st)

S[1:] = 0.0 # Only take the basis corresponding to the largest singular value, i.e., the waveform basis most similar to all ntest point signals, which is the ht waveform 
ht_rec = U @ torch.diag(S) @ Vh
ax = plt.figure(figsize=(12,2))
ax.add_subplot(1,2,1)
plt.imshow(cuda2numpy(ht_array_align) , aspect ='auto')
plt.title('Aligned ht_array')
ax.add_subplot(1,2,2)
plt.imshow(cuda2numpy(ht_rec),  aspect ='auto' )
plt.title('SVD reconstructed ht_array')
ax = plt.figure(figsize=(12,2))
ax.add_subplot(1,2,1)
plt.plot(cuda2numpy(S) )
plt.title('Singular values')
ax.add_subplot(1,2,2)
plt.plot(cuda2numpy(ht_rec[220:-180,1]))
plt.title('Extracted ht (transducer psf)')
plt.show()

ht_f = (ht_rec[220:-180,1])
ht_f = ht_f - torch.mean(ht_f)
ht_f = ht_f / torch.max(ht_f)
torch.sum(ht_f)
plt.figure()
plt.plot(cuda2numpy(ht_f))
plt.title('Final extracted ht (transducer psf)')
plt.show()
imwrite(root_dir + '/ht.tif', cuda2numpy(ht_f))