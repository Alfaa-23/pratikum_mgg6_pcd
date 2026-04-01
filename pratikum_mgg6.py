import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

# --- BAGIAN 1: PEMBUATAN KERNEL & NOISE ---

def generate_blur_kernel(sz=15, deg=30):
    kernel = np.zeros((sz, sz))
    rad = np.deg2rad(deg)
    mid = sz // 2
    
    # Menentukan titik ujung garis blur
    dx, dy = (sz/2) * np.cos(rad), (sz/2) * np.sin(rad)
    p1 = (int(mid - dx), int(mid - dy))
    p2 = (int(mid + dx), int(mid + dy))
    
    cv2.line(kernel, p1, p2, 1, thickness=1)
    return kernel / np.sum(kernel)

def add_gaussian_dist(src, std_dev=20):
    noise_layer = np.random.normal(0, std_dev, src.shape)
    out = np.add(src, noise_layer)
    return np.clip(out, 0, 255).astype(np.uint8)

def add_impulse_noise(src, density=0.05):
    res = src.copy()
    # Salt noise
    mask_s = np.random.rand(*src.shape) < (density / 2)
    res[mask_s] = 255
    # Pepper noise
    mask_p = np.random.rand(*src.shape) < (density / 2)
    res[mask_p] = 0
    return res

# --- BAGIAN 2: ALGORITMA RESTORASI ---

def apply_inverse(obs, h, threshold=0.1):
    obs_fft = np.fft.fft2(obs)
    h_fft = np.fft.fft2(h, s=obs.shape)
    
    # Menghindari pembagian dengan nol menggunakan threshold
    h_fft_stable = h_fft + threshold
    recon_fft = obs_fft / h_fft_stable
    
    output = np.abs(np.fft.ifft2(recon_fft))
    return np.uint8(np.clip(output, 0, 255))

def apply_wiener(obs, h, noise_power=0.05):
    obs_f = np.fft.fft2(obs)
    h_f = np.fft.fft2(h, s=obs.shape)
    
    h_sq = np.abs(h_f)**2
    # Formula: G * (H* / (|H|^2 + K))
    wiener_gain = np.conj(h_f) / (h_sq + noise_power)
    recon_f = wiener_gain * obs_f
    
    output = np.abs(np.fft.ifft2(recon_f))
    return np.uint8(np.clip(output, 0, 255))

def apply_richardson_lucy(obs, h, steps=15):
    obs = obs.astype(np.float64)
    recon = np.full(obs.shape, 0.5, dtype=np.float64)
    h_inv = h[::-1, ::-1] # Membalik kernel
    
    for _ in range(steps):
        est_conv = signal.convolve2d(recon, h, mode='same')
        relative_blur = obs / (est_conv + 1e-9)
        error_corr = signal.convolve2d(relative_blur, h_inv, mode='same')
        recon *= error_corr
        
    return np.uint8(np.clip(recon, 0, 255))

# --- BAGIAN 3: ANALISIS KUALITAS ---

def calculate_metrics(target, ref):
    diff = target.astype(np.float64) - ref.astype(np.float64)
    mse_val = np.mean(diff**2)
    psnr_val = 10 * np.log10((255**2) / (mse_val + 1e-9))
    
    # SSIM sederhana menggunakan OpenCV
    win_size = (11, 11)
    m1 = cv2.GaussianBlur(target.astype(np.float32), win_size, 1.5)
    m2 = cv2.GaussianBlur(ref.astype(np.float32), win_size, 1.5)
    
    v1 = cv2.GaussianBlur(target.astype(np.float32)**2, win_size, 1.5) - m1**2
    v2 = cv2.GaussianBlur(ref.astype(np.float32)**2, win_size, 1.5) - m2**2
    v12 = cv2.GaussianBlur(target.astype(np.float32)*ref.astype(np.float32), win_size, 1.5) - m1*m2
    
    c1, c2 = 6.5025, 58.5225
    ssim_map = ((2*m1*m2 + c1)*(2*v12 + c2)) / ((m1**2 + m2**2 + c1)*(v1 + v2 + c2))
    return mse_val, psnr_val, np.mean(ssim_map)

# --- BAGIAN 4: EKSEKUSI UTAMA ---

if __name__ == "__main__":
    # Load data
    raw = cv2.imread('img.jpeg', 0) 
    if raw is None:
        raw = np.zeros((256,256), dtype=np.uint8) 
    else:
        raw = cv2.resize(raw, (256, 256))

    # Proses degradasi
    psf_kernel = generate_blur_kernel(15, 30)
    blurred_img = cv2.filter2D(raw, -1, psf_kernel)
    degraded = add_gaussian_dist(blurred_img, 15)

    # Proses perbaikan
    res_inv = apply_inverse(degraded, psf_kernel, threshold=0.1)
    res_wie = apply_wiener(degraded, psf_kernel, noise_power=0.02)
    res_rl  = apply_richardson_lucy(degraded, psf_kernel, steps=12)

    # Tampilkan hasil metrik
    results = [("Inverse", res_inv), ("Wiener", res_wie), ("Rich-Lucy", res_rl)]
    for label, img_res in results:
        e, p, s = calculate_metrics(raw, img_res)
        print(f"[{label}] -> MSE: {e:.2f}, PSNR: {p:.2f}dB, SSIM: {s:.4f}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = ["Original", "Degraded", "PSF", "Inv Result", "Wiener Result", "RL Result"]
    imgs = [raw, degraded, psf_kernel, res_inv, res_wie, res_rl]

    for ax, title, im in zip(axes.ravel(), titles, imgs):
        ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()