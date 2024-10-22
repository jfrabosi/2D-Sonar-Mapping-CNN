
import os
import ctypes
import numpy as np
from numba import cuda


def test_cuda_dlls():
    cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    nvvm_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\nvvm\bin"

    os.add_dll_directory(cuda_bin_path)
    os.add_dll_directory(nvvm_bin_path)

    dlls_to_test = [
        (cuda_bin_path, "cudart64_12.dll"),
        (nvvm_bin_path, "nvvm64_40_0.dll")
    ]

    for path, dll_name in dlls_to_test:
        full_path = os.path.join(path, dll_name)
        print(f"Testing {dll_name}:")
        print(f"  Path: {full_path}")
        print(f"  Exists: {os.path.exists(full_path)}")
        try:
            ctypes.CDLL(full_path)
            print(f"  Successfully loaded")
        except Exception as e:
            print(f"  Failed to load: {e}")
        print()


def test_cuda_kernel():
    @cuda.jit
    def add_kernel(x, y, out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = x[i] + y[i]

    n = 1000000
    x = np.arange(n).astype(np.float32)
    y = 2 * x
    out = np.zeros_like(x)

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_out = cuda.to_device(out)

    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)

    out = d_out.copy_to_host()

    print("CUDA kernel test:")
    print("  Kernel executed successfully")
    print(f"  First few elements: {out[:5]}")
    print(f"  Last few elements: {out[-5:]}")


def main():
    print("CUDA Setup Test\n")

    print("CUDA DLL Test:")
    test_cuda_dlls()

    print("\nCUDA Kernel Test:")
    test_cuda_kernel()


if __name__ == "__main__":
    main()