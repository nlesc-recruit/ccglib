#!/usr/bin/env python3
import warnings

import argparse
import kernel_tuner as kt
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.pmt import PMTObserver
import numpy as np
import os
import re

try:
    import ncu_metrics
except ImportError:
    ncu_metrics = None

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def align(a, b):
    return int(np.ceil(a / b) * b)


def parse_args():
    default_ccglib_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Device name, used in output filename")
    parser.add_argument("-m", type=int, required=True, help="Size of M axis")
    parser.add_argument("-n", type=int, required=True, help="Size of N axis")
    parser.add_argument("-k", type=int, required=True, help="Size of K axis")
    parser.add_argument("-b", type=int, default=1, help="Size of batch axis (default: %(default)s)")
    parser.add_argument("--type_in", choices=["int1", "bfloat16", "float8e4m3", "float8e5m2", "float16", "float32"], required=True, help="Input data type")
    parser.add_argument("--type_out", choices=["int32", "bfloat16", "float16", "float32"], required=True, help="Output data type")
    parser.add_argument("--kernel", choices=["basic", "opt"], default="opt",
                        help="Tune the basic or opt kernel (default: %(default)s)")
    parser.add_argument("--backend", required=True, choices=["cupy", "hip"], help="Kernel Tuner backend")
    parser.add_argument("--observer", dest="observer_type", required=False, choices=["nvml", "pmt"],
                        help="Kernel Tuner power observer (default: None)")
    parser.add_argument("--ccglib", default=default_ccglib_dir, help="Path to ccglib directory (default: %(default)s)")
    parser.add_argument(
        "--ncu", action="store_true", help="Enable NCU metrics (CUDA only)"
    )
    parser.add_argument(
        "-f",
        dest="overwrite",
        action="store_true",
        help="Overwrite any existing .json files",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    name = args.name
    ccglib_dir = args.ccglib
    kernel_name = f"wmma_complex_gemm_{args.kernel}"
    m_global = args.m
    n_global = args.n
    k_global = args.k
    batch_size = args.b
    backend = args.backend
    type_in = args.type_in
    type_out = args.type_out

    observer_type = args.observer_type

    # As long as no verification of the output is done, only the size of the type matters
    type_to_numpy = {"int1": np.uint32, "int32": np.int32, "bfloat16": np.float16, "float8e4m3": np.int8, "float8e5m2": np.int8, "float16": np.float16, "float32": np.float32}
    # Item size in bits matches numpy type, except for int1 because of packing
    type_to_nbit = {k: 8 * np.dtype(v).itemsize for k, v in type_to_numpy.items()}
    type_to_nbit["int1"] = 1

    nbit_in = type_to_nbit[type_in]
    nbit_out = type_to_nbit[type_out]

    # on AMD GPUs, the warp size can be 32 or 64 and the shared memory size is different from nvidia
    if backend == "hip":
        from hip import hip
        device_properties = hip.hipDeviceProp_t()
        hip.hipGetDeviceProperties(device_properties, 0)
        warp_size = device_properties.warpSize
        smem_size = device_properties.sharedMemPerBlock
    else:
        # assume nvidia defaults
        warp_size = 32
        smem_size = 49152

    defines = {}
    defines["kernel_tuner"] = 1
    defines["BATCH_SIZE"] = batch_size
    defines["M_GLOBAL"] = m_global
    defines["N_GLOBAL"] = n_global
    defines["K_GLOBAL"] = k_global
    defines["N_PER_WARP"] = lambda p: int(p["N_PER_BLOCK"] // p["block_size_y"])
    defines["M_PER_WARP"] = lambda p: int(p["M_PER_BLOCK"] // p["block_size_z"])
    defines["K_PADDING"] = 0
    defines["NBIT_IN"] = nbit_in
    defines["NBIT_OUT"] = nbit_out
    defines["WARP_SIZE"] = warp_size
    defines["C_COMPLEX_PLANAR"] = 1
    defines["A_ROW_MAJOR"] = 1
    defines["B_COL_MAJOR"] = 1
    defines["C_ROW_MAJOR"] = 1

    # Extract ValueType from ccglib, as we need to provide the index into this enum for TYPE_IN and TYPE_OUT
    try:
        with open(f"{ccglib_dir}/include/ccglib/common/value_type.h") as f:
            header = f.read().replace('\n', ' ')
        match = re.search("enum ValueType {(.*?)};", header)
        value_type = match.group(1).replace(' ', '').split(',')
    except Exception as e:
        print("Failed to extract ValueType from ccglib:")
        raise

    type_in_idx = value_type.index(type_in)
    type_out_idx = value_type.index(type_out)
    defines["TYPE_IN"] = type_in_idx
    defines["TYPE_OUT"] = type_out_idx

    if type_in == "int1":
        kernel_file = "gemm_kernel_int1.cu"
        scaling_factor = 32
        defines["M_PER_WMMA"] = 16
        defines["N_PER_WMMA"] = 8
        defines["K_PER_WMMA"] = 256
    elif type_in in ["float8e4m3", "float8e5m2"]:
        kernel_file = "gemm_kernel_float.cu"
        scaling_factor = 1
        defines["M_PER_WMMA"] = 16
        defines["N_PER_WMMA"] = 8
        defines["K_PER_WMMA"] = 32
    elif type_in in ["bfloat16", "float16"]:
        kernel_file = "gemm_kernel_float.cu"
        scaling_factor = 1
        defines["M_PER_WMMA"] = 16
        defines["N_PER_WMMA"] = 16
        defines["K_PER_WMMA"] = 16
    elif type_in == "float32":
        kernel_file = "gemm_kernel_float.cu"
        scaling_factor = 1
        defines["M_PER_WMMA"] = 16
        defines["N_PER_WMMA"] = 16
        defines["K_PER_WMMA"] = 8
    else:
        raise ValueError(f"Invalid nbit: {nbit}")

    # block size x is always warp_size, so the other block sizes can be at
    # most 1024 / warp_size
    tune_params = {
        "block_size_x": [warp_size],  # must be warp size
        "block_size_y": [2**i for i in range(0, 6) if warp_size * 2**i <= 1024],
        "block_size_z": [2**i for i in range(0, 6) if warp_size * 2**i <= 1024],
        "M_PER_BLOCK": [2**i for i in range(3, 9) if 2**i >= defines["M_PER_WMMA"]],
        "N_PER_BLOCK": [2**i for i in range(3, 9) if 2**i >= defines["N_PER_WMMA"]],
    }

    # multiple buffers is only supported on nvidia
    # assume that when HIP is used, we are running on an AMD GPU
    if backend == "hip":
        tune_params["NBUFFER"] = [1]
    else:
        tune_params["NBUFFER"] = [1, 2, 4, 8]

    # Add tuning parameters to defines
    for k in tune_params:
        defines[k] = k

    dtype_ab = type_to_numpy[type_in]
    dtype_c = type_to_numpy[type_out]

    # optimized kernel expects padded input, allocate enough memory for maximally padded size
    if args.kernel == "opt":
        m_padded = align(m_global, max(tune_params["M_PER_BLOCK"]))
        n_padded = align(n_global, max(tune_params["M_PER_BLOCK"]))
        k_padded = align(k_global, defines["K_PER_WMMA"])
    else:
        # no padding
        m_padded = m_global
        n_padded = n_global
        k_padded = k_global

    A = np.zeros((batch_size, 2, m_padded, k_padded // scaling_factor), dtype=dtype_ab)
    B = np.zeros((batch_size, 2, n_padded, k_padded // scaling_factor), dtype=dtype_ab)
    C = np.zeros((batch_size, 2, n_padded, m_padded), dtype=dtype_c)

    problem_size = (n_padded, m_padded, batch_size)
    arguments = (C, A, B)

    grid_div = {
        "grid_div_x": lambda p: p["N_PER_BLOCK"],
        "grid_div_y": lambda p: p["M_PER_BLOCK"],
        "grid_div_z": lambda p: 1
    }

    metrics = {
        # Useful TFLOPS: excluding padding
        "TFLOPS": lambda p: 8e-9 * m_global * n_global * k_global * batch_size / p["time"],
        "N_PER_WARP": lambda p: p["N_PER_BLOCK"] // p["block_size_y"],
        "M_PER_WARP": lambda p: p["M_PER_BLOCK"] // p["block_size_z"]
    }

    observers = []
    if observer_type == "pmt":
        if backend == "hip":
            sensor_name = "rocm"
        else:
            sensor_name = "nvidia"
        pmtobserver = PMTObserver({sensor_name: 0}, use_continuous_observer=True)
        observers.append(pmtobserver)
        metrics["Watt"] = lambda p: 1e3 * p[f"{sensor_name}_energy"] / p["time"]
        metrics["TFLOPS/J"] = lambda p: 8e-12 * m_global * n_global * k_global * batch_size / p[f"{sensor_name}_energy"]
    elif observer_type == "nvml":
        nvmlobserver = NVMLObserver(["nvml_energy", "temperature"])
        observers.append(nvmlobserver)
        metrics["Watt"] = lambda p: 1e3 * p["nvml_energy"] / p["time"]
        metrics["TFLOPS/J"] = lambda p: 8e-12 * m_global * n_global * k_global * batch_size / p["nvml_energy"]

    if args.ncu:
        if not ncu_metrics:
            raise Exception("Could not import ncu_metrics")
        if backend == "hip":
            print("HIP backend selected, not enabling NCU metrics")
        else:
            observers.append(get_ncu_observer())
            metrics.update(get_ncu_metrics())

    with open(f"{ccglib_dir}/kernels/{kernel_file}", "r") as fp:
        kernel_source = fp.read()

    compiler_options = [f"-I{ccglib_dir}/kernels", f"-I{ccglib_dir}/include", f"-I{ccglib_dir}/include/ccglib/common", "-std=c++17"]

    def restrict(*args):
        param_names = list(tune_params.keys())
        assert len(args) == len(param_names)
        p = {}
        for i in range(len(param_names)):
            p[param_names[i]] = args[i]

        n_per_warp = int(p["N_PER_BLOCK"] // p["block_size_y"])
        m_per_warp = int(p["M_PER_BLOCK"] // p["block_size_z"])
        if m_per_warp == 0 or n_per_warp == 0:
            return False

        # factor 2 for complex
        ab_size = (
            2
            * (p["NBUFFER"] * (p["M_PER_BLOCK"] + p["N_PER_BLOCK"]))
            * defines["K_PER_WMMA"]
            * np.dtype(dtype_ab).itemsize
            / scaling_factor
        )
        c_size = (
            2
            * (p["M_PER_BLOCK"] / m_per_warp)
            * (p["N_PER_BLOCK"] / n_per_warp)
            * defines["M_PER_WMMA"]
            * defines["N_PER_WMMA"]
            * np.dtype(dtype_c).itemsize
        )

        m_is_padded = m_global % p["M_PER_BLOCK"] != 0
        n_is_padded = n_global % p["N_PER_BLOCK"] != 0
        # if-statement here replicates REQUIRES_SHARED_MEMORY in ccglib
        if (m_is_padded or n_is_padded or "C_COMPLEX_INTERLEAVED" in defines or nbit_out < nbit_in):
            smem_buffer_size = max(c_size, ab_size)
        else:
            smem_buffer_size = ab_size

        valid = (
            p["M_PER_BLOCK"] % m_per_warp == 0
            and p["N_PER_BLOCK"] % n_per_warp == 0
            and m_per_warp % defines["M_PER_WMMA"] == 0
            and n_per_warp % defines["N_PER_WMMA"] == 0
            and smem_buffer_size <= smem_size
            and p["block_size_x"] * p["block_size_y"] * p["block_size_z"] <= 1024
        )
        return valid

    filename_cache = (
        f"{name}_{kernel_name}_{type_in}_to_{type_out}_{batch_size}x{m_global}x{n_global}x{k_global}.json"
    )
    if args.overwrite and os.path.exists(filename_cache):
            os.remove(filename_cache)

    kt.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params,
                   restrictions=restrict,
                   compiler_options=compiler_options,
                   cache=filename_cache,
                   metrics=metrics, observers=observers,
                   defines=defines, lang=backend, verbose=True,
                   **grid_div)
