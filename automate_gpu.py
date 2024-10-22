import numpy as np
import random
import json
from wave_classes import Object2D, Emitter, Probe
from wave_sim import init_sim, update, get_lasers  # Importing necessary functions from your original script
import h5py
import numpy as np
from numba import cuda, float32


@cuda.jit
def update_gpu(u, alpha, object_mask):
    i, j = cuda.grid(2)
    if i > 0 and i < u.shape[1] - 1 and j > 0 and j < u.shape[2] - 1:
        if not object_mask[i, j]:
            new_value = alpha[i, j] * (
                    u[1, i - 1, j] + u[1, i + 1, j] +
                    u[1, i, j - 1] + u[1, i, j + 1] - 4 *
                    u[1, i, j]
            ) + 2 * u[1, i, j] - u[2, i, j]
            u[0, i, j] = new_value * 0.9995  # Energy removal

        # Debug: Print values for a specific point
        if i == 105 and j == 383:
            print(f"GPU Thread ({i}, {j}): u[1] = {u[1, i, j]}, u[2] = {u[2, i, j]}, new_value = {new_value}")

        # Update u[2] and u[1]
        u[2, i, j] = u[1, i, j]
        u[1, i, j] = u[0, i, j]


@cuda.jit
def apply_abc_gpu(u, coeff):
    i, j = cuda.grid(2)
    nx, ny = u.shape[1], u.shape[2]

    if 0 < i < nx - 1 and 0 < j < ny - 1:
        if i == 1:  # Left boundary
            u[0, i, j] = u[1, i + 1, j] + coeff * (u[0, i + 1, j] - u[1, i, j])
        elif i == nx - 2:  # Right boundary
            u[0, i, j] = u[1, i - 1, j] + coeff * (u[0, i - 1, j] - u[1, i, j])
        elif j == 1:  # Top boundary
            u[0, i, j] = u[1, i, j + 1] + coeff * (u[0, i, j + 1] - u[1, i, j])
        elif j == ny - 2:  # Bottom boundary
            u[0, i, j] = u[1, i, j - 1] + coeff * (u[0, i, j - 1] - u[1, i, j])


def init_sim_gpu(buffer_size, dim_x, dim_y, h, k):
    u = np.zeros((3, dim_x + 2 * buffer_size, dim_y + 2 * buffer_size), dtype=np.float32)
    c = 343  # The wave propagation speed
    alpha = np.zeros((dim_x + 2 * buffer_size, dim_y + 2 * buffer_size), dtype=np.float32)
    alpha[buffer_size:-buffer_size, buffer_size:-buffer_size] = ((c * k) / h) ** 2
    return cuda.to_device(u), cuda.to_device(alpha)


def update_gpu_wrapper(u_gpu, alpha_gpu, object_mask_gpu, coeff):
    threads_per_block = (32, 32)
    blocks_per_grid_x = (u_gpu.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (u_gpu.shape[2] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    print(f"Launching update_gpu kernel with grid: {blocks_per_grid}, block: {threads_per_block}")

    # Debug: Check u_gpu before update_gpu
    u_before_update = u_gpu.copy_to_host()
    print(f"Before update_gpu, max value: {np.max(np.abs(u_before_update))}")
    print(f"Before update_gpu, non-zero count: {np.count_nonzero(u_before_update)}")
    print(f"Before update_gpu, u[0,105,383]: {u_before_update[0, 105, 383]}")
    print(f"Before update_gpu, u[1,105,383]: {u_before_update[1, 105, 383]}")
    print(f"Before update_gpu, u[2,105,383]: {u_before_update[2, 105, 383]}")

    update_gpu[blocks_per_grid, threads_per_block](u_gpu, alpha_gpu, object_mask_gpu)

    # Debug: Check u_gpu after update_gpu
    u_after_update = u_gpu.copy_to_host()
    print(f"After update_gpu, max value: {np.max(np.abs(u_after_update))}")
    print(f"After update_gpu, non-zero count: {np.count_nonzero(u_after_update)}")
    print(f"After update_gpu, u[0,105,383]: {u_after_update[0, 105, 383]}")
    print(f"After update_gpu, u[1,105,383]: {u_after_update[1, 105, 383]}")
    print(f"After update_gpu, u[2,105,383]: {u_after_update[2, 105, 383]}")

    print(f"Launching apply_abc_gpu kernel with grid: {blocks_per_grid}, block: {threads_per_block}")

    # Debug: Check specific values before apply_abc_gpu
    print(f"Before apply_abc_gpu, u[0,1,1]: {u_after_update[0, 1, 1]}")
    print(f"Before apply_abc_gpu, u[0,-2,1]: {u_after_update[0, -2, 1]}")
    print(f"Before apply_abc_gpu, u[0,1,-2]: {u_after_update[0, 1, -2]}")

    apply_abc_gpu[blocks_per_grid, threads_per_block](u_gpu, coeff)

    # Debug: Check u_gpu after apply_abc_gpu
    u_after_abc = u_gpu.copy_to_host()
    print(f"After apply_abc_gpu, max value: {np.max(np.abs(u_after_abc))}")
    print(f"After apply_abc_gpu, non-zero count: {np.count_nonzero(u_after_abc)}")
    print(f"After apply_abc_gpu, u[0,105,383]: {u_after_abc[0, 105, 383]}")
    print(f"After apply_abc_gpu, u[1,105,383]: {u_after_abc[1, 105, 383]}")
    print(f"After apply_abc_gpu, u[2,105,383]: {u_after_abc[2, 105, 383]}")

    cuda.synchronize()
    print("GPU kernels executed successfully")


def generate_random_parameters():
    present = random.choice([1, 3])
    return {
        "sq_present": 1 if present in [1, 3] else 0,
        "cr_present": 1 if present in [2, 3] else 0,
        "sq_size": random.uniform(0.001, 0.03),
        "cr_size": random.uniform(0.001, 0.03),
        "sq_angle": random.uniform(-45, 45),
        "sq_x": random.uniform(0.03, 0.1),
        "cr_x": random.uniform(0.03, 0.1),
        "sq_y": random.uniform(-0.03, 0.03),
        "cr_y": random.uniform(-0.03, 0.03)
    }


def run_simulation_gpu(params):
    # Set up simulation parameters
    rec_spacing = 0.01
    num_probes = 3
    chirp_dur = 0.0001
    chirp_start_freq = 20000
    chirp_stop_freq = 200000
    cellsize = 2
    h = 0.0001
    buf_size = 5
    k = h * 0.001
    laser_spacing = 0.001

    # Calculate simulation dimensions and duration
    dist_to_objects = max((-params["sq_size"] * 1.414 + np.sqrt(
        params["sq_x"] ** 2 + (abs(params["sq_y"]) + num_probes * rec_spacing) ** 2)) * params["sq_present"],
                          (-params["cr_size"] + np.sqrt(
                              params["cr_x"] ** 2 + (abs(params["cr_y"]) + num_probes * rec_spacing) ** 2)) * params[
                              "cr_present"],
                          0.105)
    total_dur = (chirp_dur + 2 * dist_to_objects / 343) * 1.1
    d_wall_x = int(1.05 * max((params["sq_x"] + params["sq_size"] * 1.414) * params["sq_present"] / h,
                              (params["cr_x"] + params["cr_size"]) * params["cr_present"] / h,
                              300))
    d_wall_y = int(2.1 * max((abs(params["sq_y"]) + params["sq_size"] * 1.414) * params["sq_present"],
                             (abs(params["cr_y"]) + params["cr_size"]) * params["cr_present"],
                             rec_spacing * (num_probes + 1)) / h)

    # Adjust dimensions to include buffer
    dim_x = d_wall_x + 2 * buf_size
    dim_y = d_wall_y + 2 * buf_size

    # Create objects
    cr0 = Object2D(shape="circle",
                   size=int(params["cr_size"] / h),
                   angle_deg=0,
                   pos_x=10 + int(params["cr_x"] / h),
                   pos_y=d_wall_y // 2 + int(params["cr_y"] / h),
                   dim_x=dim_x,
                   dim_y=dim_y,
                   buffer_size=buf_size,
                   is_present=params["cr_present"])

    sq0 = Object2D(shape="square",
                   size=int(params["sq_size"] / h),
                   angle_deg=params["sq_angle"],
                   pos_x=10 + int(params["sq_x"] / h),
                   pos_y=d_wall_y // 2 + int(params["sq_y"] / h),
                   dim_x=dim_x,
                   dim_y=dim_y,
                   buffer_size=buf_size,
                   is_present=params["sq_present"])

    objects_mask = cr0.create_mask() | sq0.create_mask()

    # Create emitters and probes
    ems = [
        Emitter(duration_s=chirp_dur,
                amplitude=1000,
                power=2,
                start_freq_hz=chirp_start_freq,
                stop_freq_hz=chirp_stop_freq,
                pos_x=int(0.01 / h) + buf_size,
                pos_y=dim_y // 2 + int(rec_spacing * i / h),
                buffer_size=buf_size)
        for i in range(-num_probes, num_probes + 1)
    ]

    prs = [
        Probe(pos_x=int(0.01 / h) + buf_size,
              pos_y=dim_y // 2 + int(rec_spacing * i / h),
              buffer_size=buf_size)
        for i in range(-num_probes, num_probes + 1)
    ]

    print("Initial setup complete.")
    print(f"Simulation dimensions: {dim_x}x{dim_y}")
    print(f"Buffer size: {buf_size}")
    print(f"Total duration: {total_dur}")

    # Initialize GPU arrays
    u_gpu, alpha_gpu = init_sim_gpu(buf_size, dim_x, dim_y, h, k)
    object_mask_gpu = cuda.to_device(objects_mask)

    print("GPU arrays initialized.")
    print(f"u_gpu shape: {u_gpu.shape}")
    print(f"alpha_gpu shape: {alpha_gpu.shape}")
    print(f"object_mask_gpu shape: {object_mask_gpu.shape}")

    c = 343  # wave speed
    coeff = float32((c * k - h) / (c * k + h))

    # Debug: Print alpha value for the specific point
    alpha_cpu = alpha_gpu.copy_to_host()
    print(f"Alpha value at (105, 383): {alpha_cpu[105, 383]}")

    t = 0
    chirp_applied = False
    while t < total_dur:
        # Copy u_gpu to CPU for emitter application
        u_cpu = u_gpu.copy_to_host()

        if t == 0:
            print(f"Initial u_cpu max value: {np.max(np.abs(u_cpu))}")

        for emitter in ems:
            emitter.chirp(u_cpu, t)

        if not chirp_applied and np.max(np.abs(u_cpu)) > 0:
            print(f"Chirp first applied at t={t:.6f}")
            print(f"After chirp, u_cpu max value: {np.max(np.abs(u_cpu))}")
            print(f"After chirp, u_cpu[0] non-zero values: {np.count_nonzero(u_cpu[0])}")
            print(f"After chirp, u_cpu[0] sum: {np.sum(u_cpu[0])}")
            non_zero_indices = np.nonzero(u_cpu[0])
            print(
                f"After chirp, first few non-zero indices: {list(zip(non_zero_indices[0][:5], non_zero_indices[1][:5]))}")
            print(f"After chirp, first few non-zero values: {u_cpu[0][non_zero_indices][:5]}")
            print(f"After chirp, u_cpu[0,105,383]: {u_cpu[0, 105, 383]}")
            print(f"After chirp, u_cpu[1,105,383]: {u_cpu[1, 105, 383]}")
            print(f"After chirp, u_cpu[2,105,383]: {u_cpu[2, 105, 383]}")
            chirp_applied = True

        u_gpu = cuda.to_device(u_cpu)

        update_gpu_wrapper(u_gpu, alpha_gpu, object_mask_gpu, coeff)

        if t >= chirp_dur:
            # Copy u_gpu to CPU for probe data collection
            u_cpu = u_gpu.copy_to_host()
            for probe in prs:
                probe.save_data(u_cpu)

        t += k

        if t % (total_dur / 10) < k:
            print(f"Simulation progress: {t / total_dur * 100:.1f}%")

    if not chirp_applied:
        print("Warning: Chirp was never applied during the simulation!")

    print("Simulation complete.")

    # Get final results
    u_final = u_gpu.copy_to_host()
    print(f"Final u_gpu max value: {np.max(np.abs(u_final))}")

    # Get laser data
    laser_distances = get_lasers(prs, objects_mask, h, laser_spacing, buf_size)

    # Prepare output data
    microphone_data = [probe.return_data() for probe in prs]

    # Print the first five values from each microphone
    print("First five values from each microphone:")
    for i, probe_data in enumerate(microphone_data):
        print(f"Microphone {i}: {probe_data[:5]}")
    print()  # Add a blank line for better readability

    return microphone_data, laser_distances


def save_data_ml_friendly(input_data, output_data, filename="simulation_data.h5"):
    with h5py.File(filename, 'a') as f:
        # Create a new group for this simulation
        sim_group = f.create_group(f"simulation_{len(f.keys())}")

        # Save input parameters
        for key, value in input_data["parameters"].items():
            sim_group.attrs[key] = value

        # Save microphone data
        sim_group.create_dataset("microphone_data", data=np.array(input_data["microphone_data"]))

        # Save laser data
        sim_group.create_dataset("laser_distances", data=np.array(output_data["laser_distances"]))


def main():
    num_simulations = 1000  # Adjust this number as needed
    filename = "simulation_data.h5"

    for i in range(num_simulations):
        print(f"Running simulation {i + 1}/{num_simulations}")
        params = generate_random_parameters()
        microphone_data, laser_distances = run_simulation_gpu(params)

        input_data = {
            "parameters": params,
            "microphone_data": microphone_data
        }
        output_data = {
            "laser_distances": laser_distances
        }

        save_data_ml_friendly(input_data, output_data, filename)

        print(f"Simulation {i + 1} completed and data saved.")

    print(f"All {num_simulations} simulations completed. Data saved to {filename}")


if __name__ == "__main__":
    main()