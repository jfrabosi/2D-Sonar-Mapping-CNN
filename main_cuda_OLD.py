import numpy as np
import random
import json
import h5py
from numba import cuda, float32
import math
import pygame
import matplotlib.pyplot as plt
import warnings
from numba.core.errors import NumbaPerformanceWarning

# Import necessary classes and functions
from wave_classes import Object2D, Emitter, Probe
from wave_sim import init_sim, get_lasers


# CUDA kernel for wave equation update
@cuda.jit
def update_kernel(u, alpha, object_mask, buffer_size):
    i, j = cuda.grid(2)
    if (buffer_size <= i < u.shape[1] - buffer_size and
            buffer_size <= j < u.shape[2] - buffer_size):
        if not object_mask[i, j]:
            u[0, i, j] = (
                    alpha[i, j] * (
                    u[1, i - 1, j] + u[1, i + 1, j] +
                    u[1, i, j - 1] + u[1, i, j + 1] - 4 * u[1, i, j]
            ) + 2 * u[1, i, j] - u[2, i, j]
            )
        else:
            u[0, i, j] = 0

        # Energy removal (not part of the wave equation)
        u[0, i, j] *= 0.9995


# CUDA kernel for Mur's ABC
@cuda.jit
def mur_abc_kernel(u, coeff, buffer_size):
    i = cuda.grid(1)
    if i < u.shape[1] - 2 * buffer_size:
        # Left boundary
        u[0, buffer_size, buffer_size + i] = (
                u[1, buffer_size + 1, buffer_size + i] +
                coeff * (u[0, buffer_size + 1, buffer_size + i] - u[1, buffer_size, buffer_size + i])
        )
        # Right boundary
        u[0, -buffer_size - 1, buffer_size + i] = (
                u[1, -buffer_size - 2, buffer_size + i] +
                coeff * (u[0, -buffer_size - 2, buffer_size + i] - u[1, -buffer_size - 1, buffer_size + i])
        )
        # Top boundary
        u[0, buffer_size + i, buffer_size] = (
                u[1, buffer_size + i, buffer_size + 1] +
                coeff * (u[0, buffer_size + i, buffer_size + 1] - u[1, buffer_size + i, buffer_size])
        )
        # Bottom boundary
        u[0, buffer_size + i, -buffer_size - 1] = (
                u[1, buffer_size + i, -buffer_size - 2] +
                coeff * (u[0, buffer_size + i, -buffer_size - 2] - u[1, buffer_size + i, -buffer_size - 1])
        )


def update_cuda(u, alpha, object_mask, buffer_size):
    threadsperblock = (8, 8)
    blockspergrid_x = math.ceil(u.shape[1] / threadsperblock[0])
    blockspergrid_y = math.ceil(u.shape[2] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Swap u layers
    u[2], u[1] = u[1], u[2]
    u[1], u[0] = u[0], u[1]

    # Update interior
    update_kernel[blockspergrid, threadsperblock](u, alpha, object_mask, buffer_size)

    # Apply Mur's ABC
    c = 343  # wave speed
    h = 0.0001  # spatial step (assuming this value from the original script)
    k = h * 0.001 # timestep
    coeff = (c * k - h) / (c * k + h)

    threadsperblock_1d = 32
    blockspergrid_1d = math.ceil(u.shape[1] / threadsperblock_1d)
    mur_abc_kernel[blockspergrid_1d, threadsperblock_1d](u, coeff, buffer_size)

    # # Print input dimensions
    # print(f"Input dimensions: u.shape = {u.shape}")
    #
    # # Print launch configuration
    # print(f"2D Kernel: Grid Size = {blockspergrid}, Block Size = {threadsperblock}")
    #
    # # Print 1D kernel configuration
    # print(f"1D Kernel: Grid Size = {blockspergrid_1d}, Block Size = {threadsperblock_1d}")

def generate_random_parameters():
    present = random.choice([1, 3])
    sq_size = random.uniform(0.001, 0.03)
    cr_size = random.uniform(0.001, 0.03)
    sq_x = random.uniform(0.03, 0.1)
    cr_x = random.uniform(0.03, 0.1)
    sq_x = sq_x if sq_x - sq_size * 1.414 > 0.03 else 0.03 + sq_size * 1.414
    cr_x = cr_x if cr_x - cr_size * 1.414 > 0.03 else 0.03 + cr_size * 1.414
    return {
        "sq_present": 1 if present in [1, 3] else 0,
        "cr_present": 1 if present in [2, 3] else 0,
        "sq_size": sq_size,
        "cr_size": cr_size,
        "sq_angle": random.uniform(-45, 45),
        "sq_x": sq_x,
        "cr_x": cr_x,
        "sq_y": random.uniform(-0.03, 0.03),
        "cr_y": random.uniform(-0.03, 0.03)
    }


def save_data_ml_friendly(input_data, output_data, filename="simulation_data.h5"):
    with h5py.File(filename, 'a') as f:
        sim_group = f.create_group(f"simulation_{len(f.keys())}")
        for key, value in input_data["parameters"].items():
            sim_group.attrs[key] = value
        sim_group.create_dataset("microphone_data", data=np.array(input_data["microphone_data"]))
        sim_group.create_dataset("laser_distances", data=np.array(output_data["laser_distances"]))


def draw_probes(surface, probes, buffer_size, cellsize, radius=5, color=(255, 0, 0)):
    for probe in probes:
        pygame.draw.circle(surface, color,
                           ((probe.pos_x - buffer_size) * cellsize,
                            (probe.pos_y - buffer_size) * cellsize), radius)


def plot_wave_heights(time_points, probes):
    fig, axs = plt.subplots(len(probes), 1, figsize=(10, 6 * len(probes)), sharex=True)
    for i, probe in enumerate(probes):
        axs[i].plot(time_points, probe.return_data())
        axs[i].set_title(f"Wave Height at ({probe.pos_x}, {probe.pos_y})")
        axs[i].set_ylabel("Wave Height")
        axs[i].grid(True)

    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


def visualize_laser_distances(laser_distances, laser_spacing, max_y):
    y_coords = [max_y - i * laser_spacing for i in range(len(laser_distances))]
    valid_distances = [d for d in laser_distances if d != -1]
    valid_y_coords = [y for d, y in zip(laser_distances, y_coords) if d != -1]

    plt.figure(figsize=(10, 6))
    plt.scatter(valid_distances, valid_y_coords, c='blue', s=20, label='Object Detected')
    plt.axvline(x=0, color='r', linestyle='--', label='Laser Source')
    plt.xlabel('Distance (m)')
    plt.ylabel('Y-coordinate (m)')
    plt.title('Laser Distance Measurements')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    max_distance = max(valid_distances) if valid_distances else 0.1
    plt.xlim(-0.01, max_distance * 1.1)
    plt.ylim(-max_y * 1.1, max_y * 1.1)

    plt.tight_layout()
    plt.show()


def run_simulation(params, render_first=False):
    # Set up simulation parameters
    rec_spacing = 0.01
    num_probes = 3
    chirp_dur = 0.0001
    chirp_start_freq = 20000
    chirp_stop_freq = 100000
    cellsize = 1
    h = 0.0002
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

    # Run simulation
    u, alpha = init_sim(buf_size, dim_x, dim_y)

    # Move data to GPU
    u_gpu = cuda.to_device(u)
    alpha_gpu = cuda.to_device(alpha)
    objects_mask_gpu = cuda.to_device(objects_mask)

    t = 0
    t0 = 0
    time_data = []

    # Set up PyGame if rendering
    if render_first:
        pygame.init()

        # Calculate the dimensions of the central simulation area
        display_dim_x = dim_x - 2 * buf_size
        display_dim_y = dim_y - 2 * buf_size

        # Create the display surface
        display = pygame.display.set_mode((display_dim_x * cellsize, display_dim_y * cellsize))
        pygame.display.set_caption("2D Wave Sim (CUDA)")

        # We'll create pixeldata after we know its exact dimensions

    while t < total_dur:
        for emitter in ems:
            emitter.chirp(u_gpu, t)
        update_cuda(u_gpu, alpha_gpu, objects_mask_gpu, buf_size)

        # Copy data from GPU to CPU
        u_cpu = u_gpu.copy_to_host()

        if t >= chirp_dur:
            if t0 == 0:
                t0 = t
            for probe in prs:
                probe.save_data(u_cpu)
            time_data.append(t - t0)

        if render_first:
            # Extract the central part of the simulation, excluding the buffer
            central_u = u_cpu[:, buf_size:-buf_size, buf_size:-buf_size]

            # Create pixeldata with the exact dimensions of central_u
            if 'pixeldata' not in locals():
                pixeldata = np.zeros((central_u.shape[1], central_u.shape[2], 3), dtype=np.uint8)

            # Update pixeldata
            pixeldata[:, :, 0] = np.clip(central_u[0] + 128, 0, 255)
            pixeldata[:, :, 1] = np.clip(central_u[1] + 128, 0, 255)
            pixeldata[:, :, 2] = np.clip(central_u[2] + 128, 0, 255)

            # Adjust the object mask to match the central part
            central_mask = objects_mask[buf_size:-buf_size, buf_size:-buf_size]
            pixeldata[central_mask] = [64, 64, 64]  # Gray color for the object

            surf = pygame.surfarray.make_surface(pixeldata)
            display_surf = pygame.transform.scale(surf, (display_dim_x * cellsize, display_dim_y * cellsize))
            display.blit(display_surf, (0, 0))

            draw_probes(display, prs, buf_size, cellsize)

            pygame.display.update()

            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, None  # Early exit if the user closes the window

        t += k

        if t % 0.00001 < k and render_first:  # Print progress every 0.0001 seconds of simulation time
            print(f"Simulation progress: {t / total_dur * 100:.2f}%")

    if render_first:
        pygame.quit()

    # Get laser data
    laser_distances = get_lasers(prs, objects_mask, h, laser_spacing, buf_size)

    # Prepare output data
    microphone_data = [probe.return_data() for probe in prs]

    if render_first:
        plot_wave_heights(time_data, prs)
        max_y = num_probes * rec_spacing
        visualize_laser_distances(laser_distances, laser_spacing, max_y)

    # Print the first five values from each microphone
    print("First five values from each microphone:")
    for i, probe_data in enumerate(microphone_data):
        print(f"Microphone {i}: {probe_data[:5]}")
    print()  # Add a blank line for better readability

    return microphone_data, laser_distances


def main():
    num_simulations = 10000  # Adjust this number as needed
    filename = "simulation_data_simple.h5"

    for i in range(num_simulations):
        print(f"Running simulation {i + 1}/{num_simulations}")
        params = generate_random_parameters()
        # render_first = (i == 0)  # Render only the first simulation
        render_first = False

        # Suppress the specific Numba warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            microphone_data, laser_distances = run_simulation(params, render_first)

        input_data = {
            "parameters": params,
            "microphone_data": microphone_data
        }
        output_data = {
            "laser_distances": laser_distances
        }

        save_data_ml_friendly(input_data, output_data, filename)

        print(f"Simulation {i + 1} completed and data saved.")

    print(f"All simulations completed. Data saved to {filename}")


if __name__ == "__main__":
    main()