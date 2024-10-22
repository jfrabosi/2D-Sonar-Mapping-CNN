import numpy as np
import random
import json
from wave_classes import Object2D, Emitter, Probe
from wave_sim import init_sim, update, get_lasers
import h5py
import multiprocessing as mp
from functools import partial

def save_data_ml_friendly(input_data, output_data, filename="simulation_data.h5"):
    with h5py.File(filename, 'a') as f:
        sim_group = f.create_group(f"simulation_{len(f.keys())}")
        for key, value in input_data["parameters"].items():
            sim_group.attrs[key] = value
        sim_group.create_dataset("microphone_data", data=np.array(input_data["microphone_data"]))
        sim_group.create_dataset("laser_distances", data=np.array(output_data["laser_distances"]))

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

def run_simulation(params):
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

    dim_x = d_wall_x + 2 * buf_size
    dim_y = d_wall_y + 2 * buf_size

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

    u, alpha = init_sim(buf_size, dim_x, dim_y)
    t = 0
    while t < total_dur:
        for emitter in ems:
            emitter.chirp(u, t)
        update(u, alpha, objects_mask, buf_size)
        if t >= chirp_dur:
            for probe in prs:
                probe.save_data(u)
        t += k

    laser_distances = get_lasers(prs, objects_mask, h, laser_spacing, buf_size)
    microphone_data = [probe.return_data() for probe in prs]

    return microphone_data, laser_distances

def process_simulation(i, num_simulations, filename):
    print(f"Running simulation {i + 1}/{num_simulations}")
    params = generate_random_parameters()
    microphone_data, laser_distances = run_simulation(params)

    input_data = {
        "parameters": params,
        "microphone_data": microphone_data
    }
    output_data = {
        "laser_distances": laser_distances
    }

    save_data_ml_friendly(input_data, output_data, filename)
    print(f"Simulation {i + 1} completed and data saved.")

def main():
    num_simulations = 1000  # Adjust this number as needed
    filename = "simulation_data_test1.h5"

    # Determine the number of CPU cores to use
    num_cores = mp.cpu_count() // 2
    print(f"Using {num_cores} CPU cores")

    # Create a pool of worker processes
    pool = mp.Pool(processes=num_cores)

    # Use partial to fix some arguments of process_simulation
    process_func = partial(process_simulation, num_simulations=num_simulations, filename=filename)

    # Map the process_func to the pool of workers
    pool.map(process_func, range(num_simulations))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    print(f"All {num_simulations} simulations completed.")

if __name__ == "__main__":
    main()