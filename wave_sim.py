import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from wave_classes import Object2D, Emitter, Probe\

# ML PARAMS
sq_present = 1      # 0 or 1, y/n; at least one present should be 1
cr_present = 1      # 0 or 1, y/n; at least one present should be 1
sq_size = 0.002     # 0.001 <= x <= 0.03
cr_size = 0.005     # 0.001 <= x <= 0.03
sq_angle = -45      # -45 <= x <= 45
sq_x = 0.07         # 0.03 <= x <= 0.1
cr_x = 0.1          # 0.03 <= x <= 0.1
sq_y = 0.01         # -0.03 <= x <= 0.03
cr_y = -0.02        # -0.03 <= x <= 0.03

# PARAMETERS FOR SIMULATION
rec_spacing = 0.01          # m, center-to-center spacing for receivers
num_probes = 3              # number of probes on each side; 0 produces -, 1 produces - - -
chirp_dur = 0.0001          # s
chirp_start_freq = 20000    # Hz
chirp_stop_freq = 100000    # Hz
cellsize = 1                # display size of a cell in pixel
steps_per_frame = 2         # skip frames to render faster
h = 0.0002                  # spatial step width, m
buf_size = 5                # Size of the absorbing boundary layer
laser_spacing = 0.001

# non-changeable parameters
k = h * 0.001               # time step width, s
dist_to_objects = max((-sq_size*1.414 + np.sqrt(sq_x**2 + (abs(sq_y)+num_probes*rec_spacing)**2))*sq_present,
                      (-cr_size + np.sqrt(cr_x**2 + (abs(cr_y)+num_probes*rec_spacing)**2))*cr_present,
                      0.02)
total_dur = (chirp_dur + 2 * dist_to_objects / 343) * 1.1
d_wall_x = int(1.05 * max((sq_x + sq_size*1.414)*sq_present/h,
                          (cr_x + cr_size)*cr_present/h,
                          300))
d_wall_y = int(2.1 * max((abs(sq_y) + sq_size*1.414)*sq_present,
                         (abs(cr_y) + cr_size)*cr_present,
                         rec_spacing*(num_probes+1)) / h)
sq_x = sq_x if sq_x - sq_size*1.414 > 0.03 else 0.03 + sq_size*1.414
cr_x = cr_x if cr_x - cr_size*1.414 > 0.03 else 0.03 + cr_size*1.414
print("Sim time: %.6fs" % total_dur)

# PARAMETERS FOR RENDERING
dim_x = d_wall_x              # width of the simulation domain
dim_y = d_wall_y             # height of the simulation domain
sim_dur_s = total_dur         # duration of simulation in seconds

# variables for data collection
time_data = []


# Create objects with the new dimensions and buffer size
cr0 = Object2D(shape="circle",
               size=int(cr_size / h),
               angle_deg=0,
               pos_x=10 + int(cr_x / h),
               pos_y=dim_y // 2 + int(cr_y / h),
               dim_x=dim_x,
               dim_y=dim_y,
               buffer_size=buf_size,
               is_present=cr_present)

sq0 = Object2D(shape="square",
               size=int(sq_size / h),
               angle_deg=sq_angle,
               pos_x=10 + int(sq_x / h),
               pos_y=dim_y // 2 + int(sq_y / h),
               dim_x=dim_x,
               dim_y=dim_y,
               buffer_size=buf_size,
               is_present=sq_present)

objects_mask = cr0.create_mask() | sq0.create_mask()

# place emitters
ems = [
    Emitter(duration_s=chirp_dur,
            amplitude=1000,
            power=2,
            start_freq_hz=chirp_start_freq,
            stop_freq_hz=chirp_stop_freq,
            pos_x=int(0.01 / h) + buf_size,
            pos_y=dim_y // 2 + int(rec_spacing * i / h) + buf_size,
            buffer_size=buf_size)
    for i in range(-num_probes, num_probes+1)
]

prs = [
    Probe(pos_x=int(0.01 / h) + buf_size,
          pos_y=dim_y // 2 + int(rec_spacing * i / h) + buf_size,
          buffer_size=buf_size)
    for i in range(-num_probes, num_probes+1)
]


def init_sim(buffer_size, dim_x, dim_y):
    u = np.zeros((3, dim_x + 2*buffer_size, dim_y + 2*buffer_size))
    c = 343  # The wave propagation speed
    alpha = np.zeros((dim_x + 2*buffer_size, dim_y + 2*buffer_size))
    alpha[buffer_size:-buffer_size, buffer_size:-buffer_size] = ((c * k) / h) ** 2
    return u, alpha


def update(u, alpha, object_mask, buffer_size):
    u[2] = u[1]
    u[1] = u[0]

    # Update the interior of the domain
    u[0, buffer_size:-buffer_size, buffer_size:-buffer_size] = alpha[buffer_size:-buffer_size, buffer_size:-buffer_size] * (
        u[1, buffer_size-1:-buffer_size-1, buffer_size:-buffer_size] +
        u[1, buffer_size+1:-buffer_size+1, buffer_size:-buffer_size] +
        u[1, buffer_size:-buffer_size, buffer_size-1:-buffer_size-1] +
        u[1, buffer_size:-buffer_size, buffer_size+1:-buffer_size+1] - 4 *
        u[1, buffer_size:-buffer_size, buffer_size:-buffer_size]
    ) + 2 * u[1, buffer_size:-buffer_size, buffer_size:-buffer_size] - u[2, buffer_size:-buffer_size, buffer_size:-buffer_size]

    # Apply Mur's first-order ABC on the boundaries
    c = 343  # wave speed
    coeff = (c * k - h) / (c * k + h)

    # Left and right boundaries
    u[0, buffer_size, buffer_size:-buffer_size] = u[1, buffer_size+1, buffer_size:-buffer_size] + coeff * (u[0, buffer_size+1, buffer_size:-buffer_size] - u[1, buffer_size, buffer_size:-buffer_size])
    u[0, -buffer_size-1, buffer_size:-buffer_size] = u[1, -buffer_size-2, buffer_size:-buffer_size] + coeff * (u[0, -buffer_size-2, buffer_size:-buffer_size] - u[1, -buffer_size-1, buffer_size:-buffer_size])

    # Top and bottom boundaries
    u[0, buffer_size:-buffer_size, buffer_size] = u[1, buffer_size:-buffer_size, buffer_size+1] + coeff * (u[0, buffer_size:-buffer_size, buffer_size+1] - u[1, buffer_size:-buffer_size, buffer_size])
    u[0, buffer_size:-buffer_size, -buffer_size-1] = u[1, buffer_size:-buffer_size, -buffer_size-2] + coeff * (u[0, buffer_size:-buffer_size, -buffer_size-2] - u[1, buffer_size:-buffer_size, -buffer_size-1])

    # Set wave amplitude to zero inside the object
    u[0][object_mask] = 0

    # Not part of the wave equation but I need to remove energy from the system.
    # The boundary conditions are closed. Energy cannot leave and the simulation keeps adding energy.
    u[0, 1:dim_x - 1, 1:dim_y - 1] *= 0.9995


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


def draw_probes(surface, probes, buffer_size, radius=5, color=(255, 0, 0)):
    for probe in probes:
        pygame.draw.circle(surface, color,
                           ((probe.pos_x - buffer_size) * cellsize,
                            (probe.pos_y - buffer_size) * cellsize), radius)


def get_lasers(prs, objects_mask, h, laser_spacing, buf_size):
    lasers = []
    min_y = min(probe.pos_y for probe in prs)
    max_y = max(probe.pos_y for probe in prs)
    probe_x = max(probe.pos_x for probe in prs)

    for y in np.arange(min_y, max_y + laser_spacing/h, laser_spacing/h):
        y_index = int(y)
        for x_index in range(buf_size, objects_mask.shape[0] - buf_size):
            if objects_mask[x_index, y_index]:
                distance = (x_index - probe_x) * h
                lasers.append(distance)
                break
        else:
            lasers.append(-1)

    return lasers


def visualize_laser_distances(laser_distances, laser_spacing, max_y):
    # Calculate y-coordinates for each laser measurement
    y_coords = [max_y - i * laser_spacing for i in range(len(laser_distances))]

    # Filter out -1 values
    valid_distances = []
    valid_y_coords = []
    for distance, y in zip(laser_distances, y_coords):
        if distance != -1:
            valid_distances.append(distance)
            valid_y_coords.append(y)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_distances, valid_y_coords, c='blue', s=20, label='Object Detected')

    # Add vertical line at x=0 to represent the laser source
    plt.axvline(x=0, color='r', linestyle='--', label='Laser Source')

    # Set labels and title
    plt.xlabel('Distance (m)')
    plt.ylabel('Y-coordinate (m)')
    plt.title('Laser Distance Measurements')

    # Customize the plot
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # equal axes
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Set x-axis limit to show a bit more than the maximum distance
    max_distance = max(valid_distances) if valid_distances else 0.1
    plt.xlim(-0.01, max_distance * 1.1)
    plt.ylim(-max_y * 1.1, max_y * 1.1)

    # Show the plot
    plt.tight_layout()
    plt.show()


def main():
    pygame.init()
    u, alpha = init_sim(buf_size, dim_x, dim_y)

    display_dim_x = dim_x  # Remove buffer from display dimensions
    display_dim_y = dim_y

    if steps_per_frame != 0:
        display = pygame.display.set_mode((display_dim_x * cellsize, display_dim_y * cellsize))
        pygame.display.set_caption("2D Wave Sim")
        pixeldata = np.zeros((display_dim_x, display_dim_y, 3), dtype=np.uint8)

    counter = -1
    t = 0
    t0 = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for emitter in ems:
            emitter.chirp(u, t)
        update(u, alpha, objects_mask, buf_size)
        if t >= chirp_dur:
            if t0 == 0:
                t0 = t
            for probe in prs:
                probe.save_data(u)
            time_data.append(t-t0)

        t += k
        counter += 1

        if counter >= steps_per_frame and steps_per_frame != 0:
            # Extract the central part of the simulation, excluding the buffer
            central_u = u[:, buf_size:-buf_size, buf_size:-buf_size]

            pixeldata[:, :, 0] = np.clip(central_u[0] + 128, 0, 255)
            pixeldata[:, :, 1] = np.clip(central_u[1] + 128, 0, 255)
            pixeldata[:, :, 2] = np.clip(central_u[2] + 128, 0, 255)

            # Adjust the object mask to match the central part
            central_mask = objects_mask[buf_size:-buf_size, buf_size:-buf_size]
            pixeldata[central_mask] = [64, 64, 64]  # Gray color for the object

            surf = pygame.surfarray.make_surface(pixeldata)
            display_surf = pygame.transform.scale(surf, (dim_x * cellsize, dim_y * cellsize))
            display.blit(display_surf, (0, 0))

            # Adjust probe drawing to account for buffer
            draw_probes(display, prs, buf_size)

            pygame.display.update()
            print("Simulation percentage complete: %.2f" % (100 * t / sim_dur_s))
            counter = 0

        elif steps_per_frame == 0 and counter % 10 == 0:
            print("Simulation percentage complete: %.2f" % (100 * t / sim_dur_s))

        if t >= sim_dur_s:
            running = False

    pygame.quit()
    plot_wave_heights(time_data, prs)
    laser_distances = get_lasers(prs, objects_mask, h, laser_spacing, buf_size)
    max_y = num_probes * rec_spacing
    visualize_laser_distances(laser_distances, laser_spacing, max_y)


if __name__ == "__main__":
    main()