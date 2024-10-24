import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_laser_distances(laser_distances, max_y):
    # Calculate y-coordinates for each laser measurement
    laser_spacing = max_y * 2 / (len(laser_distances) - 1)
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

    # Set equal axes
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Set x-axis limit to show a bit more than the maximum distance
    max_distance = max(valid_distances) if valid_distances else 0.1
    plt.xlim(-0.01, max_distance * 1.1)
    plt.ylim(-max_y * 1.1, max_y * 1.1)

    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_second_most_recent_simulation(filename, aye):
    try:
        with h5py.File(filename, 'r') as f:
            # Get all simulation keys and sort them
            sim_keys = sorted(f.keys())

            if len(sim_keys) < 2:
                print("Not enough simulations to visualize the second-most recent.")
                return

            # Select the second-most recent simulation
            second_most_recent = sim_keys[-aye]  # Changed to -2 to get the second-most recent
            sim_data = f[second_most_recent]

            # Extract data
            params = dict(sim_data.attrs)
            print(params)
            microphone_data = np.array(sim_data['microphone_data'])
            laser_distances = np.array(sim_data['laser_distances'])

            # Print the first five values from each microphone
            print("First five values from each microphone:")
            for i, probe_data in enumerate(microphone_data):
                print(f"Microphone {i}: {probe_data[:5]}")
            print()  # Add a blank line for better readability

            # Visualize wave heights
            plt.figure(figsize=(10, 6 * len(microphone_data)))
            for i, probe_data in enumerate(microphone_data):
                plt.subplot(len(microphone_data), 1, i + 1)
                plt.plot(probe_data)
                plt.title(f"Wave Height at Probe {i}")
                plt.ylabel("Wave Height")
                plt.grid(True)

            plt.tight_layout()
            plt.show()

            # Visualize laser distances
            # max_y = params['num_probes'] * params['rec_spacing']
            max_y = 3 * 0.01
            visualize_laser_distances(laser_distances, max_y)

            # Print simulation parameters
            print("Simulation Parameters:")
            for key, value in params.items():
                print(f"{key}: {value}")

    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

def count_simulations(filename):
    try:
        with h5py.File(filename, 'r') as f:
            num_simulations = len(f.keys())
        print(f"Number of simulations saved in {filename}: {num_simulations}")
        return num_simulations
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return 0
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return 0

# Example usage
if __name__ == "__main__":
    filename = "simulation_data/simulation_data_parallel.h5"
    num_sims = count_simulations(filename)
    print(f"Total number of simulations: {num_sims}")

    if num_sims >= 2:
        visualize_second_most_recent_simulation(filename, 1)
    else:
        print("Not enough simulations to visualize the second-most recent.")