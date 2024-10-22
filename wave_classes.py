import numpy as np


class Object2D:
    def __init__(self, shape, size, angle_deg, pos_x, pos_y, dim_x, dim_y, buffer_size, is_present=True):
        match shape:
            case "circle":
                self.shape = "circle"
            case "square":
                self.shape = "square"
            case _:
                self.shape = "circle"
                print("Invalid shape, defaulted to circle")

        self.size = size
        self.angle = angle_deg * np.pi / 180
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.buffer_size = buffer_size
        self.is_present = is_present

    def create_mask(self):
        if not self.is_present:
            return np.zeros((self.dim_x + 2*self.buffer_size, self.dim_y + 2*self.buffer_size), dtype=bool)

        match self.shape:
            case "circle":
                x, y = np.ogrid[:self.dim_x + 2*self.buffer_size, :self.dim_y + 2*self.buffer_size]
                mask = (x - (self.pos_x + self.buffer_size))**2 + (y - (self.pos_y + self.buffer_size))**2 <= self.size**2
                return mask

            case "square":
                x, y = np.ogrid[:self.dim_x + 2*self.buffer_size, :self.dim_y + 2*self.buffer_size]
                x_cent = x - (self.pos_x + self.buffer_size)
                y_cent = y - (self.pos_y + self.buffer_size)
                x_rot = x_cent * np.cos(self.angle) - y_cent * np.sin(self.angle)
                y_rot = x_cent * np.sin(self.angle) + y_cent * np.cos(self.angle)
                mask = (np.abs(x_rot) <= self.size) & (np.abs(y_rot) <= self.size)
                return mask

            case _:
                print("Unexpected error")
                return np.zeros((self.dim_x + 2*self.buffer_size, self.dim_y + 2*self.buffer_size), dtype=bool)


class Emitter:
    def __init__(self, duration_s, amplitude, power, start_freq_hz, stop_freq_hz, pos_x, pos_y, buffer_size):
        self.duration_s = duration_s
        self.start_freq_hz = start_freq_hz
        self.stop_freq_hz = stop_freq_hz
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.amplitude = amplitude
        self.power = power
        self.buffer_size = buffer_size

    def chirp(self, u, t):
        if t <= self.duration_s:
            alpha = 1
            beta = (self.stop_freq_hz - self.start_freq_hz) / self.duration_s
            omega = 2 * np.pi * (self.start_freq_hz * t + beta * t**self.power / self.power)
            output = self.amplitude * alpha * np.sin(omega)
            u[0, self.pos_x, self.pos_y] = output


class Probe:
    def __init__(self, pos_x, pos_y, buffer_size):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.buffer_size = buffer_size
        self.data = []

    def save_data(self, u):
        self.data.append(u[0, self.pos_x, self.pos_y])

    def return_data(self):
        return self.data