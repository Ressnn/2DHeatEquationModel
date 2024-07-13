#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heat Equation Simulation with Fourier Transform

This script simulates the heat equation using Fourier transforms and creates an animation of the results.

Author: Pranav Devarinti (original), Claude (refactored)
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Heat Equation Simulation")
    parser.add_argument("--fourier_level", type=int, default=5, help="Fourier series truncation level")
    parser.add_argument("--x_start", type=float, default=-2, help="Start value for x")
    parser.add_argument("--x_stop", type=float, default=2.0001, help="Stop value for x")
    parser.add_argument("--y_start", type=float, default=-2, help="Start value for y")
    parser.add_argument("--y_stop", type=float, default=2.0001, help="Stop value for y")
    parser.add_argument("--spatial_step", type=float, default=0.05, help="Step size for x and y")
    parser.add_argument("--total_time", type=float, default=0.2, help="Total simulation time")
    parser.add_argument("--time_step", type=float, default=0.003, help="Time step")
    parser.add_argument("--diffusion_coeff", type=float, default=0.2, help="Diffusion coefficient")
    parser.add_argument("--output_filename", type=str, default="heat_equation_sim", help="Output filename")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the animation")
    return parser.parse_args()

def initial_temperature_distribution(x, y):
    return np.sin(x + y)

class FourierCoefficientsCalculator:
    def __init__(self, initial_function, x_start, x_stop, y_start, y_stop, fourier_level):
        self.initial_function = initial_function
        self.x_start = x_start
        self.x_length = x_stop - x_start
        self.y_start = y_start
        self.y_length = y_stop - y_start
        self.fourier_level = fourier_level       
        self.normalization_factor = 4 / (self.x_length * self.y_length)
        self._define_coefficient_functions()

    def _define_coefficient_functions(self):
        self._alpha_func = lambda x, y, n, m: (self.initial_function(x, y) * (np.cos(2 * np.pi * n * x / self.x_length)) * (np.cos(2 * np.pi * m * y / self.y_length)))
        self._beta_func = lambda x, y, n, m: (self.initial_function(x, y) * (np.cos(2 * np.pi * n * x / self.x_length)) * (np.sin(2 * np.pi * m * y / self.y_length)))
        self._gamma_func = lambda x, y, n, m: (self.initial_function(x, y) * (np.sin(2 * np.pi * n * x / self.x_length)) * (np.cos(2 * np.pi * m * y / self.y_length)))
        self._delta_func = lambda x, y, n, m: (self.initial_function(x, y) * (np.sin(2 * np.pi * n * x / self.x_length)) * (np.sin(2 * np.pi * m * y / self.y_length)))

    def _calculate_coefficient(self, func, n, m):
        grid_points = np.array(np.meshgrid(np.arange(0, n), np.arange(0, m))).transpose() 
        coefficient_grid = np.zeros(grid_points.shape[:-1])
        for i in range(len(grid_points)):
            for j in range(len(grid_points[i])):
                norm_factor = self.normalization_factor / 4 if i == 0 and j == 0 else self.normalization_factor / 2 if i == 0 or j == 0 else self.normalization_factor
                coefficient_grid[i, j] = norm_factor * integrate.dblquad(func, self.x_start, self.x_start + self.x_length, lambda x: self.y_start, lambda x: self.y_start + self.y_length, args=(i, j))[0]
        return coefficient_grid

    def calculate_alpha_coefficients(self, n, m):
        return self._calculate_coefficient(self._alpha_func, n, m)

    def calculate_beta_coefficients(self, n, m):
        return self._calculate_coefficient(self._beta_func, n, m)

    def calculate_gamma_coefficients(self, n, m):
        return self._calculate_coefficient(self._gamma_func, n, m)

    def calculate_delta_coefficients(self, n, m):
        return self._calculate_coefficient(self._delta_func, n, m)

    def get_all_coefficients(self):
        alpha = self.calculate_alpha_coefficients(self.fourier_level, self.fourier_level)
        beta = self.calculate_beta_coefficients(self.fourier_level, self.fourier_level)
        gamma = self.calculate_gamma_coefficients(self.fourier_level, self.fourier_level)
        delta = self.calculate_delta_coefficients(self.fourier_level, self.fourier_level)
        return np.array([alpha, beta, gamma, delta])

class TwoDimensionalFourierSeries:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.n_m_grid = None

    def calculate_point(self, x, y, x_length, y_length):
        if self.n_m_grid is None:
            self.n_m_grid = np.mgrid[0:self.coefficients.shape[1], 0:self.coefficients.shape[2]]
        n_m_grid = self.n_m_grid
        cos_x = np.cos((2 * np.pi * x / x_length) * (n_m_grid[0]))
        cos_y = np.cos((2 * np.pi * y / y_length) * (n_m_grid[1]))
        sin_x = np.sin((2 * np.pi * x / x_length) * (n_m_grid[0]))
        sin_y = np.sin((2 * np.pi * y / y_length) * (n_m_grid[1]))
        return np.sum(self.coefficients[3] * sin_x * sin_y + self.coefficients[0] * cos_x * cos_y + 
                      self.coefficients[2] * sin_x * cos_y + self.coefficients[1] * cos_x * sin_y)

    def get_temperature_distribution(self, x_start, x_stop, y_start, y_stop, x_step=0.01, y_step=0.01):
        x_y_grid = np.transpose(np.meshgrid(np.arange(x_start, x_stop, x_step), np.arange(y_start, y_stop, y_step)))
        x_length, y_length = x_stop - x_start, y_stop - y_start
        return [[(point[0], point[1], self.calculate_point(point[0], point[1], x_length, y_length)) for point in row] for row in x_y_grid]

class HeatEquationSolver:
    def __init__(self, initial_state_function, x_start, x_stop, y_start, y_stop, fourier_level):
        self.coeff_calculator = FourierCoefficientsCalculator(initial_state_function, x_start, x_stop, y_start, y_stop, fourier_level)
        self.coefficients = self.coeff_calculator.get_all_coefficients()
        self.x_start, self.x_stop = x_start, x_stop
        self.y_start, self.y_stop = y_start, y_stop

    @staticmethod
    def calculate_reduction_factor(diffusion_coeff, n, m, x_length, y_length):
        reduction_x = (-4 * (np.pi**2) * diffusion_coeff * (n**2)) / (x_length**2)
        reduction_y = (-4 * (np.pi**2) * diffusion_coeff * (m**2)) / (y_length**2)
        return (reduction_x, reduction_y)

    def calculate_temperature_change_rate(self):
        change_rate = self.coefficients.copy()
        for coeff_set in range(len(change_rate)):
            for i in range(len(change_rate[coeff_set])):
                for j in range(len(change_rate[coeff_set, i])):
                    change_rate[coeff_set, i, j] = np.sum(self.calculate_reduction_factor(
                        change_rate[coeff_set, i, j], i, j, self.coeff_calculator.x_length, self.coeff_calculator.y_length))
        return change_rate

    def calculate_coefficient_change(self, time_step, diffusion_coeff):
        return self.calculate_temperature_change_rate() * time_step * diffusion_coeff

    def simulate_heat_equation(self, total_time, time_step, diffusion_coeff=1):
        temperature_distributions = []
        for _ in np.arange(0, total_time, time_step):
            temperature_distributions.append(TwoDimensionalFourierSeries(self.coefficients))
            self.coefficients = self.coefficients + self.calculate_coefficient_change(time_step, diffusion_coeff)
        return temperature_distributions

def main():
    args = parse_arguments()

    # Set up the simulation
    heat_solver = HeatEquationSolver(initial_temperature_distribution, args.x_start, args.x_stop, args.y_start, args.y_stop, args.fourier_level)
    temperature_distributions = heat_solver.simulate_heat_equation(args.total_time, args.time_step, args.diffusion_coeff)

    # Generate values
    values = [np.transpose(dist.get_temperature_distribution(args.x_start, args.x_stop, args.y_start, args.y_stop, args.spatial_step, args.spatial_step)) 
              for dist in tqdm(temperature_distributions, desc="Generating temperature distributions")]
    values = np.array(values)
    temperature_array = values[:, 2].T
    x, y = values[0, 0], values[0, 1]

    # Create the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_plot(frame_number, temperature_array, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(x, y, temperature_array[:,:,frame_number], cmap="seismic")
        ax.view_init(elev=50, azim=frame_number * 0.25)

    plot = [ax.plot_surface(x, y, temperature_array[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(0, 1.1)
    ani = animation.FuncAnimation(fig, update_plot, len(temperature_array[0,0]), fargs=(temperature_array, plot), interval=1000/args.fps)

    print("Saving animation...")
    ani.save(f"{args.output_filename}.mp4", writer='ffmpeg', fps=args.fps)
    print("Animation saved successfully.")

if __name__ == "__main__":
    main()