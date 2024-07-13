# Heat Equation Model in Python

`main.py` contains code for a 2D simulation of the heat equation in Python. The simulation now supports rendering with either Matplotlib or PyVista. Example renderings can be seen in the examples folder.

## Installation

To get started with installation, run:

```
pip install -r requirements.txt
```

Note that this will install both Matplotlib and PyVista, allowing you to use either renderer.

Important: The `ffmpeg` system package is also required. Install it using your system's package manager:

- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS with Homebrew: `brew install ffmpeg`
- Windows: Download from the official FFmpeg website or use a package manager like Chocolatey.

Alternatively, you can use conda/mamba to install the same requirements.

## Usage

The simulation accepts the following command line arguments:

### Function plotting
- `--fourier_level`: The the Fourier series approximation level (default: 5)
- `--x_start`: Start value for x (default: -2)
- `--x_stop`: Stop value for x (default: 2.0001)
- `--y_start`: Start value for y (default: -2)
- `--y_stop`: Stop value for y (default: 2.0001)
- `--spatial_step`: Step size for x and y (default: 0.05)

### Time
- `--total_time`: Total simulation time (default: 0.2)
- `--time_step`: Time step for each frame (default: 0.003)

### Temperature
- `--diffusion_coeff`: Diffusion coefficient (thermal diffusivity) of the material (default: 0.2)

### Saving
- `--output_filename`: Output filename for the animation (default: "heat_equation_sim")
- `--fps`: Frames per second of the animation (default: 30)


### Renderer
- `--renderer`: Choose the rendering library ('matplotlib' or 'pyvista', default: 'matplotlib')

Renderer Comparison:
- Matplotlib:
  - Slower rendering
  - Includes axes in the visualization

- PyVista:
  - Much faster rendering
  - Does not show axes

### Initial Heat Distribution

The initial temperature distribution is determined by the `initial_temperature_distribution(x, y)` function. To modify the initial state, edit this function in the script (main.py line 44):

```python
def initial_temperature_distribution(x, y):
    return np.sin(x + y)
```

## How it Works

This program first performs a 2D Fourier approximation of the given 2D equation. This approach is used because the heat equation involves taking the second derivative of the function and subtracting it from the function itself.

With a normal function, this would result in a long chain of subtracted derivatives. However, the second derivative of a Fourier coefficient is a multiple of itself (by a decimal factor), which makes it much easier to compute.

After the approximation is calculated at the given Fourier level, we store how the function would morph with the diffusion of heat over time (in the `TwoDimensionalFourierSeries` class) and calculate it for exact points last.

The simulation uses either Matplotlib or PyVista (based on user choice) to render the results, providing a 3D visualization of how the temperature distribution evolves over time.

For more details on the mathematical background, refer to [this Stack Exchange post](https://math.stackexchange.com/questions/211689/real-valued-2d-fourier-series).
