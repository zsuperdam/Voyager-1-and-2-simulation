# Voyager Probe Trajectory Simulator

This program simulates the trajectory of Voyager 1 and 2 probes, comparing them with their real trajectories. It utilizes two methods to integrate the equations of motion: `solveRungeKutta`, which implements the 4th order Runge-Kutta method, and `odeint`, which employs the same method but with 5 points for increased precision.

## Program Description

The program is structured into the following sections:

- `Planet`: a class containing all bodies to be simulated.
- `StellarSystem`: a class containing the star (Sun) unaffected by the gravity of other bodies and around which all other bodies orbit.
Within the class, the following functions are defined:
- `solveRungeKutta`: contains the implementation of the 4th order Runge-Kutta method.
- `solve`: provides an implementation of the 5th order Runge-Kutta method using SciPy's `odeint` function.
- `animation`: handles the creation of graphs and visualizations of trajectories using Matplotlib's `animation` function.
- `distanceovertime`: a function that creates a graph of distance over time between the simulated and real probes.

## Dependencies

The program requires the installation of the following dependencies:

- [NumPy](https://numpy.org/): for efficient array manipulation and numerical computations.
- [Matplotlib](https://matplotlib.org/): for creating graphs and visualizations.
- [SciPy](https://www.scipy.org/): for using the `odeint` function.
- [SpiceyPy](https://spiceypy.readthedocs.io/en/stable/): for obtaining the initial states of Voyager probes using the SPICE toolkit.

## Usage

To use the program, follow these steps:

1. Ensure all dependencies are correctly installed.
2. Run the `simulation.py` file using Python: `python simulation.py`.
3. The program will compute the simulated trajectories of Voyager 1 and 2 probes and display them alongside their real trajectories.

## Configuration

In the `simulation.py` file, you can configure the following parameters:

- Specify the integration method to use (`'runge kutta 4'` or `'odeint'`).
- `step_size`: Specify the step size for numerical integration.

## Sources and Resources

For further information about Voyager probes and their trajectories, refer to the following resources:

- [NASA Voyager Mission](https://voyager.jpl.nasa.gov/)
- [SPICE Toolkit](https://naif.jpl.nasa.gov/naif/toolkit.html)

All necessary links for further information are available in the comments of the `simulation.py` file.

## Contributions

Special thanks to [AleAntonini](https://github.com/AleAntonini) for assistance in resolving some issues encountered during development.

## License

This program is released under the MIT license. See the `LICENSE` file for more information.

Note: The necessary kernels to run the program are not provided. They can be downloaded from the link https://naif.jpl.nasa.gov/pub/naif/. Further information is available in comments within the program.
