"""
Made by Damiano Cerioni (damiano.cerioni@studenti.unipd.it)

================================SPICEYPY================================
Annex et al., (2020). SpiceyPy: a Pythonic Wrapper for the SPICE Toolkit
Journal of Open Source Software, 5(46), 2050,
https://doi.org/10.21105/joss.02050
========================================================================

Useful links for kernels:
NASA Naif ID https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/naif_ids.html
kernel info https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html
kernels https://naif.jpl.nasa.gov/pub/naif/generic_kernels/


naif body barycenter id:

        1  Mercury barycenter
        2  Venus barycenter
        3  Earth barycenter
        4  Mars barycenter
        5  Jupiter barycenter
        6  Saturn barycenter
        7  Uranus barycenter
        8  Neptune barycenter
        9  Pluto barycenter
        10 Sun
        -31 Voyager 1 -> launch: September 5, 1977
        -32 Voyager 2 -> launch: August 20, 1977

        Summary for: Voyager_1.a54206u_V0.2_merged.bsp
        -31 VOYAGER 1 w.r.t. 10 SUN     1977 SEP 08 09:08:16.593     1979 JAN 14 15:51:03.735
        Summary for: Voyager_2.m05016u.merged.bsp
        -32 VOYAGER 2 w.r.t. 10 SUN     1977 AUG 23 11:29:10.841     1979 MAY 03 21:42:56.180
        https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/aa_summaries.txt

        masses taken from https://en.wikipedia.org/wiki/
"""

import numpy as np
import spiceypy as spice  # documentation -> https://spiceypy.readthedocs.io/en/stable/
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # CSS4 color list -> https://matplotlib.org/stable/gallery/color/named_colors.html
import datetime
import matplotlib.animation as animation
import time


# constants
year = 365*24*3600   # s in a year  [s]
AU = 1.49597871e11   # AU in m      [m]
G = 6.67430e-11      # universal gravitational constant [N * m^2 / kg^2]


# spiceypy kernel load (same folder as SistemaSolare.py)
spice.furnsh("gm_de431.tpc")    # position and speed
spice.furnsh("naif0012.tls")    # compute the time difference after 2017 (avoid 1s error)
spice.furnsh("de432s.bsp")      # solar system planets data
spice.furnsh("pck00010.tpc")    # Planet orientation and radii
spice.furnsh("voyager_1.ST+1991_a54418u.merged.bsp")  # Voyager 1
spice.furnsh("voyager_2.ST+1992_m05208u.merged.bsp")  # Voygaer 2
spice.furnsh("jup365.bsp")  # jupiter satellites
spice.furnsh("sat441.bsp")  # saturn satellites


"""
Summary for: de432s.bsp
 
Bodies: MERCURY BARYCENTER (1)  SATURN BARYCENTER (6)   MERCURY (199)
        VENUS BARYCENTER (2)    URANUS BARYCENTER (7)   VENUS (299)
        EARTH BARYCENTER (3)    NEPTUNE BARYCENTER (8)  MOON (301)
        MARS BARYCENTER (4)     PLUTO BARYCENTER (9)    EARTH (399)
        JUPITER BARYCENTER (5)  SUN (10)
        Start of Interval (ET)              End of Interval (ET)
        -----------------------------       -----------------------------
        1949 DEC 14 00:00:00.000            2050 JAN 02 00:00:00.000
"""


# date
start_datetime = datetime.datetime(day=10, month=9, year=1977)
end_datetime = datetime.datetime(day=1, month=1, year=1985)
sim_duration = (end_datetime - start_datetime).days * 24 * 3600

dt = 86400                                   # dt in seconds
n_dt = int(round(sim_duration / dt))         # number of dt
t = np.linspace(0, sim_duration, n_dt)  # array of spaced t for odeint

# date format for spiceypy
start_date_text = spice.utc2et(start_datetime.strftime("%Y-%m-%dT00:00:00"))  # simulation starts and ends at 00.00.000
end_date= spice.utc2et(end_datetime.strftime("%Y-%m-%dT00:00:00"))      # simulation starts and ends at 00.00.000

class Planet:

    def __init__(self, nasa_naif_id, name, mass, color, star):
        self.name = name
        self.nasa_naif_id = nasa_naif_id  # id are barycenter of planets
        self.mass = mass
        self.color = color
        self.star = star

        self.size = 5  # max(minimum_size, np.log10(self.mass))

        # spice.spkgeo(targ, et, ref, obs
        # VARIABLE  I/O DESCRIPTION             COMMENT
        # targ      I   Target body
        # et        I   Target epoch            start_date
        # ref       I   Target reference frame
        # obs       I   Observing body          Sun (nasa_naif_id = 10)
        # state     O   State of target         [x, y, z, vx, vy, vz]
        # lt        O   Light time [s]          one-way light time targ-obs
        self.state, self.r_sun = spice.spkgeo(self.nasa_naif_id, start_date_text,  "ECLIPB1950", 10)
        self.state = self.state * 1000  # [km] -> [m]
        self.sol = []
        self.star.add_planet(self)

    def realdata(self):
        datedifference = end_datetime - start_datetime
        sol = np.array([])
        #sol = np.append(sol, self.state / AU)
        for i in range(datedifference.days):
            time = start_datetime + datetime.timedelta(days=i)
            time = spice.utc2et(time.strftime("%Y-%m-%dT00:00:00"))
            newstate, r = spice.spkgeo(-31, time, "ECLIPB1950", 10)
            sol = np.append(sol, newstate)
        sol = sol.reshape(-1, 6)
        self.sol = sol * 1000 / AU  # km -> AU


class StellarSystem:

    def __init__(self, nasa_naif_id, star_name, star_mass, color):
        self.nasa_naif_id = nasa_naif_id
        self.star_name = star_name
        self.star_mass = star_mass
        self.planets = []
        self.color = color

    def add_planet(self, planet):
        self.planets.append(planet)

    def get_state(self):
        state = []
        for planet in self.planets:
            state.extend(planet.state)
        return state

    def deriv(self, state, t):
        dsdt = np.zeros(6 * len(self.planets), dtype=np.float64)  # vx, vy, vz, ax, ay, az from innermost to outermost
        for i, planet in enumerate(self.planets):
            dsdt[6*i:6*i+3] = state[6*i+3:6*i+6]
            r = np.sqrt(state[6*i] ** 2 + state[6*i+1] ** 2 + state[6*i+2] ** 2)  # distance from sun
            acceleration = - G * self.star_mass * np.array(state[6*i:6*i+3]) / (r ** 3)  # Force caused by sun
            for j, planet2 in enumerate(self.planets):
                if j != i:
                    r = np.sqrt((state[6*j] - state[6*i]) ** 2 + (state[6*j+1] - state[6*i+1]) ** 2 + (state[6*j+2] - state[6*i+2]) ** 2)  # distance ri - rj
                    acceleration += G * planet2.mass * (np.array(state[6*j:6*j+3]) - np.array(state[6*i:6*i+3])) / (r ** 3)
            dsdt[6*i+3:6*i+6] = acceleration[:]
        return dsdt

    def solve(self):
        state = self.get_state()
        self.sol = odeint(self.deriv, state, t) / AU
        for i, planet in enumerate(self.planets):
            planet.sol = [list(row[6*i:6*i+6]) for row in self.sol]

    def rungekutta(self, f, y0, dt):
        Y1 = y0
        Y2 = y0 + f(Y1, dt) * dt / 2.0      # df/dt = cost
        Y3 = y0 + f(Y2, dt) * dt / 2.0
        Y4 = y0 + f(Y3, dt) * dt
        y1 = y0 + (f(Y1, dt) + 2.0 * f(Y2, dt) + 2.0 * f(Y3, dt) + f(Y4, dt)) * dt / 6.0
        return y1

    def solveRungeKutta(self):
        state = self.get_state()
        self.sol = []
        self.sol.append(state)
        for _ in t:
            self.sol.append(self.rungekutta(f=self.deriv, y0=self.sol[-1], dt=dt))
        self.sol = np.divide(self.sol, AU)
        self.sol = self.sol[1:]
        for i, planet in enumerate(self.planets):
            planet.sol = [list(row[6*i:6*i+6]) for row in self.sol]

    def speeddifference(self):
        start_time = time.time()
        Sun.solve()
        odeinttime = time.time() - start_time
        print("odeint: %s seconds" % odeinttime)
        start_time = time.time()
        Sun.solveRungeKutta()
        rungekuttatime = time.time() - start_time
        print("RungeKutta: %s seconds" % rungekuttatime)
        print("odeint is %s times faster than rungekutta" % (rungekuttatime / odeinttime))

    def voyagerspeed(self):
        v = []
        d = []
        for i in range(len(t)):
            v0 = np.sqrt(Voyager1.sol[i][3] ** 2 + Voyager1.sol[i][4] ** 2 + Voyager1.sol[i][5] ** 2)
            v.append(v0)
            d0 = np.sqrt((Voyager1.sol[i][0]-Jupiter.sol[i][0]) ** 2 + (Voyager1.sol[i][1]-Jupiter.sol[i][1]) ** 2 + (Voyager1.sol[i][2]-Jupiter.sol[i][2]) ** 2)
            d.append(d0)

        plt.plot(t,v)
        plt.xlabel('time')
        plt.ylabel('speed')
        plt.show()

        plt.plot(t,d)
        plt.xlabel('time')
        plt.ylabel('distance')
        plt.show()


    def animation_full(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # axes name
        ax.set_xlabel('X (UA)')
        ax.set_ylabel('Y (UA)')
        ax.set_zlabel('Z (UA)')

        #axes limit
        radius = 1.5  # half graph side in AU
        ax.set_xlim((-radius, radius))
        ax.set_ylim((-radius, radius))
        ax.set_zlim((-radius, radius))

        ax.plot(0,0,0, marker="o", markersize=5, label="Sun", color=Sun.color)

        self.draw_planets = []
        for i in self.planets:
            i, = ax.plot(i.state[0]/AU, i.state[1]/AU, i.state[2]/AU, marker="o", markersize=i.size, label=i.name, color=i.color)
            self.draw_planets.append(i)

        try:
            trace1x = np.array([Voyager1.state[0]/AU])
            trace1y = np.array([Voyager1.state[1]/AU])
            trace1z = np.array([Voyager1.state[2]/AU])

            trace1, = ax.plot(trace1x, trace1y, trace1z, color=Voyager1.color)
        except NameError:
            pass

        try:
            trace2x = np.array([Voygaer2.state[0] / AU])
            trace2y = np.array([Voygaer2.state[1] / AU])
            trace2z = np.array([Voygaer2.state[2] / AU])

            trace2, = ax.plot(trace2x, trace2y, trace2z, color=Voygaer2.color)
        except NameError:
            pass


        self.timer = ax.text2D(0, 0, "", ha="left", va="bottom", transform=ax.transAxes)
        self.velocity = ax.text2D(0, 0, "", ha="right", va="bottom", transform=ax.transAxes)

        try:
            tracex = np.empty(len(self.planets), tracelen)
            tracey = np.empty(len(self.planets), tracelen)
            tracez = np.empty(len(self.planets), tracelen)
            self.draw_trace = []
            for i, planet in enumerate(self.planets):
                i, = ax.plot(tracex[i], tracey[i], tracez[i], color=planet.color)
                self.draw_trace.append(i)
        except NameError:
            pass

        # legend
        ax.legend()


        def update(frame):
            try:
                nonlocal trace1x, trace1y, trace1z
            except NameError:
                pass

            try:
                nonlocal trace2x, trace2y, trace2z
            except NameError:
                pass

            for i, j in zip(self.planets, self.draw_planets):
                newx = i.sol[frame][0]
                newy = i.sol[frame][1]
                newz = i.sol[frame][2]
                j.set_data(newx, newy)
                j.set_3d_properties(newz)
            time = start_datetime + datetime.timedelta(days=frame)
            time_str = time.strftime("%Y-%m-%d")
            self.timer.set_text("Date: " + time_str)

            #trace
            tracelen = 300

            try:
                trace1x = np.append(trace1x, Voyager1.sol[frame][0])
                trace1y = np.append(trace1y, Voyager1.sol[frame][1])
                trace1z = np.append(trace1z, Voyager1.sol[frame][2])

                trace1.set_xdata(trace1x[-tracelen:])
                trace1.set_ydata(trace1y[-tracelen:])
                trace1.set_3d_properties(trace1z[-tracelen:])
            except NameError:
                pass

            try:
                trace2x = np.append(trace2x, Voygaer2.sol[frame][0])
                trace2y = np.append(trace2y, Voygaer2.sol[frame][1])
                trace2z = np.append(trace2z, Voygaer2.sol[frame][2])

                trace2.set_xdata(trace2x[-tracelen:])
                trace2.set_ydata(trace2y[-tracelen:])
                trace2.set_3d_properties(trace2z[-tracelen:])
            except NameError:
                pass


        ani = animation.FuncAnimation(fig=fig, func=update, frames=n_dt-1, interval=0.0001)
        plt.show()

def distanceovertime(a, b):
    distance = np.array([])
    for i in range(len(a.sol)):
        d = np.sqrt((a.sol[i][0] - b.sol[i][0]) ** 2 + (a.sol[i][1] - b.sol[i][1]) ** 2 + (a.sol[i][2] - b.sol[i][2]) ** 2)
        distance = np.append(distance, d)
    #distance = distance.reshape(-1, 1)
    plt.plot(t, distance)
    plt.xlabel('time')
    plt.ylabel('distance')
    plt.title('Distance {} - {}'.format(a.name, b.name))
    plt.show()

Sun = StellarSystem(10, "Sun", 1.98847e30, mcolors.CSS4_COLORS['gold'])

#Mercury = Planet(1, "Mercury", 3.3011e23, mcolors.CSS4_COLORS['dimgray'], Sun)
#Venus = Planet(2, "Venus", 4.8675e24, mcolors.CSS4_COLORS['darkgoldenrod'], Sun)
Earth = Planet(3, "Earth", 5.9722e24, mcolors.CSS4_COLORS['dodgerblue'], Sun)
#Mars = Planet(4, "Mars", 6.4171e23, mcolors.CSS4_COLORS['orangered'], Sun)

Jupiter = Planet(5, "Jupiter", 1.8982e27, mcolors.CSS4_COLORS['goldenrod'], Sun)
#Io = Planet(501, "Io", 8.9319e22, mcolors.CSS4_COLORS['silver'], Sun)
#Callisto = Planet(504, "Callisto", 1.0759e23, mcolors.CSS4_COLORS['silver'], Sun)

#Saturn = Planet(6, "Saturn",  5.6834e26, mcolors.CSS4_COLORS['khaki'], Sun)
#Rhea = Planet(605, "Rhea", 2.3064854e21, mcolors.CSS4_COLORS['silver'], Sun)
#Titan = Planet(606, "Titan", 1.3452e23, mcolors.CSS4_COLORS['silver'], Sun)

#Uranus = Planet(7, "Uranus",  8.6810e25, mcolors.CSS4_COLORS['lightsteelblue'], Sun)
#Neptune = Planet(8, "Neptune", 1.0241e26, mcolors.CSS4_COLORS['royalblue'], Sun)

Voyager1 = Planet(-31,"Voyager1",721.9, mcolors.CSS4_COLORS['chartreuse'], Sun)
#Voygaer2 = Planet(-32,"Voygaer2",721.9, mcolors.CSS4_COLORS['fuchsia'], Sun)

#======================================================================================================================#

# Time difference to solve the system between RungeKutta and odeint
#Sun.speeddifference()

# solving the system
Sun.solve()
#Sun.solveRungeKutta()

# real probe data
Voyager1real = Planet(-31,"Voyager1real",721.9, mcolors.CSS4_COLORS['fuchsia'], Sun)
Voyager1real.realdata()

# distance
#distanceovertime(Voyager1, Voyager1real)

# animation
Sun.animation_full()