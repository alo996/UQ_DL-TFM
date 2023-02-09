import numpy as np
import pandas as pd
from tqdm import tqdm
import tables
from scipy.special import ellipk, ellipe
import multiprocessing as mp
import h5py
import os

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def inCircle(X, Y, x, y, r):
    return (X - x) ** 2 + (Y - y) ** 2 <= r ** 2


class EventGenerator:
    def __init__(self, params):
        self.params = params
        self.displacement_mesh = self.generate_mesh(0, 1, 0, 1, params['resolutionX'])
        self.force_mesh = self.generate_mesh(0, 1, 0, 1, params['resolutionY'])
        self.E = 1000
        self.f_res = 1 / (self.params['resolutionY'] - 1)
        self.num_generator = params['num_generator']

    def generate_mesh(self, left, right, front, back, resolution):
        X, Y = np.meshgrid(np.linspace(left, right, resolution), np.linspace(front, back, resolution))
        mesh = np.array([[(X[i][j], Y[i][j]) for j in range(len(X[i]))] for i in range(len(X))])
        return mesh

    def write_PointForces(self, PointForces):
        self.data_forces.append(np.array([PointForces]))

    def write_Displacement(self, displacement):
        self.data_disp.append(np.array([displacement]))

    def min_dist(self, PF, point, R):
        return np.min(np.sqrt((PF.x_coord - point[0]) ** 2 + (PF.y_coord - point[1]) ** 2) - 2 * R)

    def generate_PointForces(self):
        counter = 0
        PointForceSegmentation = np.zeros((self.params['resolutionY'], self.params['resolutionY']))
        PointForceMesh = np.zeros((self.params['resolutionY'],self.params['resolutionY'], 2))
        PointForces = pd.DataFrame({"x_coord": [], "y_coord": [], "force": [], "gamma": [], "radius": []})
        while counter < np.random.uniform(10, 50):
            R = np.random.uniform(0.01, 0.05)
            point = np.random.uniform(0 + R + 0.05, 1 - R - 0.05, 2)
            force = np.random.uniform(self.params['traction_min'], self.params['traction_max'])
            force = force / self.E
            gamma = np.random.uniform(0, 2 * np.pi)
            if counter == 0 or self.min_dist(PointForces, point, R) > 0.001:
                force_hook = pd.DataFrame.from_records([{"x_coord": point[0],
                                                         "y_coord": point[1],
                                                         "force": force,
                                                         "gamma": gamma,
                                                         "radius": R}])
                PointForces = pd.concat([PointForces, force_hook])
                x_f, y_f = pol2cart(force, gamma)
                PointForceSeg_temp = np.where(inCircle(self.force_mesh[:, :, 0], self.force_mesh[:, :, 1], point[0], point[1], R), counter + 1, 0)
                PointForceMesh[
                    inCircle(self.force_mesh[:, :, 0], self.force_mesh[:, :, 1], point[0], point[1], R)] += np.array(
                    [x_f, y_f])
                PointForceSegmentation += PointForceSeg_temp
                counter += 1

        return PointForceSegmentation, PointForceMesh, PointForces

    def generate_displacement(self, PointForces):
        raise NotImplementedError

    def generate(self, event_num):
        atom = tables.Float64Atom()
        f_data_disp = tables.open_file(
            f'Test data/resolution_{self.params["resolutionX"]}/dspl_{self.num_generator}.h5', mode='w')
        self.data_disp = f_data_disp.create_earray(f_data_disp.root, 'data', atom,
                                                   (0, self.params['resolutionX'], self.params['resolutionX'], 2))
        f_data_forces = tables.open_file(
            f'Test data/resolution_{self.params["resolutionX"]}/trac_{self.num_generator}.h5', mode='w')
        self.data_forces = f_data_forces.create_earray(f_data_forces.root, 'data', atom,
                                                                           (0, self.params['resolutionY'],
                                                                            self.params['resolutionY'], 3))

        for i in tqdm(range(event_num)):
            PointForceSegmentation, PointForceMesh, PointForces = self.generate_PointForces()
            PointForceInfo = np.dstack((PointForceMesh, PointForceSegmentation))
            displacement = self.generate_displacement(PointForces)
            self.write_PointForces(PointForceInfo)
            self.write_Displacement(displacement)
        f_data_disp.close()
        f_data_forces.close()

class AnalyticalEventGenerator(EventGenerator):

    def analytical(self, point, traction, R):
        p0 = traction[0]
        gamma = traction[1]
        r, theta = cart2pol(point[0], point[1])
        if r < R:
            if r < 1e-4:
                N1 = 2 * np.pi
                N2 = np.pi
                N3 = 0
                N4 = np.pi
            zeta1 = r ** 2 / R ** 2
            E0 = ellipe(zeta1)
            K0 = ellipk(zeta1)
            N1 = 4 * E0
            N2 = (4 * np.cos(2 * theta) * ((r ** 2 + R ** 2) * E0 + (r ** 2 - R ** 2) * K0)) / (
                    3 * r ** 2) + 4 * np.sin(theta) ** 2 * E0
            N3 = (2 * np.sin(2 * theta) * ((r ** 2 - 2 * R ** 2) * E0 + 2 * (R ** 2 - r ** 2) * K0)) / (3 * r ** 2)
            N4 = 4 * np.cos(theta) ** 2 * E0 - (
                    4 * np.cos(2 * theta) * ((r ** 2 + R ** 2) * E0 + (r ** 2 - R ** 2) * K0)) / (3 * r ** 2)
        else:
            zeta2 = R ** 2 / r ** 2
            E0 = ellipe(zeta2)
            K0 = ellipk(zeta2)
            N1 = (4 * (r ** 2 * E0 + (R ** 2 - r ** 2) * K0)) / (r * R)
            N2 = ((6 * r ** 2 - 2 * (r ** 2 - 2 * R ** 2) * np.cos(2 * theta)) * E0 + 2 * (r ** 2 - R ** 2) * (
                    np.cos(2 * theta) - 3) * K0) / (3 * r * R)
            N3 = (2 * np.sin(2 * theta) * ((r ** 2 - 2 * R ** 2) * E0 + (R ** 2 - r ** 2) * K0)) / (3 * r * R)
            N4 = ((6 * r ** 2 + 2 * (r ** 2 - 2 * R ** 2) * np.cos(2 * theta)) * E0 - 2 * (r ** 2 - R ** 2) * (
                    np.cos(2 * theta) + 3) * K0) / (3 * r * R)
        ux = R * (1 + self.params['nu']) / (np.pi) * (
                ((1 - self.params['nu']) * N1 + self.params['nu'] * N2) * p0 * np.cos(gamma) - self.params[
            'nu'] * N3 * p0 * np.sin(gamma))
        uy = R * (1 + self.params['nu']) / (np.pi) * (-self.params['nu'] * N3 * p0 * np.cos(gamma) + (
                (1 - self.params['nu']) * N1 + self.params['nu'] * N4) * p0 * np.sin(gamma))
        return ux, uy

    def generate_displacement(self, PointForces):
        displacement = np.zeros((len(self.displacement_mesh), len(self.displacement_mesh[0]), 2))
        for index, row in PointForces.iterrows():
            trafo = np.array([-row.x_coord, -row.y_coord])
            force = np.array([row.force, row.gamma])
            displacement += np.array(
                [
                    [self.analytical(self.displacement_mesh[i][j] + trafo, force, row.radius)
                     if self.analytical(self.displacement_mesh[i][j] + trafo, force, row.radius) is not np.nan
                     else self.analytical(self.displacement_mesh[i][(j - 1) % len(self.displacement_mesh[i])] + trafo,
                                          force, row.radius)
                     for j in range(len(self.displacement_mesh[i]))
                     ]
                    for i in range(len(self.displacement_mesh))
                ])
        return displacement


def start_data_generator(resolution, samples_per_generator, num_generator, use):
    print(f"Process {num_generator} has started")
    if use == 'training':
        np.random.seed(num_generator)
        Gen = AnalyticalEventGenerator({'resolutionX': resolution,
                                        'resolutionY': resolution,
                                        'traction_min': 0,
                                        'traction_max': 500,
                                        'nu': 0.49,
                                        'num_generator': num_generator})
        Gen.generate(samples_per_generator)
    elif use == 'validation':
        np.random.seed(100 + num_generator)
        Gen = AnalyticalEventGenerator({'resolutionX': resolution,
                                        'resolutionY': resolution,
                                        'traction_min': 0,
                                        'traction_max': 500,
                                        'nu': 0.49,
                                        'num_generator': num_generator})
        Gen.generate(samples_per_generator)
    elif use == 'test':
        np.random.seed(200 + num_generator)
        Gen = AnalyticalEventGenerator({'resolutionX': resolution,
                                        'resolutionY': resolution,
                                        'traction_min': 0,
                                        'traction_max': 500,
                                        'nu': 0.49,
                                        'num_generator': num_generator})
        Gen.generate(samples_per_generator)


def aggregate_dspl_hdf5_files(resolution):
    with h5py.File(f"Test data/resolution_{resolution}/allDisplacements.h5", "w") as f_dst:
        h5files = [f for f in os.listdir(f'Test data/resolution_{resolution}') if f.startswith("dspl")]
        h5files.sort()

        dset = f_dst.create_dataset("dspl", shape=(20, 50, resolution, resolution, 2), dtype='f8')

        for i, filename in enumerate(h5files):
            print(filename)
            with h5py.File(f'Test data/resolution_{resolution}/{filename}') as f_src:
                dset[i] = f_src["data"]


def aggregate_trac_hdf5_files(resolution):
    with h5py.File(f"Test data/resolution_{resolution}/allTractions.h5", "w") as f_dst:
        h5files = [f for f in os.listdir(f'Test data/resolution_{resolution}') if f.startswith("trac")]
        h5files.sort()

        dset = f_dst.create_dataset("trac", shape=(20, 50, resolution, resolution, 3), dtype='f8')

        for i, filename in enumerate(h5files):
            print(filename)
            with h5py.File(f'Test data/resolution_{resolution}/{filename}') as f_src:
                dset[i] = f_src["data"]


def generate_data_in_parallel(resolution, samples, num_processes, use):
    pool = mp.Pool(processes=num_processes)
    samples_per_generator = samples // num_processes
    for i in range(1, num_processes + 1):
        pool.apply_async(start_data_generator, args=(resolution, samples_per_generator, i, use))
    pool.close()
    pool.join()


#generate_data_in_parallel(resolution=104, samples=1000,  num_processes=20, use='test')
aggregate_dspl_hdf5_files(resolution=104)
aggregate_trac_hdf5_files(resolution=104)
