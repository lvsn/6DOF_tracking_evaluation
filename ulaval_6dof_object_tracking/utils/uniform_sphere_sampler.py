import numpy as np
import random
import math

from ulaval_6dof_object_tracking.utils.transform import Transform
from ulaval_6dof_object_tracking.utils.angles import angle_axis2euler


class UniformSphereSampler:
    def __init__(self, min_radius=0.4, max_radius=2):
        self.up = np.array([0, 0, 1])
        self.min_radius = min_radius
        self.max_radius = max_radius

    @staticmethod
    def sph2cart(phi, theta, r):
        points = np.zeros(3)
        points[0] = r * math.sin(phi) * math.cos(theta)
        points[1] = r * math.sin(phi) * math.sin(theta)
        points[2] = r * math.cos(phi)
        return points

    @staticmethod
    def random_direction():
        # Random pose on a sphere : https://www.jasondavies.com/maps/random-points/
        theta = random.uniform(0, 1) * math.pi * 2
        phi = math.acos((2 * (random.uniform(0, 1))) - 1)
        return UniformSphereSampler.sph2cart(phi, theta, 1)

    def get_random(self):
        eye = UniformSphereSampler.random_direction()

        distance = random.uniform(0, 1) * (self.max_radius - self.min_radius) + self.min_radius
        eye *= distance
        view = Transform.lookAt(eye, np.zeros(3), self.up)

        # Random z rotation
        angle = random.uniform(0, 1) * math.pi * 2
        cosa = math.cos(angle)
        sina = math.sin(angle)
        rotation = Transform()
        rotation.matrix[0, 0] = cosa
        rotation.matrix[1, 0] = -sina
        rotation.matrix[0, 1] = sina
        rotation.matrix[1, 1] = cosa
        ret = view.transpose()
        ret.rotate(transform=rotation.transpose())
        return ret.transpose()

    @staticmethod
    def random_normal_magnitude(max_T, max_R):
        #maximum_magnitude_T = np.sqrt(3*(max_T**2))
        #maximum_magnitude_R = np.sqrt(3*(max_R**2))

        direction_T = UniformSphereSampler.random_direction()
        magn_T = UniformSphereSampler.normal_clip(max_T)
        #magn_T = random.uniform(0, max_T)
        T = angle_axis2euler(magn_T, direction_T)
        direction_R = UniformSphereSampler.random_direction()
        magn_R = UniformSphereSampler.normal_clip(max_R)
        #magn_R = random.uniform(0, max_R)

        R = angle_axis2euler(magn_R, direction_R)
        return Transform.from_parameters(T[2], T[1], T[0], R[2], R[1], R[0])

    @staticmethod
    def normal_clip(max):
        value = max
        while value >= max or value <= -max:
            value = np.clip(np.random.normal(0, max / 2), -max, max)
        return value

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_random()
