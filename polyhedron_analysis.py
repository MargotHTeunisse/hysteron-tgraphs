from model import *
import cdd
from scipy import spatial

""""This module makes use of the cdd package and scipy.spatial to construct and analyse polyhedron.
Currently this is only used to calculate the volume of a polyhedron."""

def calculate_polyhedron_volume(switching_field_order:SwitchingFieldOrder) -> float:
    #Get the coefficients corresponding to the design inequalities
    coeffs = switching_field_order.to_coeffs()
    coeffs = np.insert(coeffs, 0, 0, axis=1)

    #Add ineqs to constrain the parameter space between 0 and 1.
    #This can be done w/o loss of generality, since only the order of switching fields matters.
    upper_limit = np.insert(-np.identity(coeffs.shape[1]-1), 0, 1, axis=1)
    lower_limit = np.insert(np.identity(coeffs.shape[1]-1), 0, 0, axis=1)
    coeffs= np.concatenate([coeffs, upper_limit, lower_limit], axis=0)

    #Use the pycddlib package to generate a polyhedron; taken from Keim and Paulsen, SciAdv (2021)
    polyhedron = cdd.Polyhedron(cdd.Matrix(coeffs))
    
    #Find the vertices of the polyhedron
    points = np.array([list(row) for row in polyhedron.get_generators()])[:, 1:]

    #Calculate the volume of the polyhedron using Scipy
    hull = spatial.ConvexHull(points)
    return hull.volume