
import numpy as np
from numpy import array as nparray, sqrt as npsqrt,  dot as npdot, degrees as npdegrees, arccos as nparccos, zeros as npzeros, cross as npcross, max as npmax, min as npmin, ones as npones, append as npappend, arange as nparange, cov as npcov
from math import dist, pi, ceil, factorial
from numpy.linalg import eig as npeig, det as npdet


def volume(ms):
    """
    Returns the volume of the mesh (Technical Tips Step 3b.1)

        Parameters:
                ms (MeshSet): Meshset containing the mesh

        Returns:
                volume (float): The volume of the mesh
    """
    barycenter = ms.get_geometric_measures()['barycenter']
    vertex_matrix = ms.current_mesh().vertex_matrix()
    sum = 0
    for face in ms.current_mesh().face_matrix():
        v1 = vertex_matrix[face[0]] - barycenter
        v2 = vertex_matrix[face[1]] - barycenter
        v3 = vertex_matrix[face[2]] - barycenter

        sum += (npdot(npcross(v1,v2),v3))
    return ((1/6)*abs(sum))

def convexity(ms, original_volume):
    """
    The shape volume divided convex hull volume.
    """
    ms.generate_convex_hull()
    convex_volume = volume(ms)
    return original_volume / convex_volume


def area(ms):
    """
    Calculates the area of a mesh.

        Parameters:
                ms (MeshSet): Meshset containing the mesh

        Returns:
                histogram (float): The volume of the mesh
    """

    vertex_matrix = ms.current_mesh().vertex_matrix()
    area = 0
    for face in ms.current_mesh().face_matrix():
        distance_a = dist(vertex_matrix[face[0]],vertex_matrix[face[1]])
        distance_b = dist(vertex_matrix[face[1]],vertex_matrix[face[2]])
        distance_c = dist(vertex_matrix[face[2]],vertex_matrix[face[0]])

        #Heron’s formula
        s = (distance_a + distance_b + distance_c) / 2
        
        to_square = s*(s-distance_a)*(s-distance_b)*(s-distance_c)
        if to_square > 0 and not np.isinf(to_square): # Due to rounding errors this sometimes is slightly smaller than zero. The area is then negliable.
            try:
                area += npsqrt(s*(s-distance_a)*(s-distance_b)*(s-distance_c))
            except:
                print(to_square)
                raise ValueError
    return area

def compactness(area,volume):
    """
    Calculates the compactness of a mesh from the area and volume.

        Parameters:
                area (float): The area of the mesh
                volume (float): The volume of the mesh

        Returns:
                compactness (float): The compactness of the mesh
    """
    return area**3/max(0.00001,36*pi*volume**2)

def eccentricity(ms):
    """
    Calculates the exccentricity of a mesh. (04 MR Features Slide 17)

        Parameters:
                ms (MeshSet): Meshset containing the mesh

        Returns:
                excentricity (float): The excentricity of the mesh
    """
    A_cov = npcov(ms.current_mesh().vertex_matrix().T)  # 3x3 matrix
    eigenvalues, eigenvectors = npeig(A_cov)

    major_eigenvalue = npmax(eigenvalues)
    minor_eigenvalue = (max(0.00001,npmin(eigenvalues)))
    return major_eigenvalue / minor_eigenvalue


def rectangularity(ms, mesh_volume=None):
    bbox = ms.get_geometric_measures()['bbox']
    if mesh_volume == None:
        mesh_volume = volume(ms)
    return (mesh_volume / (max(0.0000001,bbox.dim_x()*bbox.dim_y()*bbox.dim_z())))

def diameter_2(ms):
    ms.generate_convex_hull()
    vertex_matrix = ms.current_mesh().vertex_matrix()
    max_distance = 0
    for vertex in vertex_matrix:
        for vertex2 in vertex_matrix:
            distance = dist(vertex,vertex2)

            if(distance > max_distance):
                max_distance = distance
    return max_distance


def diameter(ms):
    vertex_matrix = ms.current_mesh().vertex_matrix()
    max_distance = 0
    for vertex in vertex_matrix:
        for vertex2 in vertex_matrix:
            distance = dist(vertex,vertex2)

            if(distance > max_distance):
                max_distance = distance
    return max_distance

def smart_subsampling_3(vertex_matrix, n, seed=42):
    """
    Generator for sampling vertices, yielding in tuples of three. (Technical Tips 3c.1.1)

        Parameters:
                vertex_matrix (numpy.ndarray): the vertex matrix of the mesh to sample from.
                n (int): the amount of samples

        Returns:
                samples (tuple(vertex,vertex,vertex)): the samples
    """
    # TODO use seed
    rng = np.random.default_rng()
    k = ceil(pow(n, 1.0/3.0))
    amount_yielded = 0
    for v1 in rng.choice(vertex_matrix, k, replace=False):
        for v2 in rng.choice(vertex_matrix, k, replace=False):
            if (v1 == v2).all():
                continue
            for v3 in rng.choice(vertex_matrix, k, replace=False):
                if (v1 == v3).all() or (v2 == v3).all():
                    continue
                if amount_yielded < n:
                    yield v1, v2, v3
                    amount_yielded += 1
                else:
                    return

def smart_subsampling_4(vertex_matrix, n, seed=42):
    """
    Generator for sampling vertices, yielding in tuples of four. (Technical Tips 3c.1.1)

        Parameters:
                vertex_matrix (numpy.ndarray): the vertex matrix of the mesh to sample from.
                n (int): the amount of samples

        Returns:
                samples (tuple(vertex,vertex,vertex)): the samples
    """
    rng = np.random.default_rng(seed=seed)
    amount_yielded = 0
    k = ceil(pow(n, 1.0/4.0))
    i = 0
    for v1 in rng.choice(vertex_matrix, k, replace=False):
        for v2 in rng.choice(vertex_matrix, k, replace=False):
            if (v1 == v2).all():
                continue
            for v3 in rng.choice(vertex_matrix, k, replace=False):
                if (v1 == v3).all() or (v2 ==v3).all():
                    continue

                for v4 in rng.choice(vertex_matrix, k, replace=False):
                    if (v1 == v4).all() or (v2 == v4).all() or (v3 == v4).all():
                        continue
                    if amount_yielded < n:
                        yield v1, v2, v3, v4
                        amount_yielded += 1
                    else:
                        return

def build_histogram(array, _range, n_bins= 10 ,normalize=True):
    """
    Returns histogram of the array given (04 MR Features Slide 81)

        Parameters:
                array (np.array<>): array of the values
                _range (tuple(int,int)): The lower and upper bound
                n_bins (int): The amount of bins
                normalize (bool): whether the histogram should be area normalized

        Returns:
                histogram (np.array<int>): list of counts of values
    """
    # Defining the bin sizes
    lower = _range[0]
    higher = _range[1]
    step = (higher - lower) / n_bins
    bin_upper = list(nparange(lower + step,higher + step,step))

    bins = {}
    for bin in range(len(bin_upper)):
        bins[bin] = 0

    for value in array:
        for i in range(len(bin_upper)):
            if value <= bin_upper[i]:
                bins[i] += 1
                break

    # Area normalizing (04 MR Feature Slide 51, Tech tips 3c.1.1, 05 MR Matching Slide 30)
    if normalize:
        histogram_sum = np.sum(list(bins.values()))
        for key in bins.keys():
            bins[key] = bins[key]/histogram_sum

    return nparray(list(bins.values()))

def a3(ms, samples=10_000, seed=42):
    """
    Returns a normalized histogram of the angles between 3 random points. (04 MR Features Slide 81)

        Parameters:
                ms (MeshSet): Meshset containing the mesh
                samples (int): The amount of angles to calculate.

        Returns:
                histogram ([int]): A 10-bins normalized histogram
    """
    vertex_matrix = ms.current_mesh().vertex_matrix()
    angles = []

    angle_index = 0

    for v1, v2, v3 in smart_subsampling_3(vertex_matrix=vertex_matrix, n=samples):

        side1 = v1 - v2
        side2 = v3 - v2

        # Calculate angles
        
        cosine_angle = npdot(side1, side2) / (dist(v2,v1) * dist(v2, v3))

        if cosine_angle < -1: # Fixed a rounding error where cosine_angle = -1.0000001
            cosine_angle = -1
        if cosine_angle > 1:
            cosine_angle = 1
        try:
            angle = nparccos(cosine_angle)
        except:
            print(cosine_angle)
            raise ValueError
        angles.append(npdegrees(angle))
    bins = build_histogram(angles,(0,180), n_bins=10,normalize=True)
    return bins

def d1(ms, samples=10_000, seed=42):
    """
    Returns a normalized histogram of the distance between the barycenter and random points (04 MR Features Slide 81).

        Parameters:
                ms (MeshSet): Meshset containing the mesh
                samples (int): The amount of distances to calculate.

        Returns:
                histogram ([int]): A 10-bins normalized histogram
    """
    # Initializing variables
    rng = np.random.default_rng(seed=seed)
    vertex_matrix = ms.current_mesh().vertex_matrix()
    if samples > vertex_matrix.shape[0]:
        vertexes = vertex_matrix
    else:
        vertexes = rng.choice(vertex_matrix, samples, replace=False)
    barycenter = ms.get_geometric_measures()['barycenter']

    # Calculating distances
    distances = []
    for v1 in vertexes:
        distances.append(dist(v1, barycenter))

    bins = build_histogram(distances,(0.0,npsqrt(3)),n_bins=10,normalize=True)
    return bins

def d2(ms,samples=10_000, seed=42):
    """
    Returns a normalized histogram of the distance between two random points (04 MR Features Slide 81).

        Parameters:
                ms (MeshSet): Meshset containing the mesh
                samples (int): The amount of distances to calculate.

        Returns:
                histogram ([int]): A 10-bins normalized histogram
    """
    # Initializing variables
    rng = np.random.default_rng(seed=seed)
    vertex_matrix = ms.current_mesh().vertex_matrix()

    distances = []
    k = ceil(pow(samples, 1.0/2.0))
    i = 0
    for v1 in rng.choice(vertex_matrix, k, replace=False):
        for v2 in rng.choice(vertex_matrix, k, replace=False):
            if (v1 ==v2).all():
                continue
            else:
                distances.append(dist(v1,v2))

    bins = build_histogram(distances,(0,(np.sqrt(3))),n_bins=10,normalize=True) # Max is sqrt of 2
    return  bins

def d3(ms, samples=10_000, seed=42):
    """
    Returns a normalized histogram of the area of the triangle made by three random points (04 MR Features Slide 81).

        Parameters:
                ms (MeshSet): Meshset containing the mesh
                samples (int): The amount of areas to calculate.

        Returns:
                histogram ([int]): A 10-bins normalized histogram
    """
    # Initializing Variables
    vertex_matrix = ms.current_mesh().vertex_matrix()
    areas         = []

    for v1, v2, v3 in smart_subsampling_3(vertex_matrix=vertex_matrix, n=samples, seed=seed):

        distance_a, distance_b, distance_c = dist(v1, v2), dist(v2, v3), dist(v3, v1)

        #Heron’s formula
        s = (distance_a + distance_b + distance_c) / 2

        to_square = s*(s-distance_a)*(s-distance_b)*(s-distance_c)
        if to_square < 0:
            to_square = 0
        areas.append(npsqrt(to_square))

    bins = build_histogram(npsqrt(areas), (0,(npsqrt(npsqrt(3)/2))), n_bins=10, normalize=True) #https://www.wolframalpha.com/input?i=Abs%5BDet%5B%7B%7B0%2C+0%2C+0%2C1%7D%2C+%7B1%2C1%2C0%2C1%7D%2C+%7B0%2C1%2C0%2C+1%7D%2C+%7B1%2C+1%2C+1%2C+1%7D%7D%5D%2F3%21%5D
    return bins

def d4(ms, samples=1_000_000, seed=42):
    """
    Returns a normalized histogram of the volume of the tetrahedron made by four random points (04 MR Features Slide 81).

        Parameters:
                ms (MeshSet): Meshset containing the mesh
                samples (int): The amount of areas to calculate.

        Returns:
                histogram ([int]): A 10-bins normalized histogram
    """
    vertex_matrix = ms.current_mesh().vertex_matrix()
    volumes = []
    for v1, v2, v3, v4 in smart_subsampling_4(vertex_matrix=vertex_matrix, n=samples, seed=seed):

        d1 = v1 - v4
        d2 = v2 - v4
        d3 = v3 - v4

        volumes.append(np.abs(np.dot(d1, (np.cross(d2, d3)))) / 6)

    bins = build_histogram(np.cbrt(volumes), (0,np.cbrt(1/3)), n_bins=10,normalize=True)
    
    return bins



if __name__ == '__main__':
    pass
    




