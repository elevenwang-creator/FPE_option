"""
The numerical solution of the Fokker-Planck equation using B-splines 
under the Heston stochastic progress.

Author: Langtao Wang
Date: June 2025
"""


#Ignore warnings
from warnings import filterwarnings
filterwarnings('ignore')

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import Optional, Union

import scipy.sparse as sp 
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal
from scipy.sparse.linalg import LinearOperator, splu
from scipy.special import roots_legendre, roots_chebyt
import seaborn as sns
import cvxpy as cp

# Set plotting style
plt.style.use("seaborn-v0_8-paper")
sns.set_style("darkgrid")


class GenerateKnots:
    def __init__(self, n, p, method='uniform', center=0.2, boundary=(0, 1), mean=50, std=0.1, cheby_num=13):
        self.n = n                      # hyperbolic function knots number
        self.p = p                      # degree of B-spline  
        self.method = method
        self.center = center
        self.boundary = boundary
        self.mean = mean
        self.std = std
        self.cheby_knots = cheby_num            # chebyshev knots number

    def normalize(self, x):
        """
        Normalize a list of numbers to the range [0, 1].    
        """
        # amazonq-ignore-next-line
        x = np.asarray(x)
        min_val, max_val = x.min(), x.max()
        return (x - min_val) / (max_val - min_val)
        
    def func_parabolic(self, n, boundary):
        """
        Generate a parabolic function for knot distribution. -- f(x) = 0.5 * a * (x - factor)^2 + centor

        Parameters:
        n: number of points
        factor: factor of increasing intervals (default: 1.0, float)
        centor: center point (default: 0.2, float)
        boundary: tuple of (low_y, high_y) for the hyperbolic function (default: (0, 1), tuple)
        """
        low_y, high_y = boundary
        centor = self.center

        # n number correct, keep symmetry
        divide = np.sqrt((high_y - centor) / (centor - low_y)) + 1
        if n % divide == 0:
            n = int(n)
        else:
            n = int(int(n / divide) * divide)
        factor = 2.0
   
        # Calculate parameter a
        a = abs(2 * (low_y - centor) / factor**2)
        
        # Calculate upward limit
        upward = np.sqrt(2 * (high_y - centor) / a) + factor
   
        x = np.concatenate([np.linspace(0, upward, n-1),[factor]])
        x = np.sort(x)
        y = np.zeros_like(x)
        
        # Calculate y values using left half formula first
        y_temp = -0.5 * a * (x - factor) ** 2 + centor
        
        # Find the actual center point (where y is closest to centor)
        distances_to_center = np.abs(y_temp - centor)
        center_idx = np.argmin(distances_to_center)
        
        # Use center_idx to split (not x < factor)
        mask = np.arange(len(x)) <= center_idx
        
        # Apply original formulas
        y[mask] = -0.5 * a * (x[mask] - factor) ** 2 + centor
        y[~mask] = 0.5 * a * (x[~mask] - factor) ** 2 + centor
        
        # Sort and ensure centor is included
        y = np.sort(y)
 
        return [n, y]

    def chebyshev_knots(self, n, a, b):
        """
        Generate Chebyshev nodes in the interval [a, b].

        Parameters:
        n: number of nodes
        a: lower bound of the interval
        b: upper bound of the interval

        Returns:
        np.array: Chebyshev nodes in [a, b]
        """
        x, _ = roots_chebyt(n)
        x = np.concatenate([[-1], x, [1]])
        # Transform from [-1, 1] to [a, b]
        x = (b - a) / 2 * x + (a + b) / 2
        return x
    
    def knots_concat(self, knots, medim_knot, left, right):
        """
        Concatenate additional knots between existing knots.

        Parameters:
        knots: original knot vector
        left: left boundary 
        right: right boundary 
        medim_knot: knots in the middle region

        Returns:
        np.array: new knot vector with inserted points
        """
        left_mask = np.isclose(knots, left, rtol=1e-10) | (knots < left)
        right_mask = np.isclose(knots, right, rtol=1e-10) | (knots > right)
        left_knots = knots[left_mask]
        right_knots = knots[right_mask]
        medim_knots = medim_knot

        points = np.concatenate([left_knots, medim_knots, right_knots])
        points = np.unique(points.round(8))
        return points
    
    def generate_knots(self):
        """
        Adjust the knot vector to create a non-uniform distribution of knots.

        Parameters:
        n: knots number
        p: degreee of B-spline
        method: parameterization method, 'uniform' or 'non-uniform'
        center: center point (0-1)
        foctor: factor of increasing (default: 0.05, float)
        """
        n = self.n
        p = self.p
        method = self.method
        #direction = self.direction
        boundary = self.boundary
        boundary_normal = self.normalize(boundary)
        mean = self.mean
        std = self.std
        x_min = mean - 4.5 * std
        x_min_normal = (x_min - boundary[0]) / (boundary[1] - boundary[0])
        x_max = mean + 4.5 * std
        x_max_normal = (x_max - boundary[0]) / (boundary[1] - boundary[0])
        n_knots = self.cheby_knots

        internal_num = n - 2 * p 
        # Generate internal knots
        if method == 'uniform':
            # Generate uniform knots
            internal_knots = np.linspace(0, 1, internal_num)
        elif method == 'non-uniform':
            x_knots = self.chebyshev_knots(n_knots, x_min, x_max)
            x_normal = (x_knots - boundary[0]) / (boundary[1] - boundary[0])

            n_new, internal_knots = self.func_parabolic(internal_num, boundary_normal)
            internal_knots = self.knots_concat(internal_knots, x_normal, x_min_normal, x_max_normal)
            
        n_new = len(internal_knots)

        # Add the boundary knots
        knots = np.zeros(n_new + 2 * p)
        knots[:p] = boundary_normal[0]  # start point repeated p times
        knots[-p:] = boundary_normal[1] # end point repeated p times
        knots[p:-p] = internal_knots

        return knots


class BSplineBasis:
    """
    Class to represent a B-spline basis function.
    """
    def __init__(self, degree, knots):
        """
        Initialize the B-spline basis function.

        Parameters:
        -----------
        degree: degree of the basis function
        knots: knot vector
        """
        if degree < 0:
            raise ValueError("Degree must be non-negative")
            
        self.degree = degree
        self.knots = np.array(knots)

    @lru_cache(maxsize=1024)
    def deBoorCoxCoeff(self, x, t_min, t_max):
        if t_min < t_max and t_min <= x <= t_max:
            return (x - t_min) / (t_max - t_min)
        return 0.0
        
    @lru_cache(maxsize=1024)
    def deBoorCox(self, x, i, k):
        """
        Compute the value of a B-spline basis function at x.
        
        Parameters:
        -----------
        x: point at which to evaluate the basis function
        i: index of the basis function
        k: degree of the basis function
        """
        # For k = 0, basis function is a step function
        if k == 0:
            if self.knots[i] <= x < self.knots[i+1]:
                return 1.0
            return 0.0
        else:
            # Compute the value using the recursive formula
            coeff1 = self.deBoorCoxCoeff(x, float(self.knots[i]), float(self.knots[i+k]))
            coeff2 = self.deBoorCoxCoeff(x, float(self.knots[i+1]), float(self.knots[i+k+1]))

            b1 = self.deBoorCox(x, i, k-1)
            b2 = self.deBoorCox(x, i+1, k-1)

            return coeff1 * b1 + (1 - coeff2) * b2
        
    def basis_function(self, x) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Compute all basis functions at point x.
        x can be a scalar or numpy array
        """
        # Convert input to numpy array if it isn't already
        x = np.asarray(x)
        num_basis = self.knots.size - self.degree - 1

        # If x is a scalar, use the original method
        if x.ndim == 0:
            return np.array([self.deBoorCox(float(x), i, self.degree) 
                             for i in range(num_basis)])
        
        # For array input, use sparse matrix
        rows, cols, data = [], [], []

        for i in range(num_basis):
            for j, xj in enumerate(x):
                value = self.deBoorCox(float(xj), i, self.degree)
                if np.abs(value) > 1e-6:   # Only add non-zero values
                    rows.append(j)
                    cols.append(i)
                    data.append(value)
        
        # Create sparse matrix
        result = sp.csr_matrix((data, (rows, cols)),
                               shape=(x.size, num_basis),
                               dtype=float)
        return result

    def first_derivative(self, x):
        """
        Compute first derivative of B-spline basis function
        """
        x = np.asarray(x)
        num_basis = self.knots.size - self.degree - 1
        rows, cols, data = [], [], []

        for i in range(num_basis):
            # Vectorized computation for all x values at once
            b1_vals = np.array([self.deBoorCox(float(xj), i, self.degree-1) for xj in x])
            b2_vals = np.array([self.deBoorCox(float(xj), i+1, self.degree-1) for xj in x])
            
            # Vectorized knot differences
            t11, t12 = self.knots[i], self.knots[i+self.degree]
            t21, t22 = self.knots[i+1], self.knots[i+self.degree+1]
            
            # Vectorized division with zero handling
            b1_vals = np.where(t11 != t12, b1_vals / (t12 - t11), 0)
            b2_vals = np.where(t21 != t22, b2_vals / (t22 - t21), 0)
            
            values = self.degree * (b1_vals - b2_vals)
            
            # Find non-zero values and add to sparse matrix data
            nonzero_mask = np.abs(values) > 1e-6
            nonzero_indices = np.where(nonzero_mask)[0]
            
            rows.extend(nonzero_indices)
            cols.extend([i] * len(nonzero_indices))
            data.extend(values[nonzero_mask])
        
        return sp.csr_matrix((data, (rows, cols)), shape=(x.size, num_basis), dtype=float)
    

class RecombinationBasis(BSplineBasis):
    """
    Class to represent a recombination B-spline basis function.
    """
    def __init__(self, degree, knots, conditions=('dirichlet', 'newmann')):
        """
        Initialize the recombination B-spline basis function.

        Parameters:
        -----------
        degree: degree of the basis function
        knots: knot vector
        conditions: tuple of boundary conditions ('dirichlet' or 'newmann') 
                    left is left boundary, right is right boundary 
        """
        super().__init__(degree, knots)

        if self.degree < 1:
            raise ValueError("Degree must be at least 1 for recombination basis")
            
        self.num_basis = self.knots.size - self.degree - 1
        self.diff_trans = len(conditions)

        self.left_cond, self.right_cond = conditions
        if self.left_cond not in ('dirichlet', 'newmann'):
            raise ValueError("Left boundary condition must be 'dirichlet' or 'newmann'")
        if self.right_cond not in ('dirichlet', 'newmann'):
            raise ValueError("Right boundary condition must be 'dirichlet' or 'newmann'")
    
    @property
    def recombination_matrix(self):
        """
        Construct the recombination matrix based on boundary conditions.
        """
        Rcol = self.num_basis - self.diff_trans
        rcomb = np.zeros((Rcol + 2, Rcol))
        rcomb[1:-1] = np.eye(Rcol)

        # Boundary conditions
        if self.left_cond == 'dirichlet' and self.right_cond == 'dirichlet':
            rcomb_matrix = rcomb
        elif self.left_cond == 'dirichlet' and self.right_cond == 'newmann':
            rcomb_matrix = rcomb
            rcomb_matrix[-1, -1] = 1.0
        elif self.left_cond == 'newmann' and self.right_cond == 'dirichlet':
            rcomb_matrix = rcomb
            rcomb_matrix[0, 0] = 1.0
        elif self.left_cond == 'newmann' and self.right_cond == 'newmann':
            rcomb_matrix = rcomb
            rcomb_matrix[0, 0] = 1.0
            rcomb_matrix[-1, -1] = 1.0
        else:
            raise ValueError("Invalid boundary conditions.")

        return sp.csr_matrix(rcomb_matrix)
        
    def basis_function(self, x):
        """
        Compute all recombination basis functions at point x.
        x can be a scalar or numpy array
        """
        # Get original B-spline basis functions
        original_basis = super().basis_function(x)
        recomb_matrix = self.recombination_matrix

        # Apply recombination matrix
        if sp.issparse(original_basis):
            result = original_basis @ recomb_matrix
            return sp.csr_matrix(result)
        else:
            return original_basis @ recomb_matrix
        
    def first_derivative(self, x):
        """
        Compute first derivative of recombination B-spline basis function
        """
        original_deriv = super().first_derivative(x)
        recomb_matrix = self.recombination_matrix

        if sp.issparse(original_deriv):
            result = original_deriv @ recomb_matrix
            return sp.csr_matrix(result)
        else:
            return original_deriv @ recomb_matrix

class MultivariateBSpline:
    """
    Multi-dimensional B-spline basis functions using tensor products
    """
    def __init__(self, degrees, knots_list, conditions_list):
        """
        Initialize multi-dimensional B-spline
        
        Parameters:
        -----------
        degrees: list of degrees for each dimension
        knots_list: list of knot vectors for each dimension
        """
        if len(degrees) != len(knots_list):
            raise ValueError("Number of degrees must match number of knot vectors")
            
        self.dim = len(degrees)

        if not conditions_list:
            self.bases = [BSplineBasis(deg, knots) 
                        for deg, knots in zip(degrees, knots_list)]
        else:
            self.bases = [RecombinationBasis(deg, knots, conditions)
                         for deg, knots, conditions in zip(degrees, knots_list, conditions_list)]
    
    def tensor_product_basis(self, points_list):
        """
        Compute tensor product basis functions for all dimensions
        
        Parameters:
        -----------
        points_list: list of points for each dimension
        
        Returns:
        ---------
        Sparse matrix representing tensor product basis functions
        """
        if len(points_list) != self.dim:
            raise ValueError("Number of point sets must match number of dimensions")
        
        # Compute and convert to CSR format in one step
        result = sp.csr_matrix(self.bases[0].basis_function(points_list[0]))
        for i in range(1, self.dim):
            basis_i = sp.csr_matrix(self.bases[i].basis_function(points_list[i]))
            result = sp.kron(result, basis_i, format='csr')
        
        return result
    
    def partial_derivative(self, points_list, dimension):
        """
        Compute partial derivative with respect to specified dimension
        
        Parameters:
        -----------
        points_list: list of points for each dimension
        dimension: index of dimension to derive (0-based)
        
        Returns:
        ---------
        Sparse matrix of partial derivatives
        """
        if dimension >= self.dim:
            raise ValueError(f"Dimension {dimension} exceeds spline dimensionality")
            
        # Compute tensor product directly with derivatives
        if dimension == 0:
            result = sp.csr_matrix(self.bases[0].first_derivative(points_list[0]))
        else:
            result = sp.csr_matrix(self.bases[0].basis_function(points_list[0]))
            
        for i in range(1, self.dim):
            if i == dimension:
                basis_i = sp.csr_matrix(self.bases[i].first_derivative(points_list[i]))
            else:
                basis_i = sp.csr_matrix(self.bases[i].basis_function(points_list[i]))
            result = sp.kron(result, basis_i, format='csr')
            
        return result
    

class HestonSolver(MultivariateBSpline):
    """
    Class to solve the Heston Fokker-Planck equation using B-spline basis functions.
    """
    def __init__(self, degrees: list, knots_list: list, conditions_list: list, params: Optional[dict]=None):
        """
        Initialize the solver with B-spline basis functions and model parameters.
        
        Parameters:
        -----------
        degrees : list
            Degrees for stock price and variance [d_s, d_v]
        knots_list : list
            Knot vectors for stock price and variance [knots_s, knots_v]
        params : dict, optional
            Heston model parameters
        """
        # Validate input parameters
        if not isinstance(degrees, list) or len(degrees) == 0:
            raise ValueError("degrees must be a non-empty list")
        if not isinstance(knots_list, list) or len(knots_list) == 0:
            raise ValueError("knots_list must be a non-empty list")
        if params is not None and not isinstance(params, dict):
            raise ValueError("params must be a dictionary or None")
            
        # Initialize parent class
        super().__init__(degrees, knots_list, conditions_list)
        
        # Default Heston parameters
        self.params = {
            'kappa': 1.2,                   # Mean reversion rate
            'theta': 0.05,                  # Long-term variance
            'sigma': 0.35,                  # Volatility of variance
            'rho': -0.4,                    # Correlation
            'r': 0.1,                       # Risk-free rate
            'T': 0.6,                       # Time horizon / years
            'S0': 60.0,                     # Initial stock price
            'V0': 0.1,                      # Initial variance
            'S_range': (50.0, 150.0),       # Stock price range
            'V_range': (0.0, 1.0)           # Variance range
        }
        
        # Update parameters if provided
        if params is not None:
            self.params.update(params)
            
        # Numerical parameters
        self.trading_days = self.params['T'] * int(252)
        self.threshold = 1e-12
        self.max_iter = 1000
        
        # Domain parameters
        self.s_range = self.params['S_range'] 
        self.v_range = self.params['V_range']
        
        #self.quad_points = 800
        self.s_degree, self.v_degree = degrees
        self.s_knots, self.v_knots = knots_list

        # Grid
        self.num_insert = 251
        self.s_std = 0.1
        self.v_std = 0.001
        self.s_grid = self.grid_create(self.params['S0'], self.s_std, self.s_range)
        self.v_grid = self.grid_create(self.params['V0'], self.v_std, self.v_range)

        # Cache to avoid recomputation
        self._basis = None
        self._weights = None
        self._nodes_weights = None
        self._delta_cache = {}
        self._q0_cache = {}
        self._qt_ode_cache = {}
        self._fpe_solver_cache = {}

        self.s_points = self.s_domain 
        self.v_points = self.points_list[1]
        
    @property
    def s_domain(self):
        s_min, s_max = self.s_range
        physics_domain = self.points_list[0] * (s_max - s_min) + s_min
        return physics_domain

    def chebyshev_nodes(self, n, a, b):
        """
        Generate Chebyshev nodes in the interval [a, b].

        Parameters:
        n: number of nodes
        a: lower bound of the interval
        b: upper bound of the interval

        Returns:
        np.array: Chebyshev nodes in [a, b]
        """
        x, _ = roots_chebyt(n)
        x = np.concatenate([[-1], x, [1]])
        # Transform from [-1, 1] to [a, b]
        x = (b - a) / 2 * x + (a + b) / 2
        return x
        
    def grid_create(self, mean, std_dev, bound=()):
        """
        Create a non-uniform grid with denser points around the mean and near v=0 for variance.
        
        Parameters:
        -----------
        mean : float
            Center of the dense region (e.g., S0 for s_points, V0 for v_points).
        std_dev : float
            Standard deviation for grid scaling (e.g., 0.1 for s, 0.01 or 0.02 for v).
        bound : tuple
            Lower and upper bounds of the domain (lb, ub), e.g., (0.0, 1.0).
        
        Returns:
        --------
        np.array
            Non-uniform grid points, sorted and unique.
        """
        lb, ub = bound
        lb_interm = mean - 5 * std_dev  # Tighter range around mean
        ub_interm = mean + 5 * std_dev
        num_interm = int(round(self.num_insert * 0.3))  # 30% of points in dense region
        right_trail = int(round(self.num_insert * 0.5))  # 50% in upper tail
        left_trail = int(round(self.num_insert * 0.2))  # 20% in lower tail
        
        # Create grid with finer resolution near mean
        x = np.concatenate([
            np.linspace(lb, lb_interm, left_trail),  # Coarse lower tail
            #self.chebyshev_nodes(left_trail, lb, lb_interm),  # Chebyshev lower tail
            np.linspace(lb_interm, mean - std_dev / 5, num_interm // 3),  # Approaching mean
            #self.chebyshev_nodes(num_interm // 3, lb_interm, mean - std_dev / 5),  # Chebyshev approaching mean
            np.linspace(mean - std_dev / 5, mean + std_dev / 5, num_interm // 3),  # Dense near mean
            #self.chebyshev_nodes(num_interm // 3, mean - std_dev / 5, mean + std_dev / 5),  # Chebyshev dense near mean
            np.array([mean]),  # Ensure mean is included
            np.linspace(mean + std_dev / 5, ub_interm, num_interm // 3),  # Leaving mean
            #self.chebyshev_nodes(num_interm // 3, mean + std_dev / 5, ub_interm),  # Chebyshev leaving mean
            np.linspace(ub_interm, ub, right_trail)  # Coarse upper tail
            #self.chebyshev_nodes(right_trail, ub_interm, ub)  # Chebyshev upper tail
        ])
        up_region  = np.linspace(ub-0.1, ub, int(self.num_insert * 0.1))
        x = np.concatenate([x, up_region])
        # For v_points, add extra points near v=0 due to Feller condition violation
        if mean == self.params['V0']:  # Check if generating v_points
            zero_region = np.linspace(0.0, 0.01, int(round(self.num_insert * 0.2)))
            x = np.concatenate([zero_region, x])
        
        return np.unique(np.sort(x))  # Sort and remove duplicates
    
    def gauss_legendre(self, n, a, b):
        """
        Compute the Gauss-Legendre quadrature points and weights.

        Parameters:
        -----------
        n : int
            Number of quadrature points
        a : float
            Lower integration limit
        b : float
            Upper integration limit

        Returns:
        --------
        x : np.array
            Quadrature points
        w : np.array
            Quadrature weights
        """
        x, w = roots_legendre(n)
        x = np.concatenate([[-1], x])
        x = (b - a) / 2 * x + (a + b) / 2
        trans_coeff = (b - a) / 2
        w = np.concatenate([[0], w])
        w = w * trans_coeff
        return [x, w]

    def normalize(self, x):
        """
        Normalize a list of numbers to the range [0, 1].    
        """
        # amazonq-ignore-next-line
        x = np.asarray(x)
        min_val, max_val = x.min(), x.max()
        return (x - min_val) / (max_val - min_val)
    
    @property
    def nodes_weights(self):
        if self._nodes_weights is not None:
            return self._nodes_weights
        
        num_gauss = np.ceil((self.s_degree + self.v_degree + 1) / 2).astype(int)
        #num_gauss = 6
        num_quad = num_gauss + 1

        def compute_quad(knots):
            unique_knots = np.unique(knots.round(6))
            n_intervals = len(unique_knots) - 1
            points = np.zeros(n_intervals * num_quad)
            weights = np.zeros(n_intervals * num_quad)
            
            for i in range(n_intervals):
                start, end = i * num_quad, (i + 1) * num_quad
                points[start:end], weights[start:end] = self.gauss_legendre(
                    num_gauss, unique_knots[i], unique_knots[i+1])
            
            return points, weights
        
        grid_s = self.normalize(self.s_grid)
        #grid_s = self.s_knots
        #grid_s = self.s_grid
        grid_v = self.v_grid
        #grid_v = self.v_knots
        
        s_points, s_weights = compute_quad(grid_s)
        v_points, v_weights = compute_quad(grid_v)
        
        self._nodes_weights = {
            's_nodes': s_points, 's_weights': s_weights,
            'v_nodes': v_points, 'v_weights': v_weights
        }
    
        return self._nodes_weights
   
    @property
    def points_list(self):
        """
        Get the list of points for stock price and variance.
        """
        return [self.nodes_weights['s_nodes'], 
                self.nodes_weights['v_nodes']]
    @property
    def basis(self):
        if self._basis is None:
            self._basis = self.tensor_product_basis(self.points_list)
        return self._basis

    @property
    def jacobian_factor(self):
        """
        Compute the Jacobian factor for the Heston Fokker-Planck equation.
        
        Returns:
        --------
        np.array
            Jacobian factor 
        """
        # Get s, v range
        s_min, s_max = self.s_range
        v_min, v_max = self.v_range

        s_diff = s_max - s_min
        v_diff = 1.0

        # Jacobian factor for normalization
        jacobian = s_diff * v_diff  

        return jacobian
    
    @property
    def integ_weights(self):
        """
        Compute integration domain for the Heston Fokker-Planck equation.

        Returns:
        --------
        np.array
            Integration domain evaluated at (s, v)
        """
        if self._weights is not None:
            return self._weights

        # Get integration weights
        s_weights = self.nodes_weights['s_weights']
        v_weights = self.nodes_weights['v_weights']
        
        # Compute Jacobian factor
        jacobian = self.jacobian_factor

        s_weights *= jacobian 
        s_weights = sp.diags(s_weights, format='csr')
        v_weights = sp.diags(v_weights, format='csr')

        self._weights = sp.kron(s_weights, v_weights)

        return self._weights

    @property
    def mass_matrix(self):
        """
        Compute mass matrix using sparse matrices
        """
        # Get basis functions at quadrature points
        two_basis = self.basis
        
        # Integration weights
        weights = self.integ_weights
        
        # Compute mass matrix using sparse operations
        mass = two_basis.T @ weights @ two_basis

        #threshold = self.threshold
        #mass.data[mass.data < threshold] = 0.0
        #mass.eliminate_zeros() 
        #mass = 0.5 * (mass + mass.T)  # Ensure symmetry
        #mass += threshold * sp.eye(mass.shape[0], format='csr')
        return mass
    
    @property
    def stiffness_matrix(self):
        """
        Compute the stiffness matrix for the Heston Fokker-Planck equation.

        Returns:
        --------
        np.array
            Stiffness matrix evaluated at (s, v)
        """
        # Get parameters
        r = self.params['r']                            # Risk-free rate
        kappa = self.params['kappa']                    # Mean reversion rate
        theta = self.params['theta']                    # Long-term variance
        eta = self.params['sigma']                      # Volatility of variance
        rho = self.params['rho']                        # Correlation
        
        # Transform factors
        j = self.jacobian_factor
        if abs(j) < 1e-12:
            raise ValueError("Jacobian factor too small, potential division by zero")        
        
        # 2. Get grid, basis, and derivative matrices
        s = self.s_points
        v = self.points_list[1]
        n_s, n_v = len(s), len(v)

        two_basis = self.basis
        s_partial = self.partial_derivative(self.points_list, 0)
        v_partial = self.partial_derivative(self.points_list, 1)

        # Integration weights
        weights = self.integ_weights

        #Create operators on the 2D grid using Kronecker products
        s_diag = sp.kron(sp.diags(s), sp.eye(n_v), format='csr')
        v_diag = sp.kron(sp.eye(n_s), sp.diags(v), format='csr')
        s_sq_diag = sp.kron(sp.diags(s**2), sp.eye(n_v), format='csr')
        s_v_diag = s_diag @ v_diag
        s_s_v_diag = s_sq_diag @ v_diag
        
        # Integrands for terms with s_partial
        k1_coeff = (-r + 0.5 * rho * eta) / j
        k2_coeff = 1.0 / j
        k3_coeff = 0.5 / (j**2)
        k4_coeff = 0.5 * rho * eta / j

        integ_ssvw =  k3_coeff * s_s_v_diag @ weights
        integ_svw =  k4_coeff * s_v_diag @ weights
        integ_sbw =   k1_coeff * s_diag @ weights + k2_coeff * s_v_diag @ weights

        K_sb = s_partial.T @ integ_sbw @ two_basis
        K_sv = s_partial.T @ integ_svw @ v_partial
        K_ssv = s_partial.T @ integ_ssvw @ s_partial

        # Integrands for terms with v_partial
        k5_coeff = 0.5 * eta**2 - kappa * theta 
        k6_coeff = kappa + 0.5 * rho * eta
        k8_coeff = 0.5 * eta**2

        integ_vbw = k5_coeff * weights + k6_coeff * v_diag @ weights
        integ_vw = k8_coeff * v_diag @ weights
        K_vb = v_partial.T @ integ_vbw @ two_basis
        K_vv = v_partial.T @ integ_vw @ v_partial
        K_vs = K_sv.T                   # K_vs is transpose of K_sv

        K = K_sb + K_vb + K_ssv + K_vv + K_vs + K_sv

        #threshold = self.threshold
        #K.data[np.abs(K.data) < threshold] = 0
        #K.eliminate_zeros() 
        return K

    def delta_approx(self, initial_std_dev):
        """
        Compute the initial condition for the Heston Fokker-Planck equation.
        Delta function approximated by a two-dimensional Gaussian distribution.

        Parameters:
        initial_std_dev : float
            Initial standard deviation
        v_stdev : float
            Initial standard deviation

        Returns:
        --------
        np.array
            Initial condition evaluated at (S0, V0)
        """
        # Validate input
        if initial_std_dev <= 0:
            raise ValueError("initial_std_dev must be positive")
            
        # Parameters for Gaussian distribution
        s0, v0 = self.params['S0'], self.params['V0']       # Delta function center (S0, V0)
        rho = 0.0

        # Create a grid of points
        s_range = self.s_points 
        v_range = self.v_points
        v_stdev = initial_std_dev /(s_range.max() - s_range.min())  # Standard deviation for stock price

        # Define mean and covariance for bivariate normal
        mean = [s0, v0]
        cov_matrix = np.array([[initial_std_dev**2, rho * initial_std_dev * v_stdev],
                        [rho * initial_std_dev * v_stdev, v_stdev**2]]) 

        try:
            # Create bivariate normal distribution
            delta_approx = multivariate_normal(mean=mean, cov=cov_matrix)
            
            # Create meshgrid for s and v
            V, S = np.meshgrid(v_range, s_range)
            grid_points = np.dstack((S, V))  

            # Evaluate the PDF at the grid points
            pdf_values = delta_approx.pdf(grid_points)
            
            # Check for invalid values
            if np.any(np.isnan(pdf_values)) or np.any(np.isinf(pdf_values)):
                raise ValueError("PDF evaluation produced invalid values")

            # Convert PDF to sparse format with threshold
            pdf_values[abs(pdf_values) < 1e-6] = 0.0
            np.maximum(pdf_values, 0, out=pdf_values)

        except np.linalg.LinAlgError as e:
            raise ValueError(f"Linear algebra error in delta approximation: {e}")
        except ValueError as e:
            raise ValueError(f"Value error in delta approximation: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in delta approximation: {e}")

        return pdf_values
    
    @property
    def basis_integral(self) -> np.ndarray:
        """
        Compute the integral of the B-spline basis functions.

        Returns:
        --------
        np.array
            Integral of the B-spline basis functions evaluated at (s, v)
        """
        # Get B-spline basis functions
        basis_matrix = self.basis

        # Integration weights
        weights = self.integ_weights

        # Compute integral
        integral = basis_matrix.T @ weights @ np.ones(basis_matrix.shape[0])

        return integral

    def q_initial(self, initial_std_dev):
        """
        Compute the initial condition for the Heston Fokker-Planck equation.
        Optimizer OSQP is used to solve the initial condition problem.
        0.5 * ||A c - b||^2 s.t. m^T c = 1, c >= 0.

        Return:
        --------
        initial condition evaluated at (S0, V0) -np.ndarray.
        """
        key = float(initial_std_dev)
        if key in self._q0_cache:
            return self._q0_cache[key]
        
        # Compute delta function approximation
        delta_pdf = self.delta_approx(initial_std_dev)
        delta = delta_pdf.flatten()

        # Get B-spline basis functions
        basis_matrix = self.basis 
        galerkin_matrix = self.mass_matrix
        weights = self.integ_weights
        galerkin_projection = basis_matrix.T @ weights @ delta

        # Scaled
        galerkin_diag = galerkin_matrix.diagonal()
        #D = sp.diags(np.sqrt(galerkin_diag))
        Dinv = sp.diags(1.0 / np.sqrt(galerkin_diag))
        galerkin_scaled = Dinv @ galerkin_matrix @ Dinv
        galerkin_proj = Dinv @ galerkin_projection

        m = self.basis_integral

        #Define the CVXPY problem
        n = galerkin_matrix.shape[1]
        c = cp.Variable(n)  

        objective = cp.Minimize(0.5 * cp.sum_squares(galerkin_scaled @ c - galerkin_proj)
                               + 1e-6 * cp.sum_squares(c)) # Regularization term
        constraints = [c >= 0] 
        prob = cp.Problem(objective, constraints)

        # Solve with OSQP
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=50000) 

        result = Dinv @ c.value
        #result = c.value
        np.maximum(result, 0, out=result)  # Ensure positivity
        result /= np.dot(m, result) # Normalize

        self._q0_cache[key] = result

        return result

    def qt_ode(self, sigma0, time_eval=None):
        """
        Compute the ODE for the Heston Fokker-Planck equation.

        Parameters:
        ----------
        sigma0 : float
            Initial standard deviation

        Returns:
        --------
        np.array
            ODE evaluated q(t)
        """
        key = float(sigma0)
        if key in self._qt_ode_cache:
            return self._qt_ode_cache[key]

        # Get initial condition
        t0 = 0.0
        t1 = self.params['T']
        q_initial  = self.q_initial(sigma0)

        if q_initial is None:
            print("q_initial returned None. Aborting ODE solve.")
            return None

        # Assemble matrices
        M = self.mass_matrix
        K = self.stiffness_matrix

        M_diag = M.diagonal()
        D = sp.diags(np.sqrt(M_diag))      # D_ii = sqrt(M_ii)
        Dinv = sp.diags(1.0 / np.sqrt(M_diag))

        M_scaled = Dinv @ M @ Dinv
        K_scaled = Dinv @ K @ Dinv
        q_initial_scale = D @ q_initial

        # Cache matrix factorization for efficiency
        try:
            M_lu = splu(M_scaled.tocsc())

            M_x = LinearOperator(M_scaled.shape, lambda x: M_lu.solve(x))

            def constrained_rhs(t, q):
                """Stable right-hand side computation"""
                # Compute K@q first
                Kq = - K_scaled @ q
                dqdt = M_x.matvec(Kq)
                return dqdt
            
            jacobian = - M_x.matmat(K_scaled.toarray())

            def stability_monitor(t, q):
                """Monitor solution stability"""
                # Check if solution is positive
                if np.any(q < -1e-6):
                    print(f"Warning: Negative values in solution at t={t}")
                    return -1.0
                
                # Check if probability sum is close to 1
                prob_sum = np.dot(self.basis_integral, q)
                if abs(prob_sum - 1.0) > 1e-6:
                    print(f"Warning: Probability sum {prob_sum} at t={t} deviates from 1.0")
                    return -1.0

                # Check if numerical stability is maintained
                if np.any(np.isnan(q)) or np.any(np.isinf(q)):
                    print(f"Warning: Numerical instability at t={t}")
                    return -1.0  
                
                return 1.0
                
            #stability_monitor.terminal = True  # Stop integration if instability is detected
            #stability_monitor.direction = 0    # Monitor all directions
                
            # Use more stable integration parameters
            solver_options = {
                'method': 'Radau',
                'atol': 1e-6,                       # Absolute tolerance
                'rtol': 1e-4,
                'max_step': 0.1,                    # Limit maximum time step
                'first_step': 1e-6,                 # Small initial step for stability
                #'events': stability_monitor,       # Monitor stability
                'jac': jacobian,                    # Use Jacobian for better convergence
            }
            
            if time_eval is not None:
                solver_options['t_eval'] = time_eval
            
            # Solve ODE with stability monitoring
            solution = solve_ivp(constrained_rhs, (t0, t1), q_initial_scale, **solver_options)
            
            
            # Check solution quality
            if not solution.success:
                print(f"Warning: ODE solution may be unreliable: {solution.message}")
                
            # Execute constraints on solution
            for i in range(solution.y.shape[1]):
                q = solution.y[:, i]
                q = Dinv @ q
                solution.y[:, i] = q

            self._qt_ode_cache[key] = solution 

            return solution
        
        except (np.linalg.LinAlgError, ValueError) as e:
            raise ValueError(f"ODE solver failed due to numerical error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in ODE solution: {e}")

    def fpe_solver(self, sigma0, time=None):
        """
        Solve the FPE for the Heston Fokker-Planck equation.

        Parameters:
        ----------
        sigma0 : float  
            Initial standard deviation

        Returns:
        --------
        np.array
            PDF evaluated at (s, v, t)
        """
        #key = float(sigma0)
        #if key in self._fpe_solver_cache:
        #    return self._fpe_solver_cache[key]
        # Get ODE solution (compute once)
        solution = self.qt_ode(sigma0, time_eval=time)
        if solution is None:
            raise ValueError("ODE solver failed")
        
        # Extract time points and solution
        t = solution.t
        qt = solution.y
        
        # Get basis functions
        basis = self.basis
        
        # Compute solution in 2D format
        pdf_2d = basis @ qt

        # Reshape for trapezoidal integration: (n_s, n_v, n_time)
        n_s = len(self.s_points)
        n_v = len(self.v_points)
        n_t = pdf_2d.shape[1]
        pdf_2d = pdf_2d.reshape(n_s, n_v, n_t)

        # Apply stability constraints and normalize using trapezoidal rule for each time step
        for i in range(n_t):
            slice_2d = pdf_2d[:, :, i]
            slice_2d[abs(slice_2d) < 1e-6] = 0.0  # Set small values to zero 
            np.maximum(slice_2d, 0.0, out=slice_2d)  # Ensure positivity in-place

        # Cache result
        #self._fpe_solver_cache[key] = [pdf_2d, t]

        return [pdf_2d, t]
