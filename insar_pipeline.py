"""
InSAR Quadtree Downsampling + Inversion (Mogi & Okada)

Pipeline:
1. Read velocity.h5 from MintPy output
2. Downsample using adaptive quadtree
3. Save downsampled data to velocity_downsample.h5
4. Invert for Mogi point source model
5. Invert for Okada dislocation model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


# ============================================================================
# Quadtree Downsampling
# ============================================================================

class QuadTreeNode:
    """A single node in the quadtree."""
    
    def __init__(self, x_min, y_min, x_max, y_max, data, depth=0):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.data = data  # 2D array of this cell
        self.depth = depth
        
        # Children
        self.children = [None, None, None, None]  # NW, NE, SW, SE
        self.is_leaf = True
        
    def get_variance(self):
        """Compute variance of data in this cell."""
        return np.var(self.data)
    
    def get_mean(self):
        """Compute mean value in this cell."""
        return np.mean(self.data)
    
    def get_center(self):
        """Return center coordinates (x, y) of this cell."""
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0
        return cx, cy
    
    def subdivide(self):
        """Divide this cell into 4 children."""
        if self.data.shape[0] < 2 or self.data.shape[1] < 2:
            return False
        
        x_mid = (self.x_min + self.x_max) / 2.0
        y_mid = (self.y_min + self.y_max) / 2.0
        
        ny, nx = self.data.shape
        ny_mid = ny // 2
        nx_mid = nx // 2
        
        # NW (top-left)
        self.children[0] = QuadTreeNode(
            self.x_min, y_mid, x_mid, self.y_max,
            self.data[:ny_mid, :nx_mid],
            depth=self.depth + 1
        )
        
        # NE (top-right)
        self.children[1] = QuadTreeNode(
            x_mid, y_mid, self.x_max, self.y_max,
            self.data[:ny_mid, nx_mid:],
            depth=self.depth + 1
        )
        
        # SW (bottom-left)
        self.children[2] = QuadTreeNode(
            self.x_min, self.y_min, x_mid, y_mid,
            self.data[ny_mid:, :nx_mid],
            depth=self.depth + 1
        )
        
        # SE (bottom-right)
        self.children[3] = QuadTreeNode(
            x_mid, self.y_min, self.x_max, y_mid,
            self.data[ny_mid:, nx_mid:],
            depth=self.depth + 1
        )
        
        self.is_leaf = False
        return True


class InSARQuadTree:
    """Adaptive quadtree for InSAR data downsampling."""
    
    def __init__(self, data):
        """
        Initialize quadtree.
        
        Parameters
        ----------
        data : 2D array
            InSAR velocity or displacement field
        """
        self.data = data
        ny, nx = data.shape
        
        self.root = QuadTreeNode(0, 0, nx, ny, data, depth=0)
        self.nodes = []
    
    def subdivide_adaptive(self, variance_threshold, max_depth=8):
        """
        Recursively subdivide cells based on variance threshold.
        
        Parameters
        ----------
        variance_threshold : float
            If variance > threshold, subdivide further
        max_depth : int
            Maximum subdivision depth
        """
        self._subdivide_recursive(self.root, variance_threshold, max_depth)
    
    def _subdivide_recursive(self, node, threshold, max_depth):
        """Recursively subdivide node if variance is high."""
        if node.depth >= max_depth:
            self.nodes.append(node)
            return
        
        variance = node.get_variance()
        
        if variance > threshold and node.subdivide():
            for child in node.children:
                self._subdivide_recursive(child, threshold, max_depth)
        else:
            self.nodes.append(node)
    
    def get_downsampled_data(self):
        """
        Extract downsampled data points from leaf nodes.
        
        Returns
        -------
        x_samples : 1D array, shape (n_samples,)
            X coordinates (pixel)
        y_samples : 1D array, shape (n_samples,)
            Y coordinates (pixel)
        z_samples : 1D array, shape (n_samples,)
            Velocity/displacement values
        """
        x_samples = []
        y_samples = []
        z_samples = []
        
        for node in self.nodes:
            cx, cy = node.get_center()
            x_samples.append(cx)
            y_samples.append(cy)
            z_samples.append(node.get_mean())
        
        return (np.array(x_samples), np.array(y_samples), np.array(z_samples))


# ============================================================================
# MintPy HDF5 I/O
# ============================================================================

def read_velocity_h5(filename):
    """
    Read velocity.h5 from MintPy output.
    
    Parameters
    ----------
    filename : str
        Path to velocity.h5
    
    Returns
    -------
    velocity : 2D array
        Velocity field (mm/yr)
    geo_coords : dict
        Geographic metadata (lat, lon, x, y indices)
    """
    print(f"Reading: {filename}")
    
    with h5py.File(filename, 'r') as f:
        # Print available datasets
        print("Available datasets:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape}")
        
        # Read velocity dataset
        # Usually stored as "velocity" or "timeseries"
        if 'velocity' in f:
            velocity = f['velocity'][:]
        elif 'timeseries' in f:
            velocity = f['timeseries'][:]
        else:
            # Use first dataset
            velocity = f[list(f.keys())[0]][:]
        
        # Read metadata
        geo_coords = {}
        if 'latitude' in f:
            geo_coords['lat'] = f['latitude'][:]
        if 'longitude' in f:
            geo_coords['lon'] = f['longitude'][:]
        if 'x' in f:
            geo_coords['x'] = f['x'][:]
        if 'y' in f:
            geo_coords['y'] = f['y'][:]
    
    print(f"Velocity shape: {velocity.shape}")
    print(f"Velocity range: {np.nanmin(velocity):.3f} to {np.nanmax(velocity):.3f}")
    
    return velocity, geo_coords


def write_velocity_h5(filename, velocity, x_samples, y_samples, z_samples):
    """
    Write downsampled velocity to HDF5 file.
    
    Parameters
    ----------
    filename : str
        Output filename
    velocity : 2D array
        Original velocity grid
    x_samples, y_samples, z_samples : 1D arrays
        Downsampled points
    """
    print(f"Writing: {filename}")
    
    with h5py.File(filename, 'w') as f:
        # Save original grid for reference
        f.create_dataset('velocity_grid', data=velocity, compression='gzip')
        
        # Save downsampled points
        f.create_dataset('x_samples', data=x_samples, compression='gzip')
        f.create_dataset('y_samples', data=y_samples, compression='gzip')
        f.create_dataset('z_samples', data=z_samples, compression='gzip')
        
        # Metadata
        f.attrs['n_samples'] = len(x_samples)
        f.attrs['compression_ratio'] = (velocity.size) / len(x_samples)
        
        print(f"  Saved {len(x_samples)} downsampled points")


# ============================================================================
# Forward Models: Mogi & Okada
# ============================================================================

def mogi_forward(params, observation_points):
    """
    Mogi point source forward model.
    
    Computes vertical displacement due to a spherical pressure source
    in an elastic half-space.
    
    Parameters
    ----------
    params : array-like, shape (4,)
        [x, y, depth, dP*V]
        x, y : source location (pixels or km)
        depth : source depth (pixels or km)
        dP*V : pressure change × volume (related to moment)
    
    observation_points : array-like, shape (n, 2)
        [x, y] coordinates of observation points
    
    Returns
    -------
    uz : 1D array, shape (n,)
        Vertical displacement at each observation point
    """
    x0, y0, depth, dpv = params
    
    # Distance from source to observation points
    dx = observation_points[:, 0] - x0
    dy = observation_points[:, 1] - y0
    r = np.sqrt(dx**2 + dy**2 + depth**2)
    
    # Mogi solution: uz = (c * dPV) / r^3
    # c is a coefficient (simplified version)
    c = 1.0 / (np.pi * 4/3)  # Simplified
    uz = c * dpv / (r**3)
    
    return uz


def okada_forward(params, observation_points):
    """
    Okada dislocation forward model (simplified).
    
    Computes vertical displacement due to a rectangular fault dislocation
    in an elastic half-space.
    
    Parameters
    ----------
    params : array-like, shape (7,)
        [x, y, depth, length, width, strike, slip]
        x, y : fault center location
        depth : fault center depth
        length : fault length
        width : fault width
        strike : strike angle (degrees)
        slip : amount of slip
    
    observation_points : array-like, shape (n, 2)
        [x, y] coordinates of observation points
    
    Returns
    -------
    uz : 1D array, shape (n,)
        Vertical displacement at each observation point
    """
    x0, y0, depth, length, width, strike, slip = params
    
    # Convert strike to radians
    strike_rad = np.deg2rad(strike)
    
    # Simplified Okada: use point dislocation approximation
    # uz depends on distance, orientation, and slip
    dx = observation_points[:, 0] - x0
    dy = observation_points[:, 1] - y0
    r = np.sqrt(dx**2 + dy**2 + depth**2)
    
    # Simplified formula (full Okada is more complex)
    # uz ∝ slip * width / r^2 * depth term
    uz = slip * (depth / (r**3)) * (length * width) / 1e6
    
    return uz


# ============================================================================
# Inversion
# ============================================================================

def invert_mogi(x_samples, y_samples, z_samples, initial_guess=None):
    """
    Invert for Mogi point source parameters.
    
    Parameters
    ----------
    x_samples, y_samples, z_samples : 1D arrays
        Downsampled observation points and their displacements
    
    initial_guess : array-like, shape (4,), optional
        Initial estimate [x, y, depth, dP*V]
    
    Returns
    -------
    params : 1D array, shape (4,)
        Fitted parameters
    residuals : 1D array
        Misfit at each observation point
    """
    print("\n" + "="*60)
    print("MOGI POINT SOURCE INVERSION")
    print("="*60)
    
    observation_points = np.column_stack([x_samples, y_samples])
    
    # Initial guess
    if initial_guess is None:
        x_center = np.mean(x_samples)
        y_center = np.mean(y_samples)
        depth = 10.0  # pixels
        dpv = 1.0
        initial_guess = [x_center, y_center, depth, dpv]
    
    def residual_func(params):
        predicted = mogi_forward(params, observation_points)
        return z_samples - predicted
    
    # Bounds: reasonable physical limits
    bounds_lower = [
        np.min(x_samples) - 20, np.min(y_samples) - 20, 1.0, 0.01
    ]
    bounds_upper = [
        np.max(x_samples) + 20, np.max(y_samples) + 20, 100.0, 100.0
    ]
    
    print(f"Initial guess: {initial_guess}")
    print(f"Bounds: {bounds_lower} to {bounds_upper}")
    
    # Least squares inversion
    result = least_squares(
        residual_func, initial_guess,
        bounds=(bounds_lower, bounds_upper),
        max_nfev=1000
    )
    
    params_fit = result.x
    residuals = result.fun
    
    print(f"\nFitted parameters:")
    print(f"  x: {params_fit[0]:.2f} (pixels)")
    print(f"  y: {params_fit[1]:.2f} (pixels)")
    print(f"  depth: {params_fit[2]:.2f} (pixels)")
    print(f"  dP*V: {params_fit[3]:.3f}")
    print(f"\nRMS misfit: {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"Chi-square: {np.sum(residuals**2):.4f}")
    
    return params_fit, residuals


def invert_okada(x_samples, y_samples, z_samples, initial_guess=None):
    """
    Invert for Okada dislocation parameters.
    
    Parameters
    ----------
    x_samples, y_samples, z_samples : 1D arrays
        Downsampled observation points and their displacements
    
    initial_guess : array-like, shape (7,), optional
        Initial estimate [x, y, depth, length, width, strike, slip]
    
    Returns
    -------
    params : 1D array, shape (7,)
        Fitted parameters
    residuals : 1D array
        Misfit at each observation point
    """
    print("\n" + "="*60)
    print("OKADA DISLOCATION INVERSION")
    print("="*60)
    
    observation_points = np.column_stack([x_samples, y_samples])
    
    # Initial guess
    if initial_guess is None:
        x_center = np.mean(x_samples)
        y_center = np.mean(y_samples)
        depth = 10.0
        length = 20.0
        width = 15.0
        strike = 45.0  # degrees
        slip = 0.5
        initial_guess = [x_center, y_center, depth, length, width, strike, slip]
    
    def residual_func(params):
        predicted = okada_forward(params, observation_points)
        return z_samples - predicted
    
    # Bounds
    bounds_lower = [
        np.min(x_samples) - 20, np.min(y_samples) - 20, 1.0,
        5.0, 5.0, 0.0, 0.001
    ]
    bounds_upper = [
        np.max(x_samples) + 20, np.max(y_samples) + 20, 100.0,
        100.0, 100.0, 360.0, 10.0
    ]
    
    print(f"Initial guess: {initial_guess}")
    print(f"Bounds: {bounds_lower} to {bounds_upper}")
    
    # Least squares inversion
    result = least_squares(
        residual_func, initial_guess,
        bounds=(bounds_lower, bounds_upper),
        max_nfev=1000
    )
    
    params_fit = result.x
    residuals = result.fun
    
    print(f"\nFitted parameters:")
    print(f"  x: {params_fit[0]:.2f} (pixels)")
    print(f"  y: {params_fit[1]:.2f} (pixels)")
    print(f"  depth: {params_fit[2]:.2f} (pixels)")
    print(f"  length: {params_fit[3]:.2f} (pixels)")
    print(f"  width: {params_fit[4]:.2f} (pixels)")
    print(f"  strike: {params_fit[5]:.2f} (degrees)")
    print(f"  slip: {params_fit[6]:.4f}")
    print(f"\nRMS misfit: {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"Chi-square: {np.sum(residuals**2):.4f}")
    
    return params_fit, residuals


# ============================================================================
# Visualization
# ============================================================================

def plot_results(velocity, x_samp, y_samp, z_samp, 
                 params_mogi, params_okada, res_mogi, res_okada):
    """Plot downsampling and inversion results."""
    
    observation_points = np.column_stack([x_samp, y_samp])
    pred_mogi = mogi_forward(params_mogi, observation_points)
    pred_okada = okada_forward(params_okada, observation_points)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Data & Downsampling
    # ----
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(velocity, cmap="RdBu_r", origin="lower")
    ax1.set_title("Original Velocity Field")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    plt.colorbar(im1, ax=ax1, label="mm/yr")
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(x_samp, y_samp, c=z_samp, cmap="RdBu_r", s=40, edgecolors='k', linewidth=0.5)
    ax2.set_xlim(0, velocity.shape[1])
    ax2.set_ylim(0, velocity.shape[0])
    ax2.set_aspect('equal')
    ax2.set_title(f"Downsampled Points ({len(x_samp)} samples)")
    ax2.set_xlabel("x (pixels)")
    ax2.set_ylabel("y (pixels)")
    
    # Row 2: Mogi Inversion
    # ----
    ax3 = plt.subplot(3, 3, 4)
    ax3.scatter(z_samp, pred_mogi, alpha=0.6, s=30)
    ax3.plot([z_samp.min(), z_samp.max()], [z_samp.min(), z_samp.max()], 'r--', lw=2)
    ax3.set_xlabel("Observed")
    ax3.set_ylabel("Predicted (Mogi)")
    ax3.set_title("Mogi: Data vs. Prediction")
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 3, 5)
    ax4.scatter(x_samp, y_samp, c=res_mogi, cmap="seismic", s=40, vmin=-np.max(np.abs(res_mogi)), 
                vmax=np.max(np.abs(res_mogi)))
    ax4.plot(params_mogi[0], params_mogi[1], 'r*', markersize=15, label='Mogi source')
    ax4.set_xlim(0, velocity.shape[1])
    ax4.set_ylim(0, velocity.shape[0])
    ax4.set_aspect('equal')
    ax4.set_title("Mogi: Residuals")
    ax4.set_xlabel("x (pixels)")
    ax4.set_ylabel("y (pixels)")
    ax4.legend()
    
    ax5 = plt.subplot(3, 3, 6)
    ax5.hist(res_mogi, bins=20, edgecolor='black', alpha=0.7)
    ax5.set_xlabel("Residual")
    ax5.set_ylabel("Count")
    ax5.set_title(f"Mogi: Residual Distribution\nRMS={np.sqrt(np.mean(res_mogi**2)):.4f}")
    ax5.grid(True, alpha=0.3)
    
    # Row 3: Okada Inversion
    # ----
    ax6 = plt.subplot(3, 3, 7)
    ax6.scatter(z_samp, pred_okada, alpha=0.6, s=30, color='green')
    ax6.plot([z_samp.min(), z_samp.max()], [z_samp.min(), z_samp.max()], 'r--', lw=2)
    ax6.set_xlabel("Observed")
    ax6.set_ylabel("Predicted (Okada)")
    ax6.set_title("Okada: Data vs. Prediction")
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 8)
    ax7.scatter(x_samp, y_samp, c=res_okada, cmap="seismic", s=40, vmin=-np.max(np.abs(res_okada)), 
                vmax=np.max(np.abs(res_okada)))
    ax7.plot(params_okada[0], params_okada[1], 'g*', markersize=15, label='Okada center')
    ax7.set_xlim(0, velocity.shape[1])
    ax7.set_ylim(0, velocity.shape[0])
    ax7.set_aspect('equal')
    ax7.set_title("Okada: Residuals")
    ax7.set_xlabel("x (pixels)")
    ax7.set_ylabel("y (pixels)")
    ax7.legend()
    
    ax8 = plt.subplot(3, 3, 9)
    ax8.hist(res_okada, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax8.set_xlabel("Residual")
    ax8.set_ylabel("Count")
    ax8.set_title(f"Okada: Residual Distribution\nRMS={np.sqrt(np.mean(res_okada**2)):.4f}")
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("insar_inversion_results.png", dpi=150, bbox_inches='tight')
    print("Saved: insar_inversion_results.png")
    plt.show()


# ============================================================================
# Main Pipeline
# ============================================================================

def main(input_file="velocity.h5", output_file="velocity_downsample.h5",
         variance_threshold=0.05, max_depth=7):
    """
    Complete pipeline: read → downsample → invert.
    
    Parameters
    ----------
    input_file : str
        Path to velocity.h5 (MintPy output)
    output_file : str
        Path to save downsampled data
    variance_threshold : float
        Quadtree subdivision threshold
    max_depth : int
        Maximum quadtree depth
    """
    
    # ====== Step 1: Read velocity data ======
    velocity, geo_coords = read_velocity_h5(input_file)
    
    # Handle NaN values
    velocity_clean = np.nan_to_num(velocity, nan=0.0)
    
    # ====== Step 2: Quadtree downsampling ======
    print("\n" + "="*60)
    print("QUADTREE DOWNSAMPLING")
    print("="*60)
    
    qtree = InSARQuadTree(velocity_clean)
    qtree.subdivide_adaptive(variance_threshold, max_depth=max_depth)
    
    x_samp, y_samp, z_samp = qtree.get_downsampled_data()
    
    print(f"Original points: {velocity.size}")
    print(f"Downsampled points: {len(x_samp)}")
    print(f"Compression ratio: {velocity.size / len(x_samp):.2f}x")
    
    # ====== Step 3: Save downsampled data ======
    write_velocity_h5(output_file, velocity_clean, x_samp, y_samp, z_samp)
    
    # ====== Step 4: Mogi inversion ======
    params_mogi, res_mogi = invert_mogi(x_samp, y_samp, z_samp)
    
    # ====== Step 5: Okada inversion ======
    params_okada, res_okada = invert_okada(x_samp, y_samp, z_samp)
    
    # ====== Step 6: Visualization ======
    plot_results(velocity_clean, x_samp, y_samp, z_samp,
                 params_mogi, params_okada, res_mogi, res_okada)
    
    # ====== Summary ======
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Quadtree variance threshold: {variance_threshold}")
    print(f"Quadtree max depth: {max_depth}")
    print(f"\nMogi source located at:")
    print(f"  x={params_mogi[0]:.2f}, y={params_mogi[1]:.2f}, depth={params_mogi[2]:.2f}")
    print(f"\nOkada fault centered at:")
    print(f"  x={params_okada[0]:.2f}, y={params_okada[1]:.2f}, depth={params_okada[2]:.2f}")
    print("="*60)


if __name__ == "__main__":
    # Example: synthetic data
    # ====
    # To use real MintPy output, replace with:
    #   main(input_file="velocity.h5", output_file="velocity_downsample.h5")
    
    print("Generating synthetic velocity data for demo...")
    
    # Create synthetic velocity field
    ny, nx = 128, 128
    y, x = np.mgrid[0:ny, 0:nx]
    x = x.astype(float)
    y = y.astype(float)
    
    # Features
    feature1 = 3.0 * np.exp(-((x - 40) ** 2 + (y - 40) ** 2) / (2 * 12.0**2))
    feature2 = -2.0 * np.exp(-((x - 90) ** 2 + (y - 90) ** 2) / (2 * 15.0**2))
    trend = 0.02 * (x + y)
    
    velocity_syn = feature1 + feature2 + trend
    
    # Add noise
    rng = np.random.default_rng(42)
    noise = 0.1 * rng.standard_normal(velocity_syn.shape)
    velocity_syn = velocity_syn + noise
    
    # Save synthetic data as HDF5
    with h5py.File("velocity_synthetic.h5", 'w') as f:
        f.create_dataset('velocity', data=velocity_syn, compression='gzip')
        f.attrs['description'] = 'Synthetic velocity field'
    
    print("Saved synthetic data to: velocity_synthetic.h5\n")
    
    # Run pipeline
    main(input_file="velocity_synthetic.h5", 
         output_file="velocity_downsample.h5",
         variance_threshold=0.05, 
         max_depth=7)
