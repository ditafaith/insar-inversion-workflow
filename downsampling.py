"""
InSAR Downsampling using Adaptive Quadtree

Reads velocity.h5 from MintPy output and produces velocity_downsample.h5
with adaptive quadtree downsampling.

Usage:
    python downsampling.py --input velocity.h5 --output velocity_downsample.h5 --threshold 0.05 --max-depth 7
"""

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
    
    def visualize_quadtree(self, ax=None, linewidth=0.5):
        """
        Draw quadtree cell boundaries on axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        linewidth : float
            Line width for cell boundaries
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        for node in self.nodes:
            width = node.x_max - node.x_min
            height = node.y_max - node.y_min
            rect = Rectangle(
                (node.x_min, node.y_min), width, height,
                linewidth=linewidth, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        
        ax.set_xlim(0, self.data.shape[1])
        ax.set_ylim(0, self.data.shape[0])
        ax.set_aspect('equal')
        return ax


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
        Velocity field
    metadata : dict
        Geographic metadata
    """
    print(f"\n{'='*60}")
    print(f"Reading: {filename}")
    print(f"{'='*60}")
    
    with h5py.File(filename, 'r') as f:
        # Print available datasets
        print("Available datasets:")
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                print(f"  {key}: {obj.shape}, dtype={obj.dtype}")
        
        # Read velocity dataset
        if 'velocity' in f:
            velocity = f['velocity'][:]
        elif 'timeseries' in f:
            velocity = f['timeseries'][-1, :, :]  # Last time step
        else:
            # Use first dataset
            first_key = list(f.keys())[0]
            velocity = f[first_key][:]
        
        # Read metadata
        metadata = {}
        for key in ['latitude', 'longitude', 'x', 'y']:
            if key in f:
                metadata[key] = f[key][:]
        
        # Read attributes
        for attr_key in f.attrs.keys():
            metadata[f'attr_{attr_key}'] = f.attrs[attr_key]
    
    print(f"Velocity shape: {velocity.shape}")
    print(f"Data type: {velocity.dtype}")
    print(f"Velocity range: {np.nanmin(velocity):.4f} to {np.nanmax(velocity):.4f}")
    
    return velocity, metadata


def write_velocity_h5(filename, velocity_original, x_samples, y_samples, z_samples, metadata=None):
    """
    Write downsampled velocity to HDF5 file.
    
    Parameters
    ----------
    filename : str
        Output filename
    velocity_original : 2D array
        Original velocity grid
    x_samples, y_samples, z_samples : 1D arrays
        Downsampled points
    metadata : dict, optional
        Metadata to save
    """
    print(f"\nWriting: {filename}")
    
    with h5py.File(filename, 'w') as f:
        # Save original grid for reference
        f.create_dataset('velocity_original', data=velocity_original, compression='gzip')
        
        # Save downsampled points
        f.create_dataset('x_samples', data=x_samples, compression='gzip')
        f.create_dataset('y_samples', data=y_samples, compression='gzip')
        f.create_dataset('z_samples', data=z_samples, compression='gzip')
        
        # Metadata
        f.attrs['n_samples'] = len(x_samples)
        f.attrs['compression_ratio'] = (velocity_original.size) / len(x_samples)
        f.attrs['original_shape'] = velocity_original.shape
        
        # Save additional metadata if provided
        if metadata:
            for key in ['latitude', 'longitude', 'x', 'y']:
                if key in metadata:
                    f.create_dataset(f'metadata_{key}', data=metadata[key], compression='gzip')
        
        print(f"  Saved {len(x_samples)} downsampled points")
        print(f"  Compression ratio: {velocity_original.size / len(x_samples):.2f}x")


def downsample_velocity(input_file, output_file, variance_threshold=0.05, max_depth=7):
    """
    Main downsampling pipeline.
    
    Parameters
    ----------
    input_file : str
        Input velocity.h5 filename
    output_file : str
        Output velocity_downsample.h5 filename
    variance_threshold : float
        Quadtree variance threshold
    max_depth : int
        Maximum quadtree depth
    """
    
    # Read input
    velocity, metadata = read_velocity_h5(input_file)
    
    # Handle NaN values
    velocity_clean = np.nan_to_num(velocity, nan=0.0)
    
    # Quadtree downsampling
    print(f"\n{'='*60}")
    print("QUADTREE DOWNSAMPLING")
    print(f"{'='*60}")
    print(f"Variance threshold: {variance_threshold}")
    print(f"Maximum depth: {max_depth}")
    
    qtree = InSARQuadTree(velocity_clean)
    qtree.subdivide_adaptive(variance_threshold, max_depth=max_depth)
    
    x_samp, y_samp, z_samp = qtree.get_downsampled_data()
    
    print(f"Original points: {velocity.size}")
    print(f"Downsampled points: {len(x_samp)}")
    print(f"Compression ratio: {velocity.size / len(x_samp):.2f}x")
    
    # Write output
    write_velocity_h5(output_file, velocity_clean, x_samp, y_samp, z_samp, metadata)
    
    # Visualization
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(16, 5))
    
    # Original data
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(velocity_clean, cmap="RdBu_r", origin="lower")
    ax1.set_title("Original Velocity Field")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Velocity")
    
    # Quadtree overlay
    ax2 = fig.add_subplot(132)
    ax2.imshow(velocity_clean, cmap="RdBu_r", origin="lower", alpha=0.7)
    qtree.visualize_quadtree(ax=ax2, linewidth=0.3)
    ax2.set_title(f"Quadtree Cells")
    ax2.set_xlabel("x (pixels)")
    ax2.set_ylabel("y (pixels)")
    
    # Downsampled points
    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(x_samp, y_samp, c=z_samp, cmap="RdBu_r", s=50, 
                         edgecolors='k', linewidth=0.5)
    ax3.set_xlim(0, velocity_clean.shape[1])
    ax3.set_ylim(0, velocity_clean.shape[0])
    ax3.set_aspect('equal')
    ax3.set_title(f"Downsampled Points ({len(x_samp)} samples)")
    ax3.set_xlabel("x (pixels)")
    ax3.set_ylabel("y (pixels)")
    cbar3 = plt.colorbar(scatter, ax=ax3)
    cbar3.set_label("Velocity")
    
    plt.tight_layout()
    plt.savefig("downsampling_result.png", dpi=150, bbox_inches='tight')
    print("Saved: downsampling_result.png")
    plt.show()
    
    print(f"\n{'='*60}")
    print("Downsampling completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample InSAR velocity using adaptive quadtree"
    )
    parser.add_argument('--input', type=str, default='velocity.h5',
                       help='Input velocity.h5 filename (default: velocity.h5)')
    parser.add_argument('--output', type=str, default='velocity_downsample.h5',
                       help='Output downsampled filename (default: velocity_downsample.h5)')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Quadtree variance threshold (default: 0.05)')
    parser.add_argument('--max-depth', type=int, default=7,
                       help='Maximum quadtree depth (default: 7)')
    
    args = parser.parse_args()
    
    downsample_velocity(
        input_file=args.input,
        output_file=args.output,
        variance_threshold=args.threshold,
        max_depth=args.max_depth
    )
