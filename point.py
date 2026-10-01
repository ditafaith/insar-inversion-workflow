"""
InSAR Mogi Point Source Inversion using MCMC (Bayesian)

Reads downsampled velocity data and performs Bayesian inversion for Mogi
point source parameters using Markov Chain Monte Carlo (MCMC).

Output:
- Posterior distributions of parameters (x, y, depth, volume_change)
- Parameter estimates with uncertainties
- Acceptance rate and convergence diagnostics
- Posterior samples and statistics saved to files

Usage:
    python point.py --input velocity_downsample.h5 --output point_result --n-samples 10000 --burn-in 2000
"""

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm
import json


class MogiForwardModel:
    """Mogi point source forward model."""
    
    @staticmethod
    def forward(params, xy):
        """
        Compute vertical displacement from Mogi point source.
        
        Parameters
        ----------
        params : array-like, shape (4,)
            [x, y, depth, volume_change]
            x, y : source location (pixels)
            depth : source depth (pixels)
            volume_change : volume change (arbitrary units)
        
        xy : array-like, shape (n, 2)
            Observation points [x, y]
        
        Returns
        -------
        uz : 1D array, shape (n,)
            Vertical displacement at observation points
        """
        x0, y0, depth, volume = params
        
        x = xy[:, 0]
        y = xy[:, 1]
        
        # Distance from source to observation points
        r = np.sqrt((x - x0)**2 + (y - y0)**2 + depth**2)
        
        # Mogi solution: uz = (c * dV) / r^3
        # c is a scaling coefficient
        c = 1.0  # Can be adjusted based on physical units
        uz = c * volume / (r**3)
        
        return uz


class BayesianMogiInversion:
    """Bayesian inversion for Mogi parameters using MCMC."""
    
    def __init__(self, x_obs, y_obs, z_obs, sigma_data=0.1):
        """
        Initialize MCMC inversion.
        
        Parameters
        ----------
        x_obs, y_obs : 1D arrays
            Observation point coordinates
        z_obs : 1D array
            Observed displacement/velocity
        sigma_data : float
            Data noise standard deviation
        """
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.z_obs = z_obs
        self.sigma_data = sigma_data
        
        self.xy_obs = np.column_stack([x_obs, y_obs])
        self.n_data = len(z_obs)
        
        # Parameter bounds for prior
        self.bounds = {
            'x': (np.min(x_obs) - 20, np.max(x_obs) + 20),
            'y': (np.min(y_obs) - 20, np.max(y_obs) + 20),
            'depth': (1.0, 100.0),
            'volume': (0.01, 100.0)
        }
        
        # Prior means and standard deviations
        self.prior_mean = np.array([
            np.mean(x_obs),
            np.mean(y_obs),
            10.0,
            1.0
        ])
        
        self.prior_std = np.array([
            (self.bounds['x'][1] - self.bounds['x'][0]) / 6,
            (self.bounds['y'][1] - self.bounds['y'][0]) / 6,
            30.0,
            10.0
        ])
        
        # Storage for MCMC chain
        self.chain = None
        self.acceptance_rate = 0.0
        self.log_likelihood_chain = None
    
    def log_prior(self, params):
        """
        Log prior probability.
        
        Uses Gaussian priors centered at mean with large standard deviations.
        """
        x, y, depth, volume = params
        
        # Check bounds
        if not (self.bounds['x'][0] <= x <= self.bounds['x'][1]):
            return -np.inf
        if not (self.bounds['y'][0] <= y <= self.bounds['y'][1]):
            return -np.inf
        if not (self.bounds['depth'][0] <= depth <= self.bounds['depth'][1]):
            return -np.inf
        if not (self.bounds['volume'][0] <= volume <= self.bounds['volume'][1]):
            return -np.inf
        
        # Gaussian prior
        log_p = 0.0
        for i, p in enumerate(params):
            log_p += norm.logpdf(p, self.prior_mean[i], self.prior_std[i])
        
        return log_p
    
    def log_likelihood(self, params):
        """
        Log likelihood of data given model.
        
        Assumes Gaussian errors with standard deviation sigma_data.
        """
        try:
            # Forward model
            z_pred = MogiForwardModel.forward(params, self.xy_obs)
            
            # Residuals
            residuals = self.z_obs - z_pred
            
            # Log likelihood (Gaussian)
            log_l = -0.5 * np.sum((residuals / self.sigma_data) ** 2)
            log_l -= self.n_data * np.log(self.sigma_data * np.sqrt(2 * np.pi))
            
            return log_l
        except:
            return -np.inf
    
    def log_posterior(self, params):
        """Log posterior probability (prior + likelihood)."""
        log_p = self.log_prior(params)
        if not np.isfinite(log_p):
            return -np.inf
        
        log_l = self.log_likelihood(params)
        return log_p + log_l
    
    def run_mcmc(self, n_samples=10000, burn_in=2000, initial_params=None, 
                 proposal_scale=None, verbose=True):
        """
        Run Metropolis-Hastings MCMC algorithm.
        
        Parameters
        ----------
        n_samples : int
            Total number of MCMC samples
        burn_in : int
            Number of burn-in samples to discard
        initial_params : array-like, shape (4,), optional
            Initial parameter guess
        proposal_scale : array-like, shape (4,), optional
            Proposal step standard deviations
        verbose : bool
            Print progress information
        
        Returns
        -------
        chain : 2D array, shape (n_samples - burn_in, 4)
            MCMC posterior samples
        """
        if initial_params is None:
            initial_params = self.prior_mean.copy()
        
        if proposal_scale is None:
            proposal_scale = self.prior_std * 0.1
        
        # Initialize chain storage
        self.chain = np.zeros((n_samples, 4))
        self.log_likelihood_chain = np.zeros(n_samples)
        
        # Initial state
        current_params = initial_params.copy()
        current_ll = self.log_posterior(current_params)
        
        accepted = 0
        
        if verbose:
            print(f"\n{'='*70}")
            print("MCMC SAMPLING (Metropolis-Hastings)")
            print(f"{'='*70}")
            print(f"Total samples: {n_samples}")
            print(f"Burn-in: {burn_in}")
            print(f"Proposal scale: {proposal_scale}")
            print(f"Initial log posterior: {current_ll:.4f}\n")
        
        for i in range(n_samples):
            # Propose new parameters
            proposal_params = current_params + np.random.normal(0, proposal_scale)
            proposal_ll = self.log_posterior(proposal_params)
            
            # Metropolis-Hastings acceptance ratio
            log_alpha = proposal_ll - current_ll
            
            # Accept or reject
            if np.log(np.random.uniform()) < log_alpha:
                current_params = proposal_params
                current_ll = proposal_ll
                accepted += 1
            
            # Store in chain
            self.chain[i] = current_params
            self.log_likelihood_chain[i] = current_ll
            
            # Progress report
            if verbose and (i + 1) % (n_samples // 10) == 0:
                acc_rate = accepted / (i + 1) * 100
                print(f"  Sample {i+1:6d} / {n_samples:6d}   "
                      f"Acceptance rate: {acc_rate:5.2f}%   "
                      f"Log posterior: {current_ll:10.4f}")
        
        self.acceptance_rate = accepted / n_samples
        
        if verbose:
            print(f"\nFinal acceptance rate: {self.acceptance_rate*100:.2f}%")
            print(f"{'='*70}\n")
        
        # Return chain after burn-in
        return self.chain[burn_in:], self.log_likelihood_chain[burn_in:]
    
    def get_posterior_stats(self, chain):
        """
        Compute posterior statistics.
        
        Parameters
        ----------
        chain : 2D array, shape (n_samples, 4)
            MCMC posterior samples
        
        Returns
        -------
        stats : dict
            Dictionary containing mean, std, and percentiles
        """
        param_names = ['x', 'y', 'depth', 'volume_change']
        
        stats = {}
        for i, name in enumerate(param_names):
            samples = chain[:, i]
            stats[name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                'percentile_2_5': np.percentile(samples, 2.5),
                'percentile_97_5': np.percentile(samples, 97.5),
                'percentile_16': np.percentile(samples, 16),
                'percentile_84': np.percentile(samples, 84),
                'min': np.min(samples),
                'max': np.max(samples),
            }
        
        return stats


def read_downsampled_data(filename):
    """
    Read downsampled velocity data from HDF5 file.
    
    Parameters
    ----------
    filename : str
        Path to velocity_downsample.h5
    
    Returns
    -------
    x, y, z : 1D arrays
        Downsampled coordinates and values
    """
    print(f"\nReading: {filename}")
    
    with h5py.File(filename, 'r') as f:
        x = f['x_samples'][:]
        y = f['y_samples'][:]
        z = f['z_samples'][:]
        
        print(f"  Loaded {len(x)} points")
    
    return x, y, z


def plot_mcmc_results(chain, log_likelihood, x_obs, y_obs, z_obs, output_prefix):
    """
    Plot MCMC results including trace plots, posteriors, and forward model.
    
    Parameters
    ----------
    chain : 2D array, shape (n_samples, 4)
        MCMC posterior samples
    log_likelihood : 1D array
        Log likelihood values
    x_obs, y_obs, z_obs : 1D arrays
        Observation points
    output_prefix : str
        Prefix for output filenames
    """
    param_names = ['x', 'y', 'depth', 'volume_change']
    
    # Figure 1: Trace plots (convergence diagnostics)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('MCMC Trace Plots (Convergence Diagnostics)', fontsize=14, fontweight='bold')
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.plot(chain[:, i], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(name)
        ax.set_xlabel('Sample')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_trace.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_trace.png")
    plt.close()
    
    # Figure 2: Posterior distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Posterior Distributions', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.hist(chain[:, i], bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel(name)
        ax.set_ylabel('Probability density')
        ax.grid(True, alpha=0.3)
        
        # Add mean and median lines
        mean_val = np.mean(chain[:, i])
        median_val = np.median(chain[:, i])
        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_posteriors.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_posteriors.png")
    plt.close()
    
    # Figure 3: Data, model, residuals
    # Use mean posterior parameters
    mean_params = np.mean(chain, axis=0)
    xy_obs = np.column_stack([x_obs, y_obs])
    z_pred = MogiForwardModel.forward(mean_params, xy_obs)
    residuals = z_obs - z_pred
    
    fig = plt.figure(figsize=(16, 5))
    
    # Data scatter
    ax1 = fig.add_subplot(141)
    scatter1 = ax1.scatter(x_obs, y_obs, c=z_obs, cmap='RdBu_r', s=50, edgecolors='k', linewidth=0.5)
    ax1.set_xlabel('x (pixels)')
    ax1.set_ylabel('y (pixels)')
    ax1.set_title('Observed Data')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='Value')
    
    # Model prediction
    ax2 = fig.add_subplot(142)
    scatter2 = ax2.scatter(x_obs, y_obs, c=z_pred, cmap='RdBu_r', s=50, edgecolors='k', linewidth=0.5)
    ax2.plot(mean_params[0], mean_params[1], 'r*', markersize=20, label='Mogi source')
    ax2.set_xlabel('x (pixels)')
    ax2.set_ylabel('y (pixels)')
    ax2.set_title('Predicted Model')
    ax2.set_aspect('equal')
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Value')
    
    # Residuals
    ax3 = fig.add_subplot(143)
    vmax = np.max(np.abs(residuals))
    scatter3 = ax3.scatter(x_obs, y_obs, c=residuals, cmap='seismic', s=50, 
                          vmin=-vmax, vmax=vmax, edgecolors='k', linewidth=0.5)
    ax3.set_xlabel('x (pixels)')
    ax3.set_ylabel('y (pixels)')
    ax3.set_title('Residuals')
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3, label='Residual')
    
    # Data vs model scatter plot
    ax4 = fig.add_subplot(144)
    ax4.scatter(z_obs, z_pred, alpha=0.6, s=30)
    z_min = min(np.min(z_obs), np.min(z_pred))
    z_max = max(np.max(z_obs), np.max(z_pred))
    ax4.plot([z_min, z_max], [z_min, z_max], 'r--', lw=2, label='Perfect fit')
    ax4.set_xlabel('Observed')
    ax4.set_ylabel('Predicted')
    ax4.set_title('Data vs. Model Fit')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Compute RMS misfit
    rms_misfit = np.sqrt(np.mean(residuals**2))
    chi2 = np.sum((residuals**2))
    
    fig.text(0.5, -0.02, f'RMS Misfit: {rms_misfit:.4f}   Chi²: {chi2:.4f}', 
            ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_data_model_residual.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_data_model_residual.png")
    plt.close()


def save_results(chain, output_prefix, acceptance_rate):
    """
    Save MCMC results and statistics to text files.
    
    Parameters
    ----------
    chain : 2D array, shape (n_samples, 4)
        MCMC posterior samples
    output_prefix : str
        Prefix for output filenames
    acceptance_rate : float
        MCMC acceptance rate
    """
    param_names = ['x', 'y', 'depth', 'volume_change']
    
    # Save chain to CSV
    np.savetxt(f"{output_prefix}_chain.csv", chain, 
              delimiter=',', header=','.join(param_names))
    print(f"Saved: {output_prefix}_chain.csv")
    
    # Save comprehensive summary statistics to single file
    with open(f"{output_prefix}_parameters.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("MOGI POINT SOURCE INVERSION - BAYESIAN MCMC RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Acceptance Rate: {acceptance_rate*100:.2f}%\n")
        f.write(f"Number of posterior samples: {len(chain)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PARAMETER ESTIMATES\n")
        f.write("="*80 + "\n\n")
        
        for i, name in enumerate(param_names):
            samples = chain[:, i]
            mean = np.mean(samples)
            median = np.median(samples)
            p2_5 = np.percentile(samples, 2.5)
            p97_5 = np.percentile(samples, 97.5)
            std = np.std(samples)
            
            f.write(f"{name.upper()}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"  Optimal (Mean):             {mean:15.6f}\n")
            f.write(f"  Optimal (Median):           {median:15.6f}\n")
            f.write(f"  Std Dev:                    {std:15.6f}\n")
            f.write(f"  95% Confidence Interval:    [{p2_5:15.6f}, {p97_5:15.6f}]\n")
            f.write(f"  Min - Max:                  [{np.min(samples):15.6f}, {np.max(samples):15.6f}]\n")
            f.write("\n")
    
    print(f"Saved: {output_prefix}_parameters.txt")


def invert_mogi(input_file, output_prefix="point_result", n_samples=10000, 
                burn_in=2000, data_std=0.1):
    """
    Main inversion pipeline for Mogi point source.
    
    Parameters
    ----------
    input_file : str
        Path to velocity_downsample.h5
    output_prefix : str
        Prefix for output files
    n_samples : int
        Number of MCMC samples
    burn_in : int
        Burn-in samples to discard
    data_std : float
        Data noise standard deviation
    """
    
    # Read data
    x_obs, y_obs, z_obs = read_downsampled_data(input_file)
    
    print(f"\n{'='*70}")
    print("MOGI POINT SOURCE BAYESIAN INVERSION (MCMC)")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    print(f"Number of observations: {len(x_obs)}")
    print(f"Data range: [{np.min(z_obs):.4f}, {np.max(z_obs):.4f}]")
    print(f"Data std (assumed): {data_std:.4f}")
    
    # Initialize inversion
    inversion = BayesianMogiInversion(x_obs, y_obs, z_obs, sigma_data=data_std)
    
    # Run MCMC
    chain, log_likelihood = inversion.run_mcmc(
        n_samples=n_samples,
        burn_in=burn_in,
        verbose=True
    )
    
    # Get statistics
    stats = inversion.get_posterior_stats(chain)
    
    print(f"\n{'='*70}")
    print("POSTERIOR STATISTICS")
    print(f"{'='*70}\n")
    
    for param_name in ['x', 'y', 'depth', 'volume_change']:
        s = stats[param_name]
        print(f"{param_name.upper():15s}")
        print(f"  Optimal (Mean):   {s['mean']:12.6f}")
        print(f"  Optimal (Median): {s['median']:12.6f}")
        print(f"  Std Dev:          {s['std']:12.6f}")
        print(f"  95% CI:           [{s['percentile_2_5']:12.6f}, {s['percentile_97_5']:12.6f}]")
        print()
    
    # Plot results
    print(f"\nGenerating plots...")
    plot_mcmc_results(chain, log_likelihood, x_obs, y_obs, z_obs, output_prefix)
    
    # Save results
    print(f"\nSaving results...")
    save_results(chain, output_prefix, inversion.acceptance_rate)
    
    print(f"\n{'='*70}")
    print("Inversion completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian MCMC inversion for Mogi point source"
    )
    parser.add_argument('--input', type=str, default='velocity_downsample.h5',
                       help='Input downsampled velocity file (default: velocity_downsample.h5)')
    parser.add_argument('--output', type=str, default='point_result',
                       help='Output prefix for results (default: point_result)')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of MCMC samples (default: 10000)')
    parser.add_argument('--burn-in', type=int, default=2000,
                       help='Burn-in samples (default: 2000)')
    parser.add_argument('--data-std', type=float, default=0.1,
                       help='Data noise standard deviation (default: 0.1)')
    
    args = parser.parse_args()
    
    invert_mogi(
        input_file=args.input,
        output_prefix=args.output,
        n_samples=args.n_samples,
        burn_in=args.burn_in,
        data_std=args.data_std
    )
