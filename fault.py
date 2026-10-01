"""
InSAR Okada Dislocation Inversion using MCMC (Bayesian)

Reads downsampled velocity data and performs Bayesian inversion for an Okada
rectangular dislocation model using Markov Chain Monte Carlo (MCMC).

Output:
- Posterior distributions of parameters (x, y, depth, length, width, strike, slip)
- Parameter estimates with uncertainties
- Acceptance rate and convergence diagnostics
- Posterior samples and statistics saved to files

Usage:
    python fault.py --input velocity_downsample.h5 --output fault_result --n-samples 10000 --burn-in 2000
"""

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm


class OkadaForwardModel:
    """Simplified Okada dislocation forward model."""

    @staticmethod
    def forward(params, xy):
        """
        Simplified Okada model for vertical displacement from a buried rectangular fault.

        Parameters
        ----------
        params : array-like, shape (7,)
            [x, y, depth, length, width, strike, slip]
            x, y : fault center location (pixels)
            depth : fault depth (pixels)
            length : fault length (pixels)
            width : fault width (pixels)
            strike : fault strike angle (degrees)
            slip : dislocation slip amplitude

        xy : array-like, shape (n, 2)
            Observation points [x, y]

        Returns
        -------
        uz : 1D array, shape (n,)
            Vertical displacement at observation points
        """
        x0, y0, depth, length, width, strike, slip = params

        x = xy[:, 0]
        y = xy[:, 1]

        # Relative distance from fault center
        dx = x - x0
        dy = y - y0

        # Convert strike to radians, but keep it simple here
        strike_rad = np.deg2rad(strike)

        # Rotate coordinates to fault-aligned frame
        # x' = x cos(strike) + y sin(strike)
        # y' = -x sin(strike) + y cos(strike)
        xr = dx * np.cos(strike_rad) + dy * np.sin(strike_rad)
        yr = -dx * np.sin(strike_rad) + dy * np.cos(strike_rad)

        # Distance from fault center to observation point
        r2 = xr**2 + yr**2 + depth**2

        # Simplified Okada-like displacement decay
        # approximates a rectangular dislocation as a dipolar source
        uz = (slip * length * width * depth) / (r2**1.5 + 1e-8)

        return uz


class BayesianOkadaInversion:
    """Bayesian inversion for Okada parameters using MCMC."""

    def __init__(self, x_obs, y_obs, z_obs, sigma_data=0.1):
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.z_obs = z_obs
        self.sigma_data = sigma_data
        self.xy_obs = np.column_stack([x_obs, y_obs])
        self.n_data = len(z_obs)

        self.bounds = {
            'x': (np.min(x_obs) - 20, np.max(x_obs) + 20),
            'y': (np.min(y_obs) - 20, np.max(y_obs) + 20),
            'depth': (1.0, 100.0),
            'length': (5.0, 100.0),
            'width': (5.0, 100.0),
            'strike': (0.0, 180.0),
            'slip': (0.01, 10.0),
        }

        self.prior_mean = np.array([
            np.mean(x_obs),
            np.mean(y_obs),
            10.0,
            30.0,
            15.0,
            45.0,
            0.5,
        ])

        self.prior_std = np.array([
            (self.bounds['x'][1] - self.bounds['x'][0]) / 6,
            (self.bounds['y'][1] - self.bounds['y'][0]) / 6,
            25.0,
            25.0,
            20.0,
            30.0,
            2.0,
        ])

        self.chain = None
        self.acceptance_rate = 0.0
        self.log_likelihood_chain = None

    def log_prior(self, params):
        x, y, depth, length, width, strike, slip = params

        if not (self.bounds['x'][0] <= x <= self.bounds['x'][1]):
            return -np.inf
        if not (self.bounds['y'][0] <= y <= self.bounds['y'][1]):
            return -np.inf
        if not (self.bounds['depth'][0] <= depth <= self.bounds['depth'][1]):
            return -np.inf
        if not (self.bounds['length'][0] <= length <= self.bounds['length'][1]):
            return -np.inf
        if not (self.bounds['width'][0] <= width <= self.bounds['width'][1]):
            return -np.inf
        if not (self.bounds['strike'][0] <= strike <= self.bounds['strike'][1]):
            return -np.inf
        if not (self.bounds['slip'][0] <= slip <= self.bounds['slip'][1]):
            return -np.inf

        log_p = 0.0
        for i, p in enumerate(params):
            log_p += norm.logpdf(p, self.prior_mean[i], self.prior_std[i])
        return log_p

    def log_likelihood(self, params):
        try:
            z_pred = OkadaForwardModel.forward(params, self.xy_obs)
            residuals = self.z_obs - z_pred
            log_l = -0.5 * np.sum((residuals / self.sigma_data) ** 2)
            log_l -= self.n_data * np.log(self.sigma_data * np.sqrt(2 * np.pi))
            return log_l
        except:
            return -np.inf

    def log_posterior(self, params):
        log_p = self.log_prior(params)
        if not np.isfinite(log_p):
            return -np.inf
        return log_p + self.log_likelihood(params)

    def run_mcmc(self, n_samples=10000, burn_in=2000, initial_params=None,
                 proposal_scale=None, verbose=True):
        if initial_params is None:
            initial_params = self.prior_mean.copy()

        if proposal_scale is None:
            proposal_scale = self.prior_std * 0.1

        self.chain = np.zeros((n_samples, 7))
        self.log_likelihood_chain = np.zeros(n_samples)

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
            proposal_params = current_params + np.random.normal(0, proposal_scale)
            proposal_ll = self.log_posterior(proposal_params)
            log_alpha = proposal_ll - current_ll

            if np.log(np.random.uniform()) < log_alpha:
                current_params = proposal_params
                current_ll = proposal_ll
                accepted += 1

            self.chain[i] = current_params
            self.log_likelihood_chain[i] = current_ll

            if verbose and (i + 1) % (n_samples // 10) == 0:
                acc_rate = accepted / (i + 1) * 100
                print(f"  Sample {i+1:6d} / {n_samples:6d}   "
                      f"Acceptance rate: {acc_rate:5.2f}%   "
                      f"Log posterior: {current_ll:10.4f}")

        self.acceptance_rate = accepted / n_samples

        if verbose:
            print(f"\nFinal acceptance rate: {self.acceptance_rate*100:.2f}%")
            print(f"{'='*70}\n")

        return self.chain[burn_in:], self.log_likelihood_chain[burn_in:]

    def get_posterior_stats(self, chain):
        param_names = ['x', 'y', 'depth', 'length', 'width', 'strike', 'slip']
        stats = {}

        for i, name in enumerate(param_names):
            samples = chain[:, i]
            stats[name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                'percentile_2_5': np.percentile(samples, 2.5),
                'percentile_97_5': np.percentile(samples, 97.5),
                'min': np.min(samples),
                'max': np.max(samples),
            }

        return stats


def read_downsampled_data(filename):
    print(f"\nReading: {filename}")
    with h5py.File(filename, 'r') as f:
        x = f['x_samples'][:]
        y = f['y_samples'][:]
        z = f['z_samples'][:]
        print(f"  Loaded {len(x)} points")
    return x, y, z


def plot_mcmc_results(chain, x_obs, y_obs, z_obs, output_prefix):
    param_names = ['x', 'y', 'depth', 'length', 'width', 'strike', 'slip']

    fig, axes = plt.subplots(7, 1, figsize=(12, 18))
    fig.suptitle('MCMC Trace Plots (Okada)', fontsize=14, fontweight='bold')
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.plot(chain[:, i], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(name)
        ax.set_xlabel('Sample')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_trace.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Posterior Distributions (Okada)', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.hist(chain[:, i], bins=40, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)

    for j in range(len(param_names), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_posteriors.png", dpi=150, bbox_inches='tight')
    plt.close()

    mean_params = np.mean(chain, axis=0)
    xy_obs = np.column_stack([x_obs, y_obs])
    z_pred = OkadaForwardModel.forward(mean_params, xy_obs)
    residuals = z_obs - z_pred

    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(141)
    sc1 = ax1.scatter(x_obs, y_obs, c=z_obs, cmap='RdBu_r', s=40, edgecolors='k')
    ax1.set_title('Observed Data')
    ax1.set_xlabel('x (pixels)')
    ax1.set_ylabel('y (pixels)')
    ax1.set_aspect('equal')
    plt.colorbar(sc1, ax=ax1, label='Value')

    ax2 = fig.add_subplot(142)
    sc2 = ax2.scatter(x_obs, y_obs, c=z_pred, cmap='RdBu_r', s=40, edgecolors='k')
    ax2.set_title('Predicted Model')
    ax2.set_xlabel('x (pixels)')
    ax2.set_ylabel('y (pixels)')
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2, label='Value')

    ax3 = fig.add_subplot(143)
    vmin = np.min(residuals)
    vmax = np.max(residuals)
    sc3 = ax3.scatter(x_obs, y_obs, c=residuals, cmap='seismic', s=40, edgecolors='k', vmin=vmin, vmax=vmax)
    ax3.set_title('Residuals')
    ax3.set_xlabel('x (pixels)')
    ax3.set_ylabel('y (pixels)')
    ax3.set_aspect('equal')
    plt.colorbar(sc3, ax=ax3, label='Residual')

    ax4 = fig.add_subplot(144)
    ax4.scatter(z_obs, z_pred, alpha=0.7, s=30)
    zmin = min(np.min(z_obs), np.min(z_pred))
    zmax = max(np.max(z_obs), np.max(z_pred))
    ax4.plot([zmin, zmax], [zmin, zmax], 'r--', lw=2, label='1:1')
    ax4.set_xlabel('Observed')
    ax4.set_ylabel('Predicted')
    ax4.set_title('Data vs Model')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_data_model_residual.png", dpi=150, bbox_inches='tight')
    plt.close()


def save_results(chain, output_prefix, acceptance_rate):
    param_names = ['x', 'y', 'depth', 'length', 'width', 'strike', 'slip']

    np.savetxt(f"{output_prefix}_chain.csv", chain, delimiter=',', header=','.join(param_names))

    with open(f"{output_prefix}_parameters.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("OKADA DISLOCATION INVERSION - BAYESIAN MCMC RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Acceptance Rate: {acceptance_rate*100:.2f}%\n")
        f.write(f"Number of posterior samples: {len(chain)}\n\n")

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
            f.write(f"  Min - Max:                  [{np.min(samples):15.6f}, {np.max(samples):15.6f}]\n\n")

    print(f"Saved: {output_prefix}_parameters.txt")


def invert_okada(input_file, output_prefix='fault_result', n_samples=10000,
                burn_in=2000, data_std=0.1):
    x_obs, y_obs, z_obs = read_downsampled_data(input_file)

    print(f"\n{'='*70}")
    print("OKADA DISLOCATION BAYESIAN INVERSION (MCMC)")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    print(f"Number of observations: {len(x_obs)}")
    print(f"Data range: [{np.min(z_obs):.4f}, {np.max(z_obs):.4f}]")
    print(f"Data std (assumed): {data_std:.4f}")

    inversion = BayesianOkadaInversion(x_obs, y_obs, z_obs, sigma_data=data_std)
    chain, _ = inversion.run_mcmc(n_samples=n_samples, burn_in=burn_in, verbose=True)
    stats = inversion.get_posterior_stats(chain)

    print(f"\n{'='*70}")
    print("POSTERIOR STATISTICS")
    print(f"{'='*70}\n")

    for param_name in ['x', 'y', 'depth', 'length', 'width', 'strike', 'slip']:
        s = stats[param_name]
        print(f"{param_name.upper():15s}")
        print(f"  Optimal (Mean):   {s['mean']:12.6f}")
        print(f"  Optimal (Median): {s['median']:12.6f}")
        print(f"  Std Dev:          {s['std']:12.6f}")
        print(f"  95% CI:           [{s['percentile_2_5']:12.6f}, {s['percentile_97_5']:12.6f}]")
        print()

    plot_mcmc_results(chain, x_obs, y_obs, z_obs, output_prefix)
    save_results(chain, output_prefix, inversion.acceptance_rate)

    print(f"\n{'='*70}")
    print("Inversion completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian MCMC inversion for Okada fault source")
    parser.add_argument('--input', type=str, default='velocity_downsample.h5',
                        help='Input downsampled velocity file (default: velocity_downsample.h5)')
    parser.add_argument('--output', type=str, default='fault_result',
                        help='Output prefix for results (default: fault_result)')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of MCMC samples (default: 10000)')
    parser.add_argument('--burn-in', type=int, default=2000,
                        help='Burn-in samples (default: 2000)')
    parser.add_argument('--data-std', type=float, default=0.1,
                        help='Data noise standard deviation (default: 0.1)')

    args = parser.parse_args()

    invert_okada(
        input_file=args.input,
        output_prefix=args.output,
        n_samples=args.n_samples,
        burn_in=args.burn_in,
        data_std=args.data_std
    )
