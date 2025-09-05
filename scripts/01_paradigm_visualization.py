"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
01_paradigm_visualization.py

MAIN OBJECTIVE:
---------------
Create a single comprehensive statistical summary visualization for
multi-frame paradigm analysis.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- seaborn
- scipy

MAIN FEATURES:
--------------
1) Entropy distribution with statistical test
2) Yearly distribution with improved readability
3) Evolution with both Shannon entropy and effective frames
4) Paradigm periods visualization

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import json
from scipy.ndimage import gaussian_filter1d

# Import statsmodels for LOESS smoothing
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available, using gaussian smoothing instead")

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.smoothing_utils import TimeSeriesSmoother

# Configure plotting for professional appearance
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
})

# Define paths
DATA_DIR = project_root / "data" / "01_paradigm_composition"
RESULTS_DIR = project_root / "results" / "01_paradigm_composition"

# Constants
MULTIFRAME_THRESHOLD = np.log(2) / np.log(8)

# Professional color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#73AB84',
    'danger': '#C73E1D',
    'neutral': '#6C757D',
    'light': '#E5E5E5',
    'dark': '#212529'
}


class StatisticalSummaryVisualizer:
    """Create comprehensive statistical summary visualization."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.smoother = TimeSeriesSmoother()
    
    def load_data(self):
        """Load processed data."""
        # Load weekly metrics
        weekly_path = DATA_DIR / "weekly_metrics.csv"
        weekly_df = pd.read_csv(weekly_path)
        weekly_df['week'] = pd.to_datetime(weekly_df['week'])
        
        # Load paradigm periods
        periods_path = DATA_DIR / "paradigm_periods.csv"
        periods_df = pd.read_csv(periods_path)
        periods_df['start_date'] = pd.to_datetime(periods_df['start_date'])
        periods_df['end_date'] = pd.to_datetime(periods_df['end_date'])
        
        # Load statistics
        stats_path = DATA_DIR / "summary_statistics.json"
        with open(stats_path, 'r') as f:
            stats_dict = json.load(f)
        
        return weekly_df, periods_df, stats_dict
    
    def create_statistical_summary(self, weekly_df: pd.DataFrame, 
                                 periods_df: pd.DataFrame, 
                                 stats_dict: dict):
        """Create the comprehensive statistical summary figure."""
        print("Creating statistical summary visualization...")
        
        # Create figure with 4 panels
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1], hspace=0.3, wspace=0.3)
        
        # 1. Entropy distribution with test (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_entropy_distribution(ax1, weekly_df, stats_dict)
        
        # 2. Yearly distribution (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_yearly_distribution(ax2, weekly_df)
        
        # 3. Evolution with dual y-axis (middle, spanning both columns)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_evolution_dual_axis(ax3, weekly_df)
        
        # 4. Paradigm periods (bottom, spanning both columns)
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_paradigm_periods(ax4, weekly_df, periods_df)
        
        # Main title
        fig.suptitle('Figure 1. Is the climate-change paradigm multi-frame?', 
                    fontsize=16, fontweight='bold', y=0.92)
        
        # Save figure
        output_path = RESULTS_DIR / "Figure1_multiframe_test.png"
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"Saved statistical summary to {output_path}")
        plt.close()
    
    def _plot_entropy_distribution(self, ax, weekly_df, stats_dict):
        """Plot entropy distribution with statistical test."""
        # Histogram
        n, bins, patches = ax.hist(weekly_df['shannon_entropy'], bins=40, 
                                  density=True, alpha=0.7, 
                                  color=COLORS['primary'], 
                                  edgecolor='black', linewidth=0.5)
        
        # Color bins based on threshold
        for i, patch in enumerate(patches):
            if bins[i] < MULTIFRAME_THRESHOLD:
                patch.set_facecolor(COLORS['danger'])
                patch.set_alpha(0.6)
            else:
                patch.set_facecolor(COLORS['success'])
                patch.set_alpha(0.6)
        
        # Fit and plot normal distribution
        mu, sigma = stats.norm.fit(weekly_df['shannon_entropy'])
        x = np.linspace(0, 1, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 
               'r-', linewidth=2, label=f'Normal fit\nμ={mu:.3f}, σ={sigma:.3f}')
        
        # Add key lines
        median = stats_dict['median_entropy']
        ax.axvline(x=median, color=COLORS['dark'], 
                  linestyle='-', linewidth=2.5, label=f'Median: {median:.3f}')
        ax.axvline(x=MULTIFRAME_THRESHOLD, color=COLORS['accent'], 
                  linestyle='--', linewidth=2, label=f'Threshold: {MULTIFRAME_THRESHOLD:.3f}')
        
        # Add CI
        ax.axvspan(stats_dict['ci_lower'], stats_dict['ci_upper'], 
                  alpha=0.2, color=COLORS['primary'], 
                  label=f"95% CI: [{stats_dict['ci_lower']:.3f}, {stats_dict['ci_upper']:.3f}]")
        
        # Add test result
        paradigm = "multi-frame" if stats_dict['is_multiframe_overall'] else "mono-frame"
        p_val = stats_dict['p_value']
        
        test_text = f"Paradigm: {paradigm}\np-value: {p_val:.4f}"
        ax.text(0.05, 0.95, test_text, transform=ax.transAxes,
               fontsize=11, va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        ax.set_xlabel('Shannon Entropy')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Weekly Shannon Entropy', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, 1.0)
    
    def _plot_yearly_distribution(self, ax, weekly_df):
        """Plot yearly distribution with improved readability."""
        # Prepare data for box plot
        yearly_data = []
        years = []
        
        for year in sorted(weekly_df['year'].unique()):
            year_data = weekly_df[weekly_df['year'] == year]
            if len(year_data) >= 1:  # Include all years with at least 1 data point
                yearly_data.append(year_data['shannon_entropy'].values)
                years.append(str(year))
        
        # Create box plot
        bp = ax.boxplot(yearly_data, labels=years, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color boxes based on median
        for i, (box, year) in enumerate(zip(bp['boxes'], years)):
            year_median = np.median(yearly_data[i])
            if year_median > MULTIFRAME_THRESHOLD:
                box.set_facecolor(COLORS['success'])
                box.set_alpha(0.7)
            else:
                box.set_facecolor(COLORS['danger'])
                box.set_alpha(0.7)
        
        # Style the plot
        for element in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp[element], color=COLORS['dark'], linewidth=1)
        plt.setp(bp['medians'], color=COLORS['dark'], linewidth=2)
        plt.setp(bp['means'], color=COLORS['accent'], linewidth=2)
        
        # Add threshold line
        ax.axhline(y=MULTIFRAME_THRESHOLD, color=COLORS['accent'], 
                  linestyle='--', linewidth=2, label='Multi-frame threshold')
        
        # Improve x-axis readability
        ax.set_ylabel('Shannon Entropy')
        ax.set_title('Yearly Distribution', fontweight='bold')
        
        # Show all years with adaptive intervals
        n_years = len(years)
        if n_years <= 10:
            # Show all years if 10 or fewer
            ax.set_xticklabels(years, rotation=45, ha='right', fontsize=8)
        elif n_years <= 20:
            # Show every other year for 11-20 years
            visible_years = [years[i] if i % 2 == 0 else '' for i in range(len(years))]
            ax.set_xticklabels(visible_years, rotation=45, ha='right', fontsize=8)
        elif n_years <= 40:
            # Show every 5th year for 21-40 years
            visible_years = [years[i] if i % 5 == 0 else '' for i in range(len(years))]
            ax.set_xticklabels(visible_years, rotation=45, ha='right', fontsize=7)
        else:
            # Show every 10th year for more than 40 years
            visible_years = [years[i] if i % 10 == 0 else '' for i in range(len(years))]
            ax.set_xticklabels(visible_years, rotation=45, ha='right', fontsize=7)
        
        # Add grid only on y-axis
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add legend
        ax.legend(['Median', 'Mean', 'Threshold'], loc='upper right', fontsize=8)
    
    def _plot_evolution_dual_axis(self, ax, weekly_df):
        """Plot evolution with both Shannon entropy and effective frames using LOESS."""
        # Create dual y-axis
        ax2 = ax.twinx()
        
        # Plot raw data as scatter (very light)
        ax.scatter(weekly_df['week'], weekly_df['shannon_entropy'], 
                  alpha=0.1, s=5, color=COLORS['light'], rasterized=True)
        
        ax2.scatter(weekly_df['week'], weekly_df['effective_frames'], 
                   alpha=0.1, s=5, color=COLORS['light'], rasterized=True)
        
        # Convert dates to numeric for LOESS
        x_numeric = pd.to_numeric(weekly_df['week']) / 1e9  # Convert to seconds since epoch
        
        # Apply LOESS smoothing with confidence intervals
        if HAS_STATSMODELS and len(weekly_df) > 20:
            # LOESS for Shannon entropy
            frac = 0.15  # Fraction of data for smoothing window
            shannon_loess = lowess(weekly_df['shannon_entropy'].values, x_numeric, 
                                  frac=frac, return_sorted=False)
            
            # Bootstrap for confidence intervals
            n_bootstrap = 100
            shannon_bootstrap = np.zeros((n_bootstrap, len(weekly_df)))
            for i in range(n_bootstrap):
                indices = np.random.choice(len(weekly_df), len(weekly_df), replace=True)
                y_resample = weekly_df['shannon_entropy'].values[indices]
                x_resample = x_numeric[indices]
                smooth_resample = lowess(y_resample, x_resample, frac=frac, xvals=x_numeric)
                shannon_bootstrap[i] = smooth_resample
            
            shannon_ci_lower = np.percentile(shannon_bootstrap, 2.5, axis=0)
            shannon_ci_upper = np.percentile(shannon_bootstrap, 97.5, axis=0)
            
            # LOESS for effective frames
            effective_loess = lowess(weekly_df['effective_frames'].values, x_numeric,
                                   frac=frac, return_sorted=False)
            
            # Bootstrap for effective frames
            effective_bootstrap = np.zeros((n_bootstrap, len(weekly_df)))
            for i in range(n_bootstrap):
                indices = np.random.choice(len(weekly_df), len(weekly_df), replace=True)
                y_resample = weekly_df['effective_frames'].values[indices]
                x_resample = x_numeric[indices]
                smooth_resample = lowess(y_resample, x_resample, frac=frac, xvals=x_numeric)
                effective_bootstrap[i] = smooth_resample
            
            effective_ci_lower = np.percentile(effective_bootstrap, 2.5, axis=0)
            effective_ci_upper = np.percentile(effective_bootstrap, 97.5, axis=0)
            
            shannon_smooth = shannon_loess
            effective_smooth = effective_loess
        else:
            # Fallback to original smoothing
            shannon_smooth = self.smoother.optimal_smooth_for_visualization(
                weekly_df['shannon_entropy'], smoothness='medium'
            )
            effective_smooth = self.smoother.optimal_smooth_for_visualization(
                weekly_df['effective_frames'], smoothness='medium'
            )
            
            # Simple confidence intervals based on rolling std
            window = 12
            shannon_std = weekly_df['shannon_entropy'].rolling(window=window, center=True).std().fillna(0.02)
            shannon_ci_lower = shannon_smooth - 1.96 * shannon_std
            shannon_ci_upper = shannon_smooth + 1.96 * shannon_std
            
            effective_std = weekly_df['effective_frames'].rolling(window=window, center=True).std().fillna(0.2)
            effective_ci_lower = effective_smooth - 1.96 * effective_std
            effective_ci_upper = effective_smooth + 1.96 * effective_std
        
        # Plot confidence intervals
        ax.fill_between(weekly_df['week'], shannon_ci_lower, shannon_ci_upper,
                       color=COLORS['primary'], alpha=0.2)
        ax2.fill_between(weekly_df['week'], effective_ci_lower, effective_ci_upper,
                        color=COLORS['success'], alpha=0.2)
        
        # Plot smoothed curves
        line1 = ax.plot(weekly_df['week'], shannon_smooth, 
                       color=COLORS['primary'], linewidth=2.5, 
                       label='Shannon Entropy')[0]
        
        line2 = ax2.plot(weekly_df['week'], effective_smooth, 
                        color=COLORS['success'], linewidth=2.5, 
                        label='Effective Frames')[0]
        
        # Add reference lines
        ax.axhline(y=MULTIFRAME_THRESHOLD, color=COLORS['accent'], 
                  linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=2.5, color=COLORS['accent'], 
                   linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Labels and styling
        ax.set_xlabel('Date')
        ax.set_ylabel('Shannon Entropy')
        ax2.set_ylabel('Effective Number of Frames')
        ax.set_title('Evolution of Paradigm Diversity Metrics', fontweight='bold')
        
        # Keep y-axis labels in black (remove color)
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Set limits
        ax.set_ylim(0, 0.85)
        ax2.set_ylim(0, 6.5)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        # Combined legend
        lines = [line1, line2]
        labels = ['Shannon Entropy', 'Effective Frames']
        ax.legend(lines, labels, loc='upper left', fontsize=9)
        
        # Add annotations for thresholds
        ax.text(weekly_df['week'].min() + pd.Timedelta(days=365), MULTIFRAME_THRESHOLD + 0.02, 
               'Multi-frame\nthreshold', ha='left', va='bottom', 
               fontsize=8, color=COLORS['accent'])
        ax2.text(weekly_df['week'].max() - pd.Timedelta(days=365), 2.5 + 0.1, 
                '2.5 frames', ha='right', va='bottom', 
                fontsize=8, color=COLORS['accent'])
    
    def _plot_paradigm_periods(self, ax, weekly_df, periods_df):
        """Plot paradigm periods as colored bands with uncertainty zone."""
        # Find the transition point from mono-frame to multi-frame
        transition_date = None
        for i in range(len(periods_df) - 1):
            current_period = periods_df.iloc[i]
            next_period = periods_df.iloc[i + 1]
            if not current_period['is_multiframe'] and next_period['is_multiframe']:
                transition_date = current_period['end_date']
                break
        
        # Plot periods as colored spans
        for _, period in periods_df.iterrows():
            color = COLORS['success'] if period['is_multiframe'] else COLORS['danger']
            ax.axvspan(period['start_date'], period['end_date'], 
                      alpha=0.3, color=color)
        
        # Add uncertainty zone around transition
        if transition_date:
            # Create uncertainty zone: ±6 months around transition
            uncertainty_start = transition_date - pd.Timedelta(days=180)
            uncertainty_end = transition_date + pd.Timedelta(days=180)
            
            # Add hatched rectangle for uncertainty
            ax.axvspan(uncertainty_start, uncertainty_end,
                      alpha=0.2, color='gray', hatch='///', edgecolor='gray',
                      linewidth=1, label='Transition uncertainty')
            
            # Add text annotation
            ax.text(transition_date, 0.35, 'Transition\nuncertainty',
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.8))
        
        # Styling
        ax.set_xlabel('Date')
        ax.set_ylabel('')
        ax.set_title('Paradigm Periods (Mono-frame vs Multi-frame)', fontweight='bold')
        ax.set_ylim(0, 0.7)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['danger'], alpha=0.3, label='Mono-frame periods'),
            Patch(facecolor=COLORS['success'], alpha=0.3, label='Multi-frame periods')
        ]
        # Add uncertainty zone to legend if it exists
        if transition_date:
            legend_elements.append(
                Patch(facecolor='gray', alpha=0.2, hatch='///', 
                     edgecolor='gray', label='Transition uncertainty')
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    def run_visualization(self):
        """Run the visualization process."""
        print("="*60)
        print("CREATING STATISTICAL SUMMARY VISUALIZATION")
        print("="*60)
        
        # Load data
        weekly_df, periods_df, stats_dict = self.load_data()
        
        # Create visualization
        self.create_statistical_summary(weekly_df, periods_df, stats_dict)
        
        print("\nVisualization complete!")
        print(f"Output saved to: {RESULTS_DIR / 'Figure1_multiframe_test.png'}")


def main():
    """Main execution function."""
    visualizer = StatisticalSummaryVisualizer()
    visualizer.run_visualization()


if __name__ == "__main__":
    main()