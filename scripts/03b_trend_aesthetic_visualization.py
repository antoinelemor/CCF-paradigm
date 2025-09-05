"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
03c_trend_aesthetic_visualization.py

MAIN OBJECTIVE:
---------------
Create aesthetic and scientific visualizations for trend analysis with monthly smoothing
and clear indicators of increase/decrease periods by significance level.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- seaborn
- scipy

MAIN FEATURES:
--------------
1) Monthly smoothed curves for each frame
2) Arrows indicating trend directions
3) Three-panel visualization by significance level
4) Professional scientific aesthetics
5) Clear period demarcation

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PatchCollection
import seaborn as sns
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Constants
FRAME_NAMES = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]
FRAME_COLORS = {
    "Cult": "#E64B35",
    "Eco": "#4DBBD5",
    "Envt": "#00A087",
    "Pbh": "#3C5488",
    "Just": "#F39B7F",
    "Pol": "#8491B4",
    "Sci": "#91D1C2",
    "Secu": "#B09C85",
}

# Paths
DATA_DIR = project_root / "data" / "03_events_effects" / "trends_data"
RESULTS_DIR = project_root / "results" / "03_events_effects" / "trends_results"
AESTHETIC_DIR = RESULTS_DIR / "aesthetic_plots"

# Create directories
AESTHETIC_DIR.mkdir(parents=True, exist_ok=True)

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


class AestheticTrendVisualizer:
    """Create aesthetic scientific visualizations for trend analysis."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.trends_df = None
        self.weekly_data = None
        self.monthly_data = None
        
    def load_data(self):
        """Load and prepare data."""
        print("Loading data for aesthetic visualization...")
        
        # Load trends
        self.trends_df = pd.read_csv(DATA_DIR / "all_trends_emd.csv")
        self.trends_df['start_date'] = pd.to_datetime(self.trends_df['start_date'])
        self.trends_df['end_date'] = pd.to_datetime(self.trends_df['end_date'])
        
        # Load weekly data
        self.weekly_data = pd.read_csv(DATA_DIR / "raw_weekly_proportions.csv")
        self.weekly_data['week'] = pd.to_datetime(self.weekly_data['week'])
        
        # Calculate monthly aggregates for smoother visualization
        self._calculate_monthly_data()
        
        print(f"  Loaded {len(self.trends_df)} trends")
        print(f"  Calculated {len(self.monthly_data)} months of data")
    
    def _calculate_monthly_data(self):
        """Calculate monthly aggregates from weekly data."""
        # Convert to monthly
        self.weekly_data['month'] = self.weekly_data['week'].dt.to_period('M').dt.to_timestamp()
        
        # Group by month and calculate mean proportions
        prop_cols = [f'{frame}_prop' for frame in FRAME_NAMES]
        self.monthly_data = self.weekly_data.groupby('month')[prop_cols].mean().reset_index()
        
        # Add sample size for each month
        monthly_counts = self.weekly_data.groupby('month').size().reset_index(name='n_weeks')
        self.monthly_data = self.monthly_data.merge(monthly_counts, on='month')
    
    def create_frame_significance_plots(self):
        """Create three-panel plots by significance level for each frame."""
        print("\nCreating aesthetic significance plots...")
        
        for frame in FRAME_NAMES:
            print(f"  Creating plot for {frame}...")
            self._create_single_frame_plot(frame)
    
    def _create_single_frame_plot(self, frame: str):
        """Create aesthetic three-panel plot for a single frame."""
        # Create figure with three panels
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 1, figure=fig, hspace=0.25, height_ratios=[1, 1, 1])
        
        # Get frame data
        frame_trends = self.trends_df[self.trends_df['frame'] == frame]
        frame_monthly = self.monthly_data.set_index('month')[f'{frame}_prop']
        
        # Define significance groups
        significance_groups = [
            ('very_high', 'Very High Significance (p < 0.001)', 0),
            ('high', 'High to Moderate Significance (0.001 < p < 0.05)', 1),
            ('low', 'Low Significance (p > 0.05)', 2)
        ]
        
        for sig_level, title, panel_idx in significance_groups:
            ax = fig.add_subplot(gs[panel_idx])
            
            if sig_level == 'very_high':
                trends = frame_trends[frame_trends['significance_level'] == 'very_high']
            elif sig_level == 'high':
                trends = frame_trends[frame_trends['significance_level'].isin(['high', 'moderate'])]
            else:  # low
                trends = frame_trends[frame_trends['significance_level'] == 'low']
            
            self._plot_panel(ax, frame_monthly, trends, frame, title)
        
        # Main title
        fig.suptitle(f'{frame} Frame - Trend Analysis by Significance Level\nMonthly Smoothed Data with Directional Indicators',
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_path = AESTHETIC_DIR / f"trends_{frame}_by_significance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    Saved to {output_path}")
    
    def _plot_panel(self, ax, monthly_series, trends, frame, title):
        """Plot a single panel with monthly smoothed data and trend indicators."""
        # Create smooth interpolation for aesthetic curve
        if len(monthly_series) > 3:
            # Apply additional smoothing for visual appeal
            smoothed_values = gaussian_filter1d(monthly_series.values, sigma=2)
            
            # Plot smoothed line
            ax.plot(monthly_series.index, smoothed_values, 
                   color=FRAME_COLORS[frame], linewidth=2.5, 
                   label='Monthly smoothed', zorder=5)
            
            # Add confidence interval
            rolling_std = pd.Series(monthly_series.values).rolling(window=12, center=True).std()
            rolling_std = rolling_std.fillna(rolling_std.mean())
            
            ax.fill_between(monthly_series.index, 
                          smoothed_values - rolling_std,
                          smoothed_values + rolling_std,
                          color=FRAME_COLORS[frame], alpha=0.15, zorder=1)
        
        # Plot raw monthly data as subtle points
        ax.scatter(monthly_series.index, monthly_series.values,
                  color=FRAME_COLORS[frame], s=10, alpha=0.3, zorder=2)
        
        # Add trend periods as shaded regions and arrows
        for _, trend in trends.iterrows():
            self._add_trend_visualization(ax, trend, monthly_series)
        
        # Styling
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_xlabel('Year', fontsize=10)
        
        # Grid
        ax.grid(True, alpha=0.2, linewidth=0.5, linestyle='--')
        ax.set_axisbelow(True)
        
        # X-axis formatting
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        
        # Y-axis formatting
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        
        # Set limits with padding
        ax.set_xlim(monthly_series.index.min() - pd.Timedelta(days=180),
                   monthly_series.index.max() + pd.Timedelta(days=180))
        
        y_min, y_max = monthly_series.min(), monthly_series.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.2 * y_range)
        
        # Add trend count annotation
        n_trends = len(trends)
        ax.text(0.02, 0.98, f'Trends: {n_trends}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Legend
        if n_trends > 0:
            # Create custom legend
            legend_elements = [
                plt.Line2D([0], [0], color=FRAME_COLORS[frame], linewidth=2.5, label='Monthly smoothed'),
                mpatches.Patch(facecolor='green', alpha=0.3, edgecolor='green', label='Increasing period'),
                mpatches.Patch(facecolor='red', alpha=0.3, edgecolor='red', label='Decreasing period')
            ]
            ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=8)
    
    def _add_trend_visualization(self, ax, trend, monthly_series):
        """Add visual elements for a single trend."""
        # Get trend period data
        mask = (monthly_series.index >= trend['start_date']) & \
               (monthly_series.index <= trend['end_date'])
        trend_data = monthly_series[mask]
        
        if len(trend_data) < 2:
            return
        
        # Determine colors based on direction
        if trend['direction'] == 'increasing':
            face_color = 'green'
            arrow_color = 'darkgreen'
            arrow_direction = 1  # Upward
        else:
            face_color = 'red'
            arrow_color = 'darkred'
            arrow_direction = -1  # Downward
        
        # Add shaded region for trend period
        ax.axvspan(trend['start_date'], trend['end_date'],
                  alpha=0.15, color=face_color, zorder=0)
        
        # Add border lines
        ax.axvline(trend['start_date'], color=face_color, alpha=0.3, 
                  linestyle='--', linewidth=1, zorder=1)
        ax.axvline(trend['end_date'], color=face_color, alpha=0.3,
                  linestyle='--', linewidth=1, zorder=1)
        
        # Calculate arrow position
        mid_date = trend['start_date'] + (trend['end_date'] - trend['start_date']) / 2
        
        # Find y-value at mid-point
        closest_idx = np.argmin(np.abs(monthly_series.index - mid_date))
        if closest_idx < len(monthly_series):
            mid_y = monthly_series.iloc[closest_idx]
        else:
            mid_y = trend_data.mean()
        
        # Add directional arrow
        arrow_length = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        arrow_start_y = mid_y + arrow_direction * arrow_length * 0.5
        arrow_end_y = mid_y + arrow_direction * arrow_length * 1.5
        
        # Scale arrow based on trend strength
        strength_scale = {'strong': 1.2, 'moderate': 1.0, 'weak': 0.8}
        scale = strength_scale.get(trend['strength'], 1.0)
        
        arrow = FancyArrowPatch(
            (mdates.date2num(mid_date), arrow_start_y),
            (mdates.date2num(mid_date), arrow_end_y),
            arrowstyle='->', mutation_scale=15*scale,
            color=arrow_color, linewidth=2*scale,
            alpha=0.7, zorder=10
        )
        ax.add_patch(arrow)
        
        # Add trend magnitude annotation for strong trends
        if trend['strength'] == 'strong' and abs(trend['relative_change']) > 0.1:
            change_pct = abs(trend['relative_change']) * 100
            annotation_text = f'{change_pct:.0f}%'
            
            ax.annotate(annotation_text,
                       xy=(mdates.date2num(mid_date), arrow_end_y),
                       xytext=(0, arrow_direction * 10),
                       textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       color=arrow_color, alpha=0.8,
                       ha='center', va='bottom' if arrow_direction > 0 else 'top')
    
    def create_combined_overview(self):
        """Create a combined overview showing all frames with key trends."""
        print("\nCreating combined overview plot...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        for idx, frame in enumerate(FRAME_NAMES):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])
            
            # Get data
            frame_trends = self.trends_df[self.trends_df['frame'] == frame]
            frame_monthly = self.monthly_data.set_index('month')[f'{frame}_prop']
            
            # Filter only high significance trends
            high_sig_trends = frame_trends[
                frame_trends['significance_level'].isin(['very_high', 'high'])
            ]
            
            # Create simplified plot
            self._plot_simplified_panel(ax, frame_monthly, high_sig_trends, frame)
        
        # Main title
        fig.suptitle('Climate Change Framing Trends - All Frames Overview\nHigh Significance Trends Only (p < 0.01)',
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save
        output_path = AESTHETIC_DIR / "all_frames_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    def _plot_simplified_panel(self, ax, monthly_series, trends, frame):
        """Create simplified panel for overview."""
        # Smooth data
        if len(monthly_series) > 3:
            smoothed_values = gaussian_filter1d(monthly_series.values, sigma=2)
            
            # Main line
            ax.plot(monthly_series.index, smoothed_values,
                   color=FRAME_COLORS[frame], linewidth=2,
                   label=frame, zorder=5)
            
            # Confidence band
            ax.fill_between(monthly_series.index,
                          smoothed_values * 0.95,
                          smoothed_values * 1.05,
                          color=FRAME_COLORS[frame], alpha=0.1, zorder=1)
        
        # Add trend periods
        for _, trend in trends.iterrows():
            color = 'green' if trend['direction'] == 'increasing' else 'red'
            ax.axvspan(trend['start_date'], trend['end_date'],
                      alpha=0.2, color=color, zorder=0)
        
        # Styling
        ax.set_title(f'{frame} Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel('Proportion', fontsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # X-axis
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Annotations
        n_increasing = len(trends[trends['direction'] == 'increasing'])
        n_decreasing = len(trends[trends['direction'] == 'decreasing'])
        
        ax.text(0.02, 0.98, f'↑ {n_increasing} | ↓ {n_decreasing}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', alpha=0.8))
    
    def create_trend_summary_table(self):
        """Create a visual summary table of key trends."""
        print("\nCreating trend summary table...")
        
        # Prepare summary data
        summary_data = []
        
        for frame in FRAME_NAMES:
            frame_trends = self.trends_df[self.trends_df['frame'] == frame]
            
            # Calculate statistics by significance
            for sig_level in ['very_high', 'high', 'moderate', 'low']:
                sig_trends = frame_trends[frame_trends['significance_level'] == sig_level]
                
                if len(sig_trends) > 0:
                    summary_data.append({
                        'Frame': frame,
                        'Significance': sig_level.replace('_', ' ').title(),
                        'Count': len(sig_trends),
                        'Avg Duration (years)': sig_trends['duration_years'].mean(),
                        'Max Change (%)': abs(sig_trends['relative_change'].max()) * 100,
                        'Increasing': len(sig_trends[sig_trends['direction'] == 'increasing']),
                        'Decreasing': len(sig_trends[sig_trends['direction'] == 'decreasing'])
                    })
        
        if not summary_data:
            print("  No data for summary table")
            return
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = summary_df.values
        col_labels = summary_df.columns.tolist()
        
        # Color code by frame
        cell_colors = []
        for _, row in summary_df.iterrows():
            frame = row['Frame']
            row_color = [FRAME_COLORS[frame] + '20'] * len(col_labels)  # Add transparency
            cell_colors.append(row_color)
        
        table = ax.table(cellText=table_data,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        cellColours=cell_colors,
                        colColours=['lightgray'] * len(col_labels))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(col_labels)):
            cell = table[(0, i)]
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
        
        plt.title('Trend Summary Statistics by Frame and Significance Level',
                 fontsize=14, fontweight='bold', pad=20)
        
        # Save
        output_path = AESTHETIC_DIR / "trend_summary_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    def create_temporal_flow_diagram(self):
        """Create a flow diagram showing trend transitions over time."""
        print("\nCreating temporal flow diagram...")
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Time axis setup
        years = range(1990, 2025, 5)
        y_positions = {frame: i for i, frame in enumerate(FRAME_NAMES)}
        
        # Plot baseline for each frame
        for frame, y_pos in y_positions.items():
            ax.plot([1990, 2024], [y_pos, y_pos], 
                   color='lightgray', linewidth=0.5, alpha=0.5, zorder=0)
            ax.text(1988, y_pos, frame, fontsize=10, ha='right', va='center')
        
        # Add trends as flows
        for frame in FRAME_NAMES:
            frame_trends = self.trends_df[
                (self.trends_df['frame'] == frame) &
                (self.trends_df['significance_level'].isin(['very_high', 'high']))
            ]
            
            y_pos = y_positions[frame]
            
            for _, trend in frame_trends.iterrows():
                # Calculate positions
                start_year = trend['start_date'].year
                end_year = trend['end_date'].year
                
                # Color and width based on properties
                color = 'green' if trend['direction'] == 'increasing' else 'red'
                alpha = 0.7 if trend['significance_level'] == 'very_high' else 0.4
                linewidth = 3 if trend['strength'] == 'strong' else 2
                
                # Draw trend line
                ax.plot([start_year, end_year], [y_pos, y_pos],
                       color=color, alpha=alpha, linewidth=linewidth,
                       solid_capstyle='round', zorder=2)
                
                # Add arrow at midpoint
                mid_year = (start_year + end_year) / 2
                arrow_y = y_pos + (0.1 if trend['direction'] == 'increasing' else -0.1)
                
                ax.annotate('', xy=(mid_year, arrow_y), xytext=(mid_year, y_pos),
                          arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=1))
        
        # Styling
        ax.set_xlim(1988, 2026)
        ax.set_ylim(-0.5, len(FRAME_NAMES) - 0.5)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_title('Temporal Flow of Significant Trends Across Frames',
                    fontsize=14, fontweight='bold')
        
        # Add year markers
        for year in years:
            ax.axvline(year, color='gray', alpha=0.2, linestyle=':', linewidth=0.5)
        
        ax.set_xticks(years)
        ax.set_yticks([])
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', linewidth=3, alpha=0.7, label='Increasing trend'),
            plt.Line2D([0], [0], color='red', linewidth=3, alpha=0.7, label='Decreasing trend'),
            plt.Line2D([0], [0], color='gray', linewidth=3, alpha=0.7, label='Very high significance'),
            plt.Line2D([0], [0], color='gray', linewidth=3, alpha=0.4, label='High significance')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Save
        output_path = AESTHETIC_DIR / "temporal_flow_diagram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    def run_aesthetic_visualization(self):
        """Run complete aesthetic visualization pipeline."""
        print("="*80)
        print("AESTHETIC TREND VISUALIZATION")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Create visualizations
        self.create_frame_significance_plots()
        self.create_combined_overview()
        self.create_trend_summary_table()
        self.create_temporal_flow_diagram()
        
        print("\n" + "="*80)
        print("AESTHETIC VISUALIZATION COMPLETE")
        print("="*80)
        print(f"All visualizations saved to: {AESTHETIC_DIR}")


def main():
    """Main execution function."""
    visualizer = AestheticTrendVisualizer()
    visualizer.run_aesthetic_visualization()


if __name__ == "__main__":
    main()