"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
03_event_cascade_impact_analysis.py

MAIN OBJECTIVE:
---------------
This script creates focused visualizations analyzing:
1) Impact of Event_Detection on media cascades and frame changes
2) Which specific event types (event_1 to event_8) have the most impact on cascades and frame changes
3) Event-frame-cascade relationships

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- plotly

MAIN FEATURES:
--------------
1) Event → Cascade → Frame change flow analysis
2) Event type impact comparison
3) Frame-specific event effectiveness
4) Temporal dynamics visualization
5) Statistical significance testing

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Constants
FRAME_NAMES = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]
FRAME_COLORS = {
    "Cult": "#E64B35",    # Red
    "Eco": "#4DBBD5",     # Cyan
    "Envt": "#00A087",    # Teal
    "Pbh": "#3C5488",     # Blue
    "Just": "#F39B7F",    # Light orange
    "Pol": "#8491B4",     # Light blue
    "Sci": "#91D1C2",     # Light teal
    "Secu": "#B09C85",    # Brown
}

EVENT_NAMES = {
    'event_1': 'Extreme Weather',
    'event_2': 'Conference',
    'event_3': 'Publication',
    'event_4': 'Election',
    'event_5': 'Policy',
    'event_6': 'Court Decision',
    'event_7': 'Cultural',
    'event_8': 'Protest'
}

EVENT_COLORS = {
    'event_1': '#FF6B6B',  # Red - extreme weather
    'event_2': '#4ECDC4',  # Teal - conferences
    'event_3': '#45B7D1',  # Blue - publications
    'event_4': '#F7DC6F',  # Yellow - elections
    'event_5': '#BB8FCE',  # Purple - policy
    'event_6': '#85C1E2',  # Light blue - court
    'event_7': '#F8C471',  # Orange - cultural
    'event_8': '#82E0AA'   # Green - protest
}

DATA_DIR = project_root / "data" / "03_events_effects"
RESULTS_DIR = project_root / "results" / "03_events_effects"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EventCascadeImpactAnalyzer:
    """Analyzes the impact of events on media cascades and frame changes."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.results = self._load_all_data()
        
    def _load_all_data(self) -> dict:
        """Load all necessary data files."""
        results = {}
        
        # Load cascade summary
        cascade_path = DATA_DIR / "media_cascade_summary.csv"
        if cascade_path.exists():
            results['cascade_summary'] = pd.read_csv(cascade_path)
            results['cascade_summary']['critical_date'] = pd.to_datetime(
                results['cascade_summary']['critical_date']
            )
            print(f"Loaded cascade summary with columns: {list(results['cascade_summary'].columns)}")
        
        # Load event impact summary
        event_path = DATA_DIR / "event_impact_summary.csv"
        if event_path.exists():
            results['event_summary'] = pd.read_csv(event_path)
            results['event_summary']['critical_date'] = pd.to_datetime(
                results['event_summary']['critical_date']
            )
            print(f"Loaded event summary with columns: {list(results['event_summary'].columns)}")
        
        # Load detailed cascade results
        cascade_detailed_path = DATA_DIR / "detailed_cascade_results.json"
        if cascade_detailed_path.exists():
            with open(cascade_detailed_path, 'r') as f:
                results['cascade_detailed'] = json.load(f)
        
        # Load event impact detailed
        event_detailed_path = DATA_DIR / "event_impact_detailed.json"
        if event_detailed_path.exists():
            with open(event_detailed_path, 'r') as f:
                results['event_detailed'] = json.load(f)
        
        # Load focusing events
        focusing_path = DATA_DIR / "focusing_events.csv"
        if focusing_path.exists():
            results['focusing_events'] = pd.read_csv(focusing_path)
            results['focusing_events']['date'] = pd.to_datetime(
                results['focusing_events']['date']
            )
        
        # Load critical dates
        critical_path = DATA_DIR / "critical_trend_dates.csv"
        if critical_path.exists():
            results['critical_dates'] = pd.read_csv(critical_path)
            results['critical_dates']['date'] = pd.to_datetime(
                results['critical_dates']['date']
            )
        
        return results
    
    def create_figure1_event_cascade_flow(self):
        """Figure 1: Event → Cascade → Frame Change Flow Analysis."""
        print("\nCreating Figure 1: Event-Cascade-Frame Flow...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Impact of Events on Media Cascades and Frame Changes', 
                     fontsize=16, fontweight='bold')
        
        # Panel A: Events vs No Events - Cascade Strength Comparison
        ax1 = axes[0, 0]
        
        if 'event_summary' in self.results and 'cascade_summary' in self.results:
            # Merge data on critical dates
            merged = pd.merge(
                self.results['cascade_summary'],
                self.results['event_summary'],
                on=['frame', 'critical_date'],
                how='inner',
                suffixes=('_cascade', '_event')
            )
            
            print(f"Merged data columns: {list(merged.columns)}")
            
            # Check column names and handle appropriately
            cascade_col = None
            for col in ['cascade_strength', 'cascade_strength_cascade', 'cascade_strength_x']:
                if col in merged.columns:
                    cascade_col = col
                    break
            
            if cascade_col and 'total_events' in merged.columns:
                # Separate by event presence
                with_events = merged[merged['total_events'] > 0][cascade_col].dropna()
                without_events = merged[merged['total_events'] == 0][cascade_col].dropna()
                
                # Box plot
                data_to_plot = []
                labels = []
                
                if len(with_events) > 0:
                    data_to_plot.append(with_events)
                    labels.append(f'With Events\n(n={len(with_events)})')
                
                if len(without_events) > 0:
                    data_to_plot.append(without_events)
                    labels.append(f'No Events\n(n={len(without_events)})')
                
                if data_to_plot:
                    bp = ax1.boxplot(data_to_plot, labels=labels,
                                    patch_artist=True, showfliers=False)
                    
                    # Color boxes
                    colors = ['#FF6B6B', '#CCCCCC']
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)
                    
                    # Statistical test if both groups exist
                    if len(data_to_plot) == 2 and len(data_to_plot[0]) > 0 and len(data_to_plot[1]) > 0:
                        statistic, p_value = mannwhitneyu(data_to_plot[0], data_to_plot[1])
                        ax1.text(0.5, 0.95, f'Mann-Whitney U: p = {p_value:.3f}', 
                                transform=ax1.transAxes, ha='center',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax1.text(0.5, 0.5, 'Data not available', transform=ax1.transAxes, 
                        ha='center', va='center')
            
            ax1.set_ylabel('Cascade Strength')
            ax1.set_title('A. Event Impact on Cascade Strength')
            ax1.grid(True, alpha=0.3)
        
        # Panel B: Event Type Effectiveness
        ax2 = axes[0, 1]
        
        if 'event_detailed' in self.results and 'event_impacts' in self.results['event_detailed']:
            # Extract event type effectiveness
            event_effectiveness = {}
            
            for event_type in EVENT_NAMES.keys():
                scores = []
                
                # Collect all impact scores for this event type
                for key, data in self.results['event_detailed']['event_impacts'].items():
                    if 'impact_scores' in data and event_type in data['impact_scores']:
                        scores.append(data['impact_scores'][event_type])
                
                if scores:
                    event_effectiveness[event_type] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'count': len(scores)
                    }
            
            if event_effectiveness:
                # Sort by mean effectiveness
                sorted_events = sorted(event_effectiveness.items(), 
                                     key=lambda x: x[1]['mean'], reverse=True)
                
                events = [e[0] for e in sorted_events]
                means = [e[1]['mean'] for e in sorted_events]
                stds = [e[1]['std'] for e in sorted_events]
                counts = [e[1]['count'] for e in sorted_events]
                
                # Bar plot with error bars
                x_pos = np.arange(len(events))
                bars = ax2.bar(x_pos, means, yerr=stds, capsize=5,
                              color=[EVENT_COLORS[e] for e in events],
                              edgecolor='black', linewidth=0.5)
                
                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax2.text(i, bar.get_height() + stds[i] + 0.01, f'n={count}', 
                            ha='center', fontsize=8)
                
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([EVENT_NAMES[e] for e in events], 
                                   rotation=45, ha='right')
                ax2.set_ylabel('Average Impact Score')
                ax2.set_title('B. Event Type Effectiveness')
                ax2.set_ylim(0, max(means) * 1.2 if means else 1)
                ax2.grid(True, axis='y', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No event effectiveness data', 
                        transform=ax2.transAxes, ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, 'Event detailed data not available', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        # Panel C: Cascade Strength vs Frame Change Magnitude
        ax3 = axes[1, 0]
        
        if 'cascade_summary' in self.results:
            cascade_data = self.results['cascade_summary']
            
            # Check for required columns
            cascade_col = 'cascade_strength' if 'cascade_strength' in cascade_data.columns else None
            change_col = 'article_volume_change' if 'article_volume_change' in cascade_data.columns else None
            
            if cascade_col and change_col:
                # Calculate frame change magnitude
                cascade_data['change_magnitude'] = np.abs(cascade_data[change_col])
                
                # Remove NaN values
                plot_data = cascade_data[[cascade_col, 'change_magnitude', 'frame']].dropna()
                
                if len(plot_data) > 0:
                    # Scatter plot
                    for frame in FRAME_NAMES:
                        frame_data = plot_data[plot_data['frame'] == frame]
                        if len(frame_data) > 0:
                            ax3.scatter(frame_data[cascade_col],
                                      frame_data['change_magnitude'],
                                      color=FRAME_COLORS[frame],
                                      alpha=0.6, s=50, label=frame,
                                      edgecolors='black', linewidth=0.5)
                    
                    # Add trend line
                    if len(plot_data) > 10:
                        z = np.polyfit(plot_data[cascade_col], plot_data['change_magnitude'], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(plot_data[cascade_col].min(), 
                                            plot_data[cascade_col].max(), 100)
                        ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
                        
                        # Correlation
                        corr, p_val = spearmanr(plot_data[cascade_col], plot_data['change_magnitude'])
                        ax3.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {p_val:.3f}',
                                transform=ax3.transAxes, 
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
                              fontsize=8, ncol=1)
            else:
                ax3.text(0.5, 0.5, 'Required columns not found', 
                        transform=ax3.transAxes, ha='center', va='center')
            
            ax3.set_xlabel('Cascade Strength')
            ax3.set_ylabel('Frame Change Magnitude')
            ax3.set_title('C. Cascade → Frame Change Relationship')
            ax3.grid(True, alpha=0.3)
        
        # Panel D: Event Density vs Cascade Acceleration
        ax4 = axes[1, 1]
        
        if 'event_summary' in self.results and 'cascade_summary' in self.results:
            # Merge data again
            merged = pd.merge(
                self.results['cascade_summary'],
                self.results['event_summary'],
                on=['frame', 'critical_date'],
                how='inner',
                suffixes=('_cascade', '_event')
            )
            
            # Find the right column names
            cascade_col = None
            for col in ['cascade_strength', 'cascade_strength_cascade', 'cascade_strength_x']:
                if col in merged.columns:
                    cascade_col = col
                    break
            
            accel_col = None
            for col in ['journalist_acceleration', 'journalist_acceleration_cascade', 'journalist_acceleration_x']:
                if col in merged.columns:
                    accel_col = col
                    break
            
            if 'total_events' in merged.columns and cascade_col and accel_col:
                # Event density (events per week in window)
                merged['event_density'] = merged['total_events'] / 6  # 6-week window
                
                # Remove NaN values
                plot_data = merged[['event_density', accel_col, cascade_col]].dropna()
                
                if len(plot_data) > 0:
                    # Scatter plot
                    scatter = ax4.scatter(plot_data['event_density'],
                                        plot_data[accel_col],
                                        c=plot_data[cascade_col],
                                        cmap='YlOrRd', s=50, alpha=0.6,
                                        edgecolors='black', linewidth=0.5)
                    
                    cbar = plt.colorbar(scatter, ax=ax4)
                    cbar.set_label('Cascade Strength', rotation=270, labelpad=15)
                    
                    # Add trend line if enough data
                    if len(plot_data) > 10:
                        z = np.polyfit(plot_data['event_density'], plot_data[accel_col], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(plot_data['event_density'].min(), 
                                            plot_data['event_density'].max(), 100)
                        ax4.plot(x_trend, p(x_trend), "b--", alpha=0.8, linewidth=2)
            else:
                ax4.text(0.5, 0.5, 'Required data not available', 
                        transform=ax4.transAxes, ha='center', va='center')
            
            ax4.set_xlabel('Event Density (events/week)')
            ax4.set_ylabel('Journalist Adoption Acceleration')
            ax4.set_title('D. Event Density Impact on Cascade Dynamics')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = RESULTS_DIR / "Figure1_event_cascade_flow.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 1 saved to {output_path}")
    
    def create_figure2_event_type_impact_matrix(self):
        """Figure 2: Event Type Impact Matrix by Frame."""
        print("\nCreating Figure 2: Event Type Impact Matrix...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Event Type Impact Analysis by Frame', 
                     fontsize=16, fontweight='bold')
        
        # Panel A: Event-Frame Impact Heatmap
        if 'event_detailed' in self.results and 'event_impacts' in self.results['event_detailed']:
            # Build impact matrix
            impact_matrix = np.zeros((len(EVENT_NAMES), len(FRAME_NAMES)))
            count_matrix = np.zeros((len(EVENT_NAMES), len(FRAME_NAMES)))
            
            for key, data in self.results['event_detailed']['event_impacts'].items():
                frame = data.get('frame')
                if frame in FRAME_NAMES:
                    frame_idx = FRAME_NAMES.index(frame)
                    
                    for event_type, score in data.get('impact_scores', {}).items():
                        if event_type in EVENT_NAMES:
                            event_idx = list(EVENT_NAMES.keys()).index(event_type)
                            impact_matrix[event_idx, frame_idx] += score
                            count_matrix[event_idx, frame_idx] += 1
            
            # Average impacts
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_impact_matrix = np.divide(impact_matrix, count_matrix)
                avg_impact_matrix[np.isnan(avg_impact_matrix)] = 0
            
            # Create heatmap
            im1 = ax1.imshow(avg_impact_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0)
            
            # Labels
            ax1.set_xticks(range(len(FRAME_NAMES)))
            ax1.set_xticklabels(FRAME_NAMES)
            ax1.set_yticks(range(len(EVENT_NAMES)))
            ax1.set_yticklabels([EVENT_NAMES[e] for e in EVENT_NAMES.keys()])
            
            # Add text annotations
            for i in range(len(EVENT_NAMES)):
                for j in range(len(FRAME_NAMES)):
                    if avg_impact_matrix[i, j] > 0:
                        text = ax1.text(j, i, f'{avg_impact_matrix[i, j]:.2f}',
                                      ha="center", va="center",
                                      color="white" if avg_impact_matrix[i, j] > 0.5 else "black",
                                      fontsize=8)
            
            # Colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Average Impact Score', rotation=270, labelpad=15)
            
            ax1.set_title('A. Event Type Impact by Frame')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Event Type')
        else:
            ax1.text(0.5, 0.5, 'Event impact data not available', 
                    transform=ax1.transAxes, ha='center', va='center')
        
        # Panel B: Cascade Response by Event-Frame Combination
        if 'cascade_summary' in self.results and 'event_summary' in self.results:
            merged = pd.merge(
                self.results['cascade_summary'],
                self.results['event_summary'],
                on=['frame', 'critical_date'],
                how='inner',
                suffixes=('_cascade', '_event')
            )
            
            # Find cascade strength column
            cascade_col = None
            for col in ['cascade_strength', 'cascade_strength_cascade', 'cascade_strength_x']:
                if col in merged.columns:
                    cascade_col = col
                    break
            
            if cascade_col and 'top_impact_event' in merged.columns:
                # Get top event types for each frame
                frame_event_cascade = {}
                
                for frame in FRAME_NAMES:
                    frame_data = merged[merged['frame'] == frame]
                    
                    for event_type in EVENT_NAMES.keys():
                        event_frame_data = frame_data[
                            frame_data['top_impact_event'] == event_type
                        ]
                        
                        if len(event_frame_data) > 0:
                            key = f"{frame}-{EVENT_NAMES[event_type][:8]}"
                            frame_event_cascade[key] = {
                                'mean': event_frame_data[cascade_col].mean(),
                                'std': event_frame_data[cascade_col].std(),
                                'count': len(event_frame_data)
                            }
                
                if frame_event_cascade:
                    # Sort by mean cascade strength
                    sorted_combinations = sorted(frame_event_cascade.items(),
                                               key=lambda x: x[1]['mean'],
                                               reverse=True)[:15]  # Top 15
                    
                    # Bar plot
                    labels = [comb[0] for comb in sorted_combinations]
                    means = [comb[1]['mean'] for comb in sorted_combinations]
                    stds = [comb[1]['std'] for comb in sorted_combinations]
                    counts = [comb[1]['count'] for comb in sorted_combinations]
                    
                    x_pos = np.arange(len(labels))
                    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5,
                                  color='steelblue', edgecolor='black', linewidth=0.5)
                    
                    # Add count labels
                    for i, (bar, count) in enumerate(zip(bars, counts)):
                        ax2.text(i, means[i] + stds[i] + 0.01, f'n={count}',
                                ha='center', fontsize=7)
                    
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels(labels, rotation=45, ha='right')
                    ax2.set_ylabel('Cascade Strength (mean ± std)')
                    ax2.set_title('B. Top Frame-Event Combinations for Cascades')
                    ax2.grid(True, axis='y', alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No frame-event combinations found', 
                            transform=ax2.transAxes, ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, 'Required columns not found', 
                        transform=ax2.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        # Save
        output_path = RESULTS_DIR / "Figure2_event_type_impact_matrix.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 2 saved to {output_path}")
    
    def create_figure3_temporal_dynamics(self):
        """Figure 3: Temporal Dynamics of Event-Cascade-Frame Relationships."""
        print("\nCreating Figure 3: Temporal Dynamics...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
        
        # Panel A: Event Frequency Over Time
        ax1 = fig.add_subplot(gs[0, :])
        
        if 'critical_dates' in self.results:
            # Group by month
            critical_dates = self.results['critical_dates'].copy()
            critical_dates['month'] = critical_dates['date'].dt.to_period('M').dt.to_timestamp()
            
            # Count events per month
            monthly_events = critical_dates.groupby('month').size()
            
            if len(monthly_events) > 0:
                ax1.bar(monthly_events.index, monthly_events.values,
                       width=20, color='gray', alpha=0.6, edgecolor='black')
                
                # Add trend line if enough data
                if len(monthly_events) > 5:
                    x_numeric = np.arange(len(monthly_events))
                    z = np.polyfit(x_numeric, monthly_events.values, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(0, len(monthly_events)-1, 100)
                    y_smooth = p(x_smooth)
                    
                    # Map back to dates
                    dates_smooth = monthly_events.index[0] + pd.to_timedelta(
                        x_smooth * (monthly_events.index[-1] - monthly_events.index[0]).days / (len(monthly_events) - 1), 
                        unit='D'
                    )
                    
                    ax1.plot(dates_smooth, y_smooth, 'r--', linewidth=2, label='Trend')
                    ax1.legend()
                
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Number of Critical Events')
                ax1.set_title('A. Critical Event Frequency Over Time')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No critical dates data', 
                        transform=ax1.transAxes, ha='center', va='center')
        
        # Panel B: Cascade Strength Evolution
        ax2 = fig.add_subplot(gs[1, 0])
        
        if 'cascade_summary' in self.results:
            cascade_data = self.results['cascade_summary'].copy()
            
            # Find cascade strength column
            cascade_col = None
            for col in ['cascade_strength', 'cascade_strength_cascade']:
                if col in cascade_data.columns:
                    cascade_col = col
                    break
            
            if cascade_col:
                cascade_data['year'] = cascade_data['critical_date'].dt.year
                
                # Annual average cascade strength
                annual_cascade = cascade_data.groupby('year')[cascade_col].agg(['mean', 'std', 'count'])
                
                if len(annual_cascade) > 0:
                    # Only plot years with sufficient data
                    valid_years = annual_cascade[annual_cascade['count'] >= 3]
                    
                    if len(valid_years) > 0:
                        ax2.errorbar(valid_years.index, valid_years['mean'],
                                    yerr=valid_years['std'], fmt='o-',
                                    capsize=5, markersize=8, linewidth=2)
                        
                        # Add count labels
                        for year, row in valid_years.iterrows():
                            ax2.text(year, row['mean'] + row['std'] + 0.02, 
                                    f'n={row["count"]}', ha='center', fontsize=8)
                        
                        ax2.set_xlabel('Year')
                        ax2.set_ylabel('Cascade Strength')
                        ax2.set_title('B. Annual Cascade Strength Evolution')
                        ax2.grid(True, alpha=0.3)
                    else:
                        ax2.text(0.5, 0.5, 'Insufficient data for annual analysis', 
                                transform=ax2.transAxes, ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, 'Cascade strength column not found', 
                        transform=ax2.transAxes, ha='center', va='center')
        
        # Panel C: Event Type Distribution Over Time
        ax3 = fig.add_subplot(gs[1, 1])
        
        if 'event_summary' in self.results and 'top_impact_event' in self.results['event_summary'].columns:
            event_data = self.results['event_summary'].copy()
            event_data['year'] = event_data['critical_date'].dt.year
            
            # Count event types by year
            event_type_counts = {}
            years = sorted(event_data['year'].unique())
            
            for year in years:
                year_data = event_data[event_data['year'] == year]
                counts = year_data['top_impact_event'].value_counts()
                event_type_counts[year] = counts
            
            if event_type_counts:
                # Convert to DataFrame for stacked bar
                event_type_df = pd.DataFrame(event_type_counts).fillna(0).T
                
                # Plot stacked bar
                bottom = np.zeros(len(event_type_df))
                
                for event_type in EVENT_NAMES.keys():
                    if event_type in event_type_df.columns:
                        values = event_type_df[event_type].values
                        ax3.bar(event_type_df.index, values, bottom=bottom,
                               label=EVENT_NAMES[event_type], color=EVENT_COLORS[event_type])
                        bottom += values
                
                ax3.set_xlabel('Year')
                ax3.set_ylabel('Number of Events')
                ax3.set_title('C. Event Type Distribution Over Time')
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax3.grid(True, axis='y', alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No event type distribution data', 
                        transform=ax3.transAxes, ha='center', va='center')
        
        # Panel D: Event-Cascade Lag Analysis
        ax4 = fig.add_subplot(gs[2, :])
        
        if 'cascade_summary' in self.results and 'peak_lag_weeks' in self.results['cascade_summary'].columns:
            cascade_data = self.results['cascade_summary']
            
            # Group by frame and plot lag distributions
            frames_with_data = []
            lag_data = []
            
            for frame in FRAME_NAMES:
                frame_data = cascade_data[cascade_data['frame'] == frame]
                if len(frame_data) > 0:
                    lags = frame_data['peak_lag_weeks'].dropna()
                    if len(lags) > 3:  # Need at least 3 data points
                        frames_with_data.append(frame)
                        lag_data.append(lags.values)
            
            if lag_data:
                bp = ax4.boxplot(lag_data, labels=frames_with_data,
                               patch_artist=True, showfliers=False)
                
                # Color boxes by frame
                for i, (patch, frame) in enumerate(zip(bp['boxes'], frames_with_data)):
                    patch.set_facecolor(FRAME_COLORS[frame])
                    patch.set_alpha(0.7)
                
                ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Event timing')
                ax4.set_xlabel('Frame')
                ax4.set_ylabel('Peak Cascade Lag (weeks)')
                ax4.set_title('D. Time Lag Between Event and Peak Cascade by Frame')
                ax4.legend()
                ax4.grid(True, axis='y', alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient lag data', 
                        transform=ax4.transAxes, ha='center', va='center')
        else:
            ax4.text(0.5, 0.5, 'Peak lag data not available', 
                    transform=ax4.transAxes, ha='center', va='center')
        
        plt.suptitle('Temporal Dynamics of Event-Cascade Relationships',
                    fontsize=16, fontweight='bold')
        
        # Save
        output_path = RESULTS_DIR / "Figure3_temporal_dynamics.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 3 saved to {output_path}")
    
    def create_figure4_cascade_pathway_analysis(self):
        """Figure 4: Cascade Pathway Analysis."""
        print("\nCreating Figure 4: Cascade Pathway Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Media Cascade Pathway Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Panel A: Journalist vs Media Response
        ax1 = axes[0, 0]
        
        if 'cascade_summary' in self.results:
            cascade_data = self.results['cascade_summary']
            
            # Check for required columns
            journalist_col = None
            media_col = None
            cascade_col = None
            
            for col in ['journalist_acceleration', 'journalist_acceleration_cascade']:
                if col in cascade_data.columns:
                    journalist_col = col
                    break
                    
            for col in ['media_expansion', 'media_expansion_cascade']:
                if col in cascade_data.columns:
                    media_col = col
                    break
                    
            for col in ['cascade_strength', 'cascade_strength_cascade']:
                if col in cascade_data.columns:
                    cascade_col = col
                    break
            
            if journalist_col and media_col and cascade_col:
                # Remove NaN values
                plot_data = cascade_data[[journalist_col, media_col, cascade_col]].dropna()
                
                if len(plot_data) > 0:
                    # Scatter plot of journalist vs media metrics
                    scatter = ax1.scatter(plot_data[journalist_col],
                                        plot_data[media_col],
                                        c=plot_data[cascade_col],
                                        cmap='viridis', s=60, alpha=0.7,
                                        edgecolors='black', linewidth=0.5)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax1)
                    cbar.set_label('Overall Cascade Strength', rotation=270, labelpad=15)
                    
                    # Add diagonal line
                    max_val = max(plot_data[journalist_col].max(),
                                 plot_data[media_col].max())
                    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal response')
                    
                    ax1.set_xlabel('Journalist Adoption Acceleration')
                    ax1.set_ylabel('Media Outlet Expansion')
                    ax1.set_title('A. Journalist vs Media Cascade Components')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'No data to plot', 
                            transform=ax1.transAxes, ha='center', va='center')
            else:
                ax1.text(0.5, 0.5, 'Required columns not found', 
                        transform=ax1.transAxes, ha='center', va='center')
        
        # Panel B: Cascade Component Contribution
        ax2 = axes[0, 1]
        
        if 'cascade_detailed' in self.results:
            # Extract component contributions
            journalist_contrib = []
            media_contrib = []
            article_contrib = []
            
            for key, data in self.results['cascade_detailed'].items():
                journalist_metrics = data.get('journalist_metrics', {})
                media_metrics = data.get('media_metrics', {})
                article_metrics = data.get('article_metrics', {})
                
                # Calculate contributions (simplified)
                j_score = journalist_metrics.get('adoption_acceleration', 0) * 0.3
                m_score = media_metrics.get('new_media_ratio', 0) * 0.3
                a_score = article_metrics.get('volume_change', 0) * 0.4
                
                # Cap values to reasonable range
                journalist_contrib.append(min(max(j_score, -2), 2))
                media_contrib.append(min(max(m_score, -2), 2))
                article_contrib.append(min(max(a_score, -2), 2))
            
            # Stacked bar chart
            if journalist_contrib:
                x = np.arange(3)
                width = 0.6
                
                means = [np.mean(journalist_contrib),
                        np.mean(media_contrib),
                        np.mean(article_contrib)]
                stds = [np.std(journalist_contrib),
                       np.std(media_contrib),
                       np.std(article_contrib)]
                
                bars = ax2.bar(x, means, width, yerr=stds, capsize=10,
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                             edgecolor='black', linewidth=0.5)
                
                # Add value labels
                for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                    ax2.text(i, mean + std + 0.01, f'{mean:.3f}', 
                            ha='center', fontsize=9)
                
                ax2.set_xticks(x)
                ax2.set_xticklabels(['Journalists', 'Media Outlets', 'Articles'])
                ax2.set_ylabel('Average Contribution to Cascade')
                ax2.set_title('B. Cascade Component Contributions')
                ax2.grid(True, axis='y', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No cascade component data', 
                        transform=ax2.transAxes, ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, 'Detailed cascade data not available', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        # Panel C: Frame-Specific Cascade Patterns
        ax3 = axes[1, 0]
        
        if 'cascade_summary' in self.results:
            cascade_data = self.results['cascade_summary']
            
            # Find columns
            journalist_col = None
            media_col = None
            volume_col = None
            
            for col in ['journalist_acceleration', 'journalist_acceleration_cascade']:
                if col in cascade_data.columns:
                    journalist_col = col
                    break
                    
            for col in ['media_expansion', 'media_expansion_cascade']:
                if col in cascade_data.columns:
                    media_col = col
                    break
                    
            for col in ['article_volume_change', 'article_volume_change_cascade']:
                if col in cascade_data.columns:
                    volume_col = col
                    break
            
            if journalist_col and media_col and volume_col:
                # Calculate cascade profile for each frame
                cascade_profiles = []
                frame_labels = []
                
                for frame in FRAME_NAMES:
                    frame_data = cascade_data[cascade_data['frame'] == frame]
                    if len(frame_data) > 3:
                        profile = [
                            frame_data[journalist_col].mean(),
                            frame_data[media_col].mean(),
                            frame_data[volume_col].mean()
                        ]
                        cascade_profiles.append(profile)
                        frame_labels.append(frame)
                
                if cascade_profiles:
                    # Normalize profiles
                    cascade_profiles = np.array(cascade_profiles)
                    
                    # Handle edge cases
                    for j in range(cascade_profiles.shape[1]):
                        col_max = np.abs(cascade_profiles[:, j]).max()
                        if col_max > 0:
                            cascade_profiles[:, j] = cascade_profiles[:, j] / col_max
                    
                    # Create heatmap
                    im = ax3.imshow(cascade_profiles.T, cmap='RdBu_r', aspect='auto',
                                   vmin=-1, vmax=1)
                    
                    ax3.set_xticks(range(len(frame_labels)))
                    ax3.set_xticklabels(frame_labels)
                    ax3.set_yticks(range(3))
                    ax3.set_yticklabels(['Journalist\nAcceleration',
                                       'Media\nExpansion',
                                       'Article\nVolume'])
                    
                    # Add text annotations
                    for i in range(len(frame_labels)):
                        for j in range(3):
                            text = ax3.text(i, j, f'{cascade_profiles[i, j]:.2f}',
                                          ha="center", va="center",
                                          color="white" if abs(cascade_profiles[i, j]) > 0.5 else "black",
                                          fontsize=8)
                    
                    ax3.set_title('C. Frame-Specific Cascade Profiles (Normalized)')
                    
                    # Colorbar
                    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                    cbar.set_label('Normalized Score', rotation=270, labelpad=15)
                else:
                    ax3.text(0.5, 0.5, 'Insufficient frame data', 
                            transform=ax3.transAxes, ha='center', va='center')
            else:
                ax3.text(0.5, 0.5, 'Required columns not found', 
                        transform=ax3.transAxes, ha='center', va='center')
        
        # Panel D: Cascade Success Rate by Event Presence
        ax4 = axes[1, 1]
        
        if 'cascade_summary' in self.results and 'event_summary' in self.results:
            merged = pd.merge(
                self.results['cascade_summary'],
                self.results['event_summary'],
                on=['frame', 'critical_date'],
                how='inner',
                suffixes=('_cascade', '_event')
            )
            
            # Find cascade strength column
            cascade_col = None
            for col in ['cascade_strength', 'cascade_strength_cascade', 'cascade_strength_x']:
                if col in merged.columns:
                    cascade_col = col
                    break
            
            if cascade_col and 'total_events' in merged.columns:
                # Define cascade success (strength > 0.5)
                merged['cascade_success'] = merged[cascade_col] > 0.5
                
                # Group by event presence
                success_rates = []
                labels = []
                counts = []
                
                # No events
                no_events = merged[merged['total_events'] == 0]
                if len(no_events) > 0:
                    success_rates.append(no_events['cascade_success'].mean())
                    labels.append('No Events')
                    counts.append(len(no_events))
                
                # Low events (1-5)
                low_events = merged[(merged['total_events'] > 0) & (merged['total_events'] <= 5)]
                if len(low_events) > 0:
                    success_rates.append(low_events['cascade_success'].mean())
                    labels.append('Low (1-5)')
                    counts.append(len(low_events))
                
                # Medium events (6-20)
                med_events = merged[(merged['total_events'] > 5) & (merged['total_events'] <= 20)]
                if len(med_events) > 0:
                    success_rates.append(med_events['cascade_success'].mean())
                    labels.append('Medium (6-20)')
                    counts.append(len(med_events))
                
                # High events (>20)
                high_events = merged[merged['total_events'] > 20]
                if len(high_events) > 0:
                    success_rates.append(high_events['cascade_success'].mean())
                    labels.append('High (>20)')
                    counts.append(len(high_events))
                
                # Bar plot
                if success_rates:
                    bars = ax4.bar(range(len(success_rates)), 
                                  [r * 100 for r in success_rates],
                                  color=['#CCCCCC', '#FFE5B4', '#FFB84D', '#FF6B6B'][:len(success_rates)])
                    
                    # Add percentage and count labels
                    for i, (bar, rate, count) in enumerate(zip(bars, success_rates, counts)):
                        ax4.text(i, bar.get_height() + 1, f'{rate*100:.1f}%\n(n={count})',
                                ha='center', fontsize=8)
                    
                    ax4.set_xticks(range(len(labels)))
                    ax4.set_xticklabels(labels)
                    ax4.set_ylabel('Cascade Success Rate (%)')
                    ax4.set_ylim(0, max(110, max([r * 100 for r in success_rates]) * 1.2))
                    ax4.set_title('D. Cascade Success by Event Density')
                    ax4.grid(True, axis='y', alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No event density data', 
                            transform=ax4.transAxes, ha='center', va='center')
            else:
                ax4.text(0.5, 0.5, 'Required columns not found', 
                        transform=ax4.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        # Save
        output_path = RESULTS_DIR / "Figure4_cascade_pathway_analysis.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 4 saved to {output_path}")
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard for detailed exploration."""
        print("\nCreating Interactive Dashboard...")
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Event Impact Score Distribution',
                'Cascade Strength by Event Type',
                'Frame Change Dynamics',
                'Event-Cascade-Frame Network'
            ),
            specs=[[{"type": "box"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter3d"}]]
        )
        
        # 1. Event Impact Distribution
        if 'event_summary' in self.results and 'top_impact_event' in self.results['event_summary'].columns:
            event_data = self.results['event_summary']
            
            for event_type in EVENT_NAMES.keys():
                type_data = event_data[event_data['top_impact_event'] == event_type]
                if len(type_data) > 0 and 'top_impact_score' in type_data.columns:
                    fig.add_trace(
                        go.Box(y=type_data['top_impact_score'],
                              name=EVENT_NAMES[event_type],
                              marker_color=EVENT_COLORS[event_type]),
                        row=1, col=1
                    )
        
        # 2. Cascade Strength by Event Type
        if 'cascade_summary' in self.results and 'event_summary' in self.results:
            merged = pd.merge(
                self.results['cascade_summary'],
                self.results['event_summary'],
                on=['frame', 'critical_date'],
                how='inner',
                suffixes=('_cascade', '_event')
            )
            
            # Find cascade strength column
            cascade_col = None
            for col in ['cascade_strength', 'cascade_strength_cascade', 'cascade_strength_x']:
                if col in merged.columns:
                    cascade_col = col
                    break
            
            if cascade_col and 'top_impact_event' in merged.columns:
                event_cascade_means = merged.groupby('top_impact_event')[cascade_col].mean()
                event_cascade_means = event_cascade_means.sort_values(ascending=False)
                
                fig.add_trace(
                    go.Bar(x=[EVENT_NAMES.get(e, e) for e in event_cascade_means.index],
                          y=event_cascade_means.values,
                          marker_color=[EVENT_COLORS.get(e, 'gray') for e in event_cascade_means.index]),
                    row=1, col=2
                )
        
        # 3. Frame Change Dynamics
        if 'cascade_summary' in self.results:
            cascade_data = self.results['cascade_summary']
            
            # Find columns
            cascade_col = None
            volume_col = None
            
            for col in ['cascade_strength', 'cascade_strength_cascade']:
                if col in cascade_data.columns:
                    cascade_col = col
                    break
                    
            for col in ['article_volume_change', 'article_volume_change_cascade']:
                if col in cascade_data.columns:
                    volume_col = col
                    break
            
            if cascade_col and volume_col:
                for frame in FRAME_NAMES:
                    frame_data = cascade_data[cascade_data['frame'] == frame]
                    if len(frame_data) > 0:
                        fig.add_trace(
                            go.Scatter(x=frame_data[cascade_col],
                                      y=frame_data[volume_col],
                                      mode='markers',
                                      name=frame,
                                      marker=dict(color=FRAME_COLORS[frame], size=8)),
                            row=2, col=1
                        )
        
        # 4. 3D Event-Cascade-Frame Network
        if 'focusing_events' in self.results and len(self.results['focusing_events']) > 0:
            focusing = self.results['focusing_events']
            
            # Check for required columns
            if all(col in focusing.columns for col in ['impact', 'cascade', 'composite_score', 'frame', 'event_type']):
                # Create 3D visualization
                fig.add_trace(
                    go.Scatter3d(
                        x=focusing['impact'],
                        y=focusing['cascade'],
                        z=focusing['composite_score'],
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color=[FRAME_COLORS.get(f, 'gray') for f in focusing['frame']],
                            line=dict(width=1, color='black')
                        ),
                        text=[f"{row['frame']}<br>{EVENT_NAMES.get(row['event_type'], row['event_type'])}"
                              for _, row in focusing.iterrows()],
                        textposition='top center',
                        hovertemplate='%{text}<br>Impact: %{x:.3f}<br>Cascade: %{y:.3f}<br>Score: %{z:.3f}'
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Event-Cascade-Frame Impact Dashboard",
            showlegend=True,
            height=800
        )
        
        # Update axes
        fig.update_xaxes(title_text="Event Type", row=1, col=1)
        fig.update_yaxes(title_text="Impact Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Event Type", row=1, col=2)
        fig.update_yaxes(title_text="Average Cascade Strength", row=1, col=2)
        
        fig.update_xaxes(title_text="Cascade Strength", row=2, col=1)
        fig.update_yaxes(title_text="Article Volume Change", row=2, col=1)
        
        fig.update_scenes(
            xaxis_title="Impact",
            yaxis_title="Cascade",
            zaxis_title="Composite Score",
            row=2, col=2
        )
        
        # Save
        output_path = RESULTS_DIR / "event_cascade_interactive_dashboard.html"
        fig.write_html(str(output_path))
        print(f"✓ Interactive dashboard saved to {output_path}")
    
    def create_summary_statistics_table(self):
        """Create summary statistics table."""
        print("\nCreating Summary Statistics...")
        
        stats = []
        
        # Overall event impact
        if 'event_summary' in self.results:
            event_data = self.results['event_summary']
            
            if 'total_events' in event_data.columns:
                stats.append({
                    'Metric': 'Total Events Analyzed',
                    'Value': f"{event_data['total_events'].sum():,}"
                })
            
            if 'top_impact_score' in event_data.columns:
                stats.append({
                    'Metric': 'Average Event Impact Score',
                    'Value': f"{event_data['top_impact_score'].mean():.3f}"
                })
        
        # Cascade statistics
        if 'cascade_summary' in self.results:
            cascade_data = self.results['cascade_summary']
            
            # Find cascade strength column
            cascade_col = None
            for col in ['cascade_strength', 'cascade_strength_cascade']:
                if col in cascade_data.columns:
                    cascade_col = col
                    break
            
            if cascade_col:
                stats.append({
                    'Metric': 'Average Cascade Strength',
                    'Value': f"{cascade_data[cascade_col].mean():.3f}"
                })
                stats.append({
                    'Metric': 'Cascade Success Rate (>0.5)',
                    'Value': f"{(cascade_data[cascade_col] > 0.5).mean()*100:.1f}%"
                })
        
        # Event effectiveness
        if 'event_detailed' in self.results:
            effectiveness = self.results['event_detailed'].get('relationships', {}).get('avg_event_effectiveness', {})
            if effectiveness:
                most_effective = max(effectiveness, key=effectiveness.get)
                stats.append({
                    'Metric': 'Most Effective Event Type',
                    'Value': EVENT_NAMES.get(most_effective, most_effective)
                })
                stats.append({
                    'Metric': 'Most Effective Event Score',
                    'Value': f"{effectiveness[most_effective]:.3f}"
                })
        
        # Frame sensitivity
        if 'event_detailed' in self.results:
            sensitivity = self.results['event_detailed'].get('relationships', {}).get('avg_frame_sensitivity', {})
            if sensitivity:
                most_sensitive = max(sensitivity, key=sensitivity.get)
                stats.append({
                    'Metric': 'Most Event-Sensitive Frame',
                    'Value': most_sensitive
                })
                stats.append({
                    'Metric': 'Sensitivity Score',
                    'Value': f"{sensitivity[most_sensitive]:.3f}"
                })
        
        # Save as CSV
        if stats:
            stats_df = pd.DataFrame(stats)
            output_path = RESULTS_DIR / "event_cascade_summary_statistics.csv"
            stats_df.to_csv(output_path, index=False)
            print(f"✓ Summary statistics saved to {output_path}")
            
            # Also print to console
            print("\nSummary Statistics:")
            print("="*50)
            for stat in stats:
                print(f"{stat['Metric']}: {stat['Value']}")
        else:
            print("No statistics to report")
    
    def run_all_analyses(self):
        """Run all visualization analyses."""
        print("="*80)
        print("EVENT CASCADE IMPACT ANALYSIS")
        print("="*80)
        
        # Check if data is available
        if not self.results:
            print("ERROR: No data found. Please run the analysis scripts first!")
            print("\nRequired scripts to run first:")
            print("1. 03_trend_reversal_detection.py")
            print("2. 03_media_cascade_analysis.py")
            print("3. 03_event_impact_analysis.py")
            return
        
        # Create all figures
        try:
            self.create_figure1_event_cascade_flow()
        except Exception as e:
            print(f"Error creating Figure 1: {e}")
            
        try:
            self.create_figure2_event_type_impact_matrix()
        except Exception as e:
            print(f"Error creating Figure 2: {e}")
            
        try:
            self.create_figure3_temporal_dynamics()
        except Exception as e:
            print(f"Error creating Figure 3: {e}")
            
        try:
            self.create_figure4_cascade_pathway_analysis()
        except Exception as e:
            print(f"Error creating Figure 4: {e}")
            
        try:
            self.create_interactive_dashboard()
        except Exception as e:
            print(f"Error creating interactive dashboard: {e}")
            
        try:
            self.create_summary_statistics_table()
        except Exception as e:
            print(f"Error creating summary statistics: {e}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll visualizations saved to: {RESULTS_DIR}")


def main():
    """Main execution function."""
    analyzer = EventCascadeImpactAnalyzer()
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()