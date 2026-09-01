"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
02_dominant_frames_visualization.py

MAIN OBJECTIVE:
---------------
This script creates publication-quality visualizations for top-tier journals
showing frame dominance analysis and paradigm evolution with sophisticated design.

Dependencies:
-------------
- pandas, numpy, matplotlib, seaborn, networkx, scipy

MAIN FEATURES:
--------------
1) Figure 1: Comprehensive method comparison with dominance determination
2) Figure 2: Sophisticated temporal evolution with network analysis by decade
3) Nature/Science publication standards

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
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy import stats
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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

# Import database access utilities
from src.database_access.db_connector import DatabaseConnector
from src.database_access.data_processor import FrameDataProcessor

# Configure matplotlib for Nature/Science quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'figure.titlesize': 10,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'grid.linewidth': 0.3,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'text.usetex': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Professional color palette
COLORS = {
    "Cult": "#d62728",    # Brick red
    "Eco": "#1f77b4",     # Muted blue
    "Envt": "#2ca02c",    # Forest green
    "Pbh": "#ff7f0e",     # Safety orange
    "Just": "#9467bd",    # Muted purple
    "Pol": "#8c564b",     # Chestnut brown
    "Sci": "#17becf",     # Strong cyan
    "Secu": "#e377c2",    # Raspberry
}

FRAME_NAMES = {
    "Cult": "Culture",
    "Eco": "Economy", 
    "Envt": "Environment",
    "Pbh": "Public Health",
    "Just": "Justice",
    "Pol": "Politics",
    "Sci": "Science",
    "Secu": "Security",
}

# Paths
DATA_DIR = project_root / "data" / "02_dominant_frames"
RESULTS_DIR = project_root / "results" / "02_dominant_frames"


class PublicationVisualizer:
    """Create sophisticated visualizations for paradigm analysis."""
    
    def __init__(self):
        """Initialize visualizer."""
        print("="*60)
        print("CREATING PUBLICATION-QUALITY VISUALIZATIONS")
        print("="*60)
    
    def load_data(self):
        """Load all required data."""
        print("\nLoading data...")
        
        data = {}
        
        # Load core data
        data['dominance'] = pd.read_csv(DATA_DIR / "overall_dominance_scores.csv", index_col=0)
        data['temporal'] = pd.read_csv(DATA_DIR / "temporal_paradigm_analysis.csv")
        data['weekly'] = pd.read_csv(DATA_DIR / "weekly_paradigm_evolution.csv")
        data['weekly']['date'] = pd.to_datetime(data['weekly']['date'])
        data['frames'] = pd.read_csv(DATA_DIR / "processed_frame_data.csv")
        data['frames']['date'] = pd.to_datetime(data['frames']['date'])
        
        # Load network evolution
        network_path = DATA_DIR / "network_evolution.json"
        with open(network_path, 'r') as f:
            data['network'] = json.load(f)
        
        print("✓ Data loaded successfully")
        return data
    
    def load_temporal_proportions(self):
        """Load temporal frame proportions from database."""
        print("\nLoading temporal frame proportions from database...")
        
        # Initialize database connection
        db_connector = DatabaseConnector(exclude_2025=True)
        
        try:
            # Use SQL aggregation for efficiency
            query = '''
            WITH monthly_data AS (
                SELECT
                    DATE_TRUNC('month', date) as month,
                    doc_id,
                    AVG("cultural_frame") as "cultural_frame",
                    AVG("economic_frame") as "economic_frame",
                    AVG("environmental_frame") as "environmental_frame",
                    AVG("health_frame") as "health_frame",
                    AVG("justice_frame") as "justice_frame",
                    AVG("political_frame") as "political_frame",
                    AVG("scientific_frame") as "scientific_frame",
                    AVG("security_frame") as "security_frame"
                FROM "CCF_processed_data"
                WHERE date <= %(end_date)s::date
                AND sentences IS NOT NULL
                AND sentences != ''
                GROUP BY DATE_TRUNC('month', date), doc_id
            )
            SELECT
                month,
                AVG("cultural_frame") as "Cult",
                AVG("economic_frame") as "Eco",
                AVG("environmental_frame") as "Envt",
                AVG("health_frame") as "Pbh",
                AVG("justice_frame") as "Just",
                AVG("political_frame") as "Pol",
                AVG("scientific_frame") as "Sci",
                AVG("security_frame") as "Secu"
            FROM monthly_data
            GROUP BY month
            ORDER BY month
            '''
            params = {'end_date': '2024-12-31'}
            
            print("Fetching aggregated data from database...")
            monthly_proportions = db_connector.read_data(query, params, show_progress=True)
            
            if monthly_proportions.empty:
                print("Warning: No data retrieved from database")
                return pd.DataFrame()
            
            # Set month as index and ensure it's a DatetimeIndex
            # Convert timezone-aware to timezone-naive
            monthly_proportions['month'] = pd.to_datetime(monthly_proportions['month'], utc=True).dt.tz_localize(None)
            monthly_proportions.set_index('month', inplace=True)
            
            print("✓ Temporal proportions loaded successfully")
            return monthly_proportions
            
        finally:
            # Close database connection
            db_connector.close()
    
    def create_figure1_method_comparison(self, data):
        """Create Figure 1: Comprehensive method comparison."""
        print("\nCreating Figure 1: Method Comparison...")
        
        # Load temporal proportions for new panel a
        temporal_proportions = self.load_temporal_proportions()
        
        # Setup
        dominance = data['dominance']
        overall = data['temporal'][data['temporal']['period'] == 'Overall'].iloc[0]
        dominant_frames = overall['dominant_frames'].split(', ') if overall['dominant_frames'] else []
        
        # Create figure
        fig = plt.figure(figsize=(7.2, 9.6))  # 183mm x 244mm (slightly taller for 5 panels)
        
        # Create grid with 4 rows (removed empty row)
        gs = gridspec.GridSpec(4, 4, figure=fig,
                            height_ratios=[1.5, 1.2, 1.2, 2.0],
                            width_ratios=[1, 1, 1, 1],
                            hspace=0.5, wspace=0.25,
                            left=0.08, right=0.92,
                            bottom=0.05, top=0.92) 
        
        # Sort frames by dominance
        sorted_frames = dominance.sort_values('dominance_score', ascending=False).index
        
        # --- NEW Panel A: Temporal frame proportions ---
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot monthly proportions
        for frame in FRAME_NAMES.keys():
            if frame in temporal_proportions.columns:
                # Apply smoothing
                values = temporal_proportions[frame].values
                smoothed = gaussian_filter1d(values, sigma=2)
                
                ax1.plot(temporal_proportions.index, smoothed, 
                        color=COLORS[frame], label=FRAME_NAMES[frame],
                        linewidth=2.5 if frame in dominant_frames else 1.5,
                        alpha=0.95 if frame in dominant_frames else 0.7)
        
        # Customize
        ax1.set_xlabel('Date', fontsize=8)
        ax1.set_ylabel('Average proportion of\nsentences per article', fontsize=8)
        ax1.set_ylim(0, None)
        # Move legend to the right side of the plot
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  fontsize=6, frameon=False, handlelength=1.5)
        ax1.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.set_title('Temporal evolution of frame proportions in climate coverage', 
                     fontsize=9, pad=10)
        
        # Format x-axis
        years = temporal_proportions.index.year.unique()
        if len(years) > 20:
            # Show every 5th year
            year_ticks = [pd.Timestamp(f'{y}-01-01') for y in years[::5]]
            ax1.set_xticks(year_ticks)
            ax1.set_xticklabels([y.year for y in year_ticks], rotation=45, ha='right')
        
        ax1.text(-0.08, 1.02, 'a', transform=ax1.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # --- Panel B (formerly a): Composite dominance scores ---
        ax2 = fig.add_subplot(gs[1, :])
        
        # Determine threshold
        if dominant_frames:
            threshold = dominance.loc[dominant_frames[-1], 'dominance_score']
        else:
            threshold = dominance['dominance_score'].median()
        
        # Create horizontal bars
        y_pos = np.arange(len(sorted_frames))
        scores = dominance.loc[sorted_frames, 'dominance_score'].values
        
        # Plot bars
        bars = ax2.barh(y_pos, scores, height=0.6)
        
        # Color bars based on dominance
        for i, (bar, frame) in enumerate(zip(bars, sorted_frames)):
            if frame in dominant_frames:
                bar.set_color(COLORS[frame])
                bar.set_alpha(0.9)
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)
            else:
                bar.set_color(COLORS[frame])
                bar.set_alpha(0.4)
                bar.set_edgecolor('gray')
                bar.set_linewidth(0.5)
            
            # Add score text
            ax2.text(scores[i] + 0.01, y_pos[i], f'{scores[i]:.3f}',
                    va='center', ha='left', fontsize=6,
                    color='black' if frame in dominant_frames else 'gray')
        
        # Add subtle threshold line
        ax2.axvline(threshold, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax2.text(threshold, len(sorted_frames) - 0.8, f'Threshold = {threshold:.3f}',
                ha='center', va='center', fontsize=6, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='black', alpha=0.8))
        
        # Customize
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([FRAME_NAMES[f] for f in sorted_frames], fontsize=7)
        ax2.set_xlabel('Composite dominance score', fontsize=8)
        ax2.set_xlim(0, 1.05)
        ax2.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.set_title('Composite dominance scores by frame', fontsize=9, pad=10)
        ax2.text(-0.08, 1.02, 'b', transform=ax2.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # --- Panel C (formerly b): Method contribution breakdown ---
        ax3 = fig.add_subplot(gs[2, :2])
        
        # Stacked bar chart for each frame
        method_scores = {
            'Information Theory': dominance['information_theory_score'].values,
            'Network Analysis': dominance['network_analysis_score'].values,
            'Causality': dominance['causality_score'].values,
            'Proportional': dominance['proportional_score'].values
        }
        
        x = np.arange(len(sorted_frames))
        bottom = np.zeros(len(sorted_frames))
        
        colors_method = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (method, scores) in enumerate(method_scores.items()):
            scores_sorted = scores[[list(dominance.index).index(f) for f in sorted_frames]]
            ax3.bar(x, scores_sorted, bottom=bottom, label=method,
                   color=colors_method[i], alpha=0.8, edgecolor='white', linewidth=0.5)
            bottom += scores_sorted
        
        ax3.set_xticks(x)
        ax3.set_xticklabels([FRAME_NAMES[f][:4] for f in sorted_frames], 
                          rotation=45, ha='right', fontsize=7)
        ax3.set_ylabel('Method contribution', fontsize=8)
        ax3.legend(loc='upper right', fontsize=6, frameon=False)
        ax3.set_ylim(0, 4.2)
        ax3.set_title('Method contribution breakdown', fontsize=9, pad=10)
        ax3.text(-0.15, 1.02, 'c', transform=ax3.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # --- Panel D (formerly c): Network visualization ---
        ax4 = fig.add_subplot(gs[2, 2:])
        
        # Build network
        G = nx.Graph()
        for frame in FRAME_NAMES.keys():
            G.add_node(frame, score=dominance.loc[frame, 'dominance_score'])
        
        # Add edges based on correlation
        corr = data['frames'][list(FRAME_NAMES.keys())].corr()
        for i, f1 in enumerate(FRAME_NAMES.keys()):
            for j, f2 in enumerate(FRAME_NAMES.keys()):
                if i < j and corr.loc[f1, f2] > 0.3:
                    G.add_edge(f1, f2, weight=corr.loc[f1, f2])
        
        # Layout - use kamada_kawai for better spread
        pos = nx.kamada_kawai_layout(G)
        
        # Scale up all positions to make network larger
        scale_factor = 1.5
        pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}
        
        # Draw edges
        edges = G.edges()
        for edge in edges:
            weight = G[edge[0]][edge[1]]['weight']
            ax4.plot([pos[edge[0]][0], pos[edge[1]][0]],
                    [pos[edge[0]][1], pos[edge[1]][1]],
                    color='gray', alpha=weight*0.6, 
                    linewidth=weight*3, zorder=1)
        
        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            score = G.nodes[node]['score']
            size = 400 + score * 800
            
            circle = Circle((x, y), radius=size/5000,
                          facecolor=COLORS[node],
                          edgecolor='black' if node in dominant_frames else 'gray',
                          linewidth=2 if node in dominant_frames else 1,
                          alpha=0.9 if node in dominant_frames else 0.5,
                          zorder=3)
            ax4.add_patch(circle)
            
            ax4.text(x, y, FRAME_NAMES[node], ha='center', va='center',
                    fontsize=7, fontweight='bold' if node in dominant_frames else 'normal',
                    color='black')
        
        # Set limits based on actual positions
        all_x = [pos[node][0] for node in G.nodes()]
        all_y = [pos[node][1] for node in G.nodes()]
        x_margin = (max(all_x) - min(all_x)) * 0.2
        y_margin = (max(all_y) - min(all_y)) * 0.2
        
        ax4.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax4.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title('Frame interaction network', fontsize=9, pad=10)
        ax4.text(-0.15, 1.02, 'd', transform=ax4.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # --- Panel E (formerly a): Method comparison heatmap ---
        ax5 = fig.add_subplot(gs[3, :])  # Last row
        
        # Prepare data for heatmap
        methods = {
            'Information Theory': ['mutual_information', 'entropy_reduction', 'information_content'],
            'Network Analysis': ['pagerank', 'eigenvector_centrality'],
            'Causality': ['granger_causality', 'transfer_entropy'],
            'Proportional': ['mean_proportion', 'elevation_score', 'consistency']
        }
        
        # Create normalized scores matrix
        method_names = []
        score_matrix = []
        
        for method_name, metrics in methods.items():
            for metric in metrics:
                if metric in dominance.columns:
                    method_names.append(f"{method_name}\n{metric.replace('_', ' ')}")
                    scores = dominance.loc[sorted_frames, metric].values
                    # Normalize
                    if scores.max() > scores.min():
                        normalized = (scores - scores.min()) / (scores.max() - scores.min())
                    else:
                        normalized = np.full_like(scores, 0.5)
                    score_matrix.append(normalized)
        
        score_matrix = np.array(score_matrix)
        
        # Create heatmap
        im = ax5.imshow(score_matrix, cmap='RdYlBu_r', aspect='auto', 
                       interpolation='nearest', vmin=0, vmax=1)
        
        # Add value annotations
        for i in range(len(method_names)):
            for j in range(len(sorted_frames)):
                text = ax5.text(j, i, f'{score_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=5,
                               color='white' if score_matrix[i, j] > 0.5 else 'black')
        
        # Customize axes
        ax5.set_xticks(range(len(sorted_frames)))
        ax5.set_xticklabels([FRAME_NAMES[f] for f in sorted_frames], 
                          rotation=45, ha='right', fontsize=7)
        ax5.set_yticks(range(len(method_names)))
        ax5.set_yticklabels(method_names, fontsize=6)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, fraction=0.02, pad=0.02)
        cbar.set_label('Normalized score', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        
        # Highlight dominant frames
        for j, frame in enumerate(sorted_frames):
            if frame in dominant_frames:
                rect = Rectangle((j-0.5, -0.5), 1, len(method_names), 
                              fill=False, edgecolor='red', 
                              linewidth=2, linestyle='-')
                ax5.add_patch(rect)
        
        ax5.set_title('Comparative analysis of all methods across frames', 
                     fontsize=9, pad=10)
        ax5.text(-0.08, 1.02, 'e', transform=ax5.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # Main title
        fig.suptitle('Multi-method frame dominance analysis', 
                    fontsize=11, fontweight='bold', y=0.96, x=0.5)
        
        # Save
        output_path = RESULTS_DIR / "Figure1_dominance_analysis.pdf"
        plt.savefig(output_path, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Figure 1 saved to: {output_path}")
    
    def create_figure2_temporal_evolution(self, data):
        """Create Figure 2: Temporal evolution with decadal networks."""
        print("\nCreating Figure 2: Temporal Evolution...")
        
        # Setup
        temporal = data['temporal']
        network_data = data['network']
        frames_data = data['frames']
        
        # Filter yearly data
        yearly = temporal[temporal['period'].str.match(r'^\d{4}$', na=False)].copy()
        yearly['year'] = yearly['period'].astype(int)
        years = yearly['year'].values
        
        # Create figure with adjusted layout
        fig = plt.figure(figsize=(7.2, 12))  # 183mm x 305mm
        
        # Create grid with 5 rows for better network visualization
        gs = gridspec.GridSpec(5, 3, figure=fig,
                              height_ratios=[1.8, 0.8, 1.2, 1.5, 1.5],
                              width_ratios=[1, 1, 1],
                              hspace=0.4, wspace=0.15,  # Reduced wspace for networks
                              left=0.07, right=0.97,
                              bottom=0.04, top=0.94)
        
        # --- Panel A: Ranking grid visualization with uncertainty zone ---
        ax1 = fig.add_subplot(gs[0, :])
        
        frames = list(FRAME_NAMES.keys())
        n_frames = len(frames)
        n_years = len(years)
        
        # Identify monoframe period and uncertainty zone
        # Find the end of monoframe period (when n_dominant_frames > 1)
        n_dominant = yearly['n_dominant_frames'].values
        monoframe_end_idx = None
        uncertainty_start_idx = None
        uncertainty_end_idx = None
        
        # Find transition from monoframe to multiframe
        for i in range(len(n_dominant)):
            if n_dominant[i] > 1:
                monoframe_end_idx = i
                # Define uncertainty zone: 2 years before and after transition
                uncertainty_start_idx = max(0, i - 2)
                uncertainty_end_idx = min(len(years) - 1, i + 2)
                break
        
        # Add uncertainty zone as a shaded region
        if uncertainty_start_idx is not None and uncertainty_end_idx is not None:
            # Add shaded rectangle for uncertainty zone
            uncertainty_rect = Rectangle(
                (uncertainty_start_idx - 0.5, -0.5),
                uncertainty_end_idx - uncertainty_start_idx + 1,
                n_frames,
                facecolor='gray', alpha=0.15, zorder=0,
                edgecolor='none'
            )
            ax1.add_patch(uncertainty_rect)
            
            # Add text label for uncertainty zone
            ax1.text((uncertainty_start_idx + uncertainty_end_idx) / 2, n_frames - 0.2,
                    'Transition\nuncertainty', ha='center', va='top',
                    fontsize=6, style='italic', color='gray')
        
        # Create grid
        for i, frame in enumerate(frames):
            # Draw horizontal line
            ax1.axhline(y=i, color='#e0e0e0', linewidth=0.5, zorder=1)
            
            # Get rankings for this frame
            for j, year_row in enumerate(yearly.iterrows()):
                year_data = year_row[1]
                year = int(year_data['period'])
                
                # Find rank of this frame
                for rank in range(1, 9):
                    rank_col = f'rank_{rank}_frame'
                    if rank_col in year_data and year_data[rank_col] == frame:
                        # Draw square
                        square = Rectangle((j - 0.4, i-0.4), 0.8, 0.8,
                                         facecolor=COLORS[frame], 
                                         edgecolor='black',
                                         linewidth=1 if rank <= 3 else 0.5,
                                         alpha=0.9 if rank <= 3 else 0.6,
                                         zorder=3)
                        ax1.add_patch(square)
                        
                        # Add rank number
                        ax1.text(j, i, str(rank), ha='center', va='center',
                                fontsize=6, fontweight='bold', color='white')
                        break
        
        # Add frame labels below the plot
        for i, frame in enumerate(frames):
            ax1.text(-1, i, FRAME_NAMES[frame], ha='right', va='center',
                    fontsize=7, fontweight='bold', color='black')
        
        # Customize axes - show all years or reasonable interval
        ax1.set_xlim(-2, n_years)
        ax1.set_ylim(-0.5, n_frames-0.5)
        
        # Determine tick interval based on number of years
        if n_years <= 15:
            tick_interval = 1  # Show every year if 15 or fewer years
        elif n_years <= 30:
            tick_interval = 2  # Show every 2 years if 30 or fewer years
        elif n_years <= 50:
            tick_interval = 5  # Show every 5 years if 50 or fewer years
        else:
            tick_interval = 10  # Show every 10 years for longer periods
        
        tick_positions = range(0, n_years, tick_interval)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([years[i] for i in tick_positions],
                          rotation=45, ha='right', fontsize=6)
        ax1.set_yticks([])
        ax1.set_xlabel('Year', fontsize=8)
        ax1.set_title('Frame dominance rankings over time', fontsize=9, pad=10)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Remove left spine to avoid overlap
        ax1.spines['left'].set_visible(False)
        
        ax1.text(-0.05, 1.02, 'a', transform=ax1.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # --- Panel B & C: Expanded to full width ---
        # B: Number of dominant frames (full width)
        ax2 = fig.add_subplot(gs[1, :2])  # Span first two columns
        
        n_dominant = yearly['n_dominant_frames'].values
        
        # Smooth curve
        if len(years) > 10:
            window = min(7, len(years)//4)
            n_smooth = pd.Series(n_dominant).rolling(window=window, center=True).mean()
            n_smooth = n_smooth.bfill().ffill()
            
            ax2.fill_between(years, 0, n_smooth, alpha=0.3, color='#3498db')
            ax2.plot(years, n_smooth, color='#2c3e50', linewidth=2.5)
        else:
            ax2.plot(years, n_dominant, 'o-', color='#2c3e50', linewidth=2)
        
        ax2.scatter(years, n_dominant, color='#2c3e50', s=20, zorder=5,
                   edgecolor='white', linewidth=1)
        
        ax2.set_xlabel('Year', fontsize=8)
        ax2.set_ylabel('Number of\ndominant frames', fontsize=8)
        ax2.set_ylim(0, 8)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax2.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.text(-0.1, 1.02, 'b', transform=ax2.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # C: Diversity metrics with LOESS smoothing and confidence intervals
        ax3 = fig.add_subplot(gs[1, 2])
        
        shannon = yearly['shannon_diversity'].values
        concentration = yearly['paradigm_concentration'].values
        
        # Apply LOESS smoothing with confidence intervals
        if HAS_STATSMODELS and len(years) > 10:
            # LOESS smoothing for Shannon diversity
            frac = 0.3  # Fraction of data used for smoothing
            shannon_smooth_result = lowess(shannon, years, frac=frac, return_sorted=True)
            shannon_smooth = shannon_smooth_result[:, 1]
            
            # Calculate confidence intervals using bootstrap
            n_bootstrap = 100
            shannon_bootstrap = np.zeros((n_bootstrap, len(years)))
            for i in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(len(years), len(years), replace=True)
                y_resample = shannon[indices]
                x_resample = years[indices]
                smooth_resample = lowess(y_resample, x_resample, frac=frac, xvals=years)
                shannon_bootstrap[i] = smooth_resample
            
            shannon_ci_lower = np.percentile(shannon_bootstrap, 2.5, axis=0)
            shannon_ci_upper = np.percentile(shannon_bootstrap, 97.5, axis=0)
            
            # LOESS smoothing for concentration
            conc_smooth_result = lowess(concentration, years, frac=frac, return_sorted=True)
            conc_smooth = conc_smooth_result[:, 1]
            
            # Calculate confidence intervals for concentration
            conc_bootstrap = np.zeros((n_bootstrap, len(years)))
            for i in range(n_bootstrap):
                indices = np.random.choice(len(years), len(years), replace=True)
                y_resample = concentration[indices]
                x_resample = years[indices]
                smooth_resample = lowess(y_resample, x_resample, frac=frac, xvals=years)
                conc_bootstrap[i] = smooth_resample
            
            conc_ci_lower = np.percentile(conc_bootstrap, 2.5, axis=0)
            conc_ci_upper = np.percentile(conc_bootstrap, 97.5, axis=0)
        else:
            # Fallback to gaussian smoothing if statsmodels not available
            if len(years) > 10:
                shannon_smooth = gaussian_filter1d(shannon, sigma=2)
                conc_smooth = gaussian_filter1d(concentration, sigma=2)
                # Simple confidence intervals based on standard error
                window = 5
                shannon_std = pd.Series(shannon).rolling(window=window, center=True).std().fillna(0.05)
                shannon_ci_lower = shannon_smooth - 1.96 * shannon_std
                shannon_ci_upper = shannon_smooth + 1.96 * shannon_std
                conc_std = pd.Series(concentration).rolling(window=window, center=True).std().fillna(0.05)
                conc_ci_lower = conc_smooth - 1.96 * conc_std
                conc_ci_upper = conc_smooth + 1.96 * conc_std
            else:
                shannon_smooth = shannon
                conc_smooth = concentration
                shannon_ci_lower = shannon
                shannon_ci_upper = shannon
                conc_ci_lower = concentration
                conc_ci_upper = concentration
        
        # Plot Shannon diversity with confidence interval
        ax3.fill_between(years, shannon_ci_lower, shannon_ci_upper, 
                        color='#16a085', alpha=0.2)
        ax3.plot(years, shannon_smooth, '-', color='#16a085', linewidth=2,
                label='Shannon diversity', alpha=0.9)
        ax3.scatter(years[::5], shannon[::5], color='#16a085', s=15, 
                   alpha=0.6, edgecolor='white', linewidth=0.5)
        
        ax3.set_xlabel('Year', fontsize=8)
        ax3.set_ylabel('Shannon diversity', fontsize=8)
        ax3.set_ylim(0, 1)
        
        ax3b = ax3.twinx()
        # Plot concentration with confidence interval
        ax3b.fill_between(years, conc_ci_lower, conc_ci_upper,
                         color='#e74c3c', alpha=0.2)
        ax3b.plot(years, conc_smooth, '-', color='#e74c3c', linewidth=2,
                 label='Concentration', alpha=0.9)
        ax3b.scatter(years[::5], concentration[::5], color='#e74c3c', s=15, 
                    alpha=0.6, edgecolor='white', linewidth=0.5)
        ax3b.set_ylabel('Concentration', fontsize=8)
        ax3b.set_ylim(0, 1)
        
        # Combined legend
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, 
                  loc='best', frameon=False, fontsize=6)
        
        ax3.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        ax3.text(-0.2, 1.02, 'c', transform=ax3.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # --- Panel D: Network centrality heatmap ---
        ax4 = fig.add_subplot(gs[2, :])
        
        if network_data:
            net_years = sorted([int(y) for y in network_data.keys()])
            frames_list = list(FRAME_NAMES.keys())
            
            centrality_matrix = np.zeros((len(frames_list), len(net_years)))
            
            for j, year in enumerate(net_years):
                year_data = network_data[str(year)]
                pagerank = year_data['pagerank']
                for i, frame in enumerate(frames_list):
                    if frame in pagerank:
                        centrality_matrix[i, j] = pagerank[frame]
            
            # Smooth
            for i in range(len(frames_list)):
                if np.any(centrality_matrix[i, :] > 0):
                    centrality_matrix[i, :] = gaussian_filter1d(centrality_matrix[i, :], sigma=1)
            
            im = ax4.imshow(centrality_matrix, cmap='YlOrRd', aspect='auto',
                          interpolation='bilinear')
            
            ax4.set_xticks(range(0, len(net_years), max(1, len(net_years)//10)))
            ax4.set_xticklabels([net_years[i] for i in range(0, len(net_years), 
                               max(1, len(net_years)//10))], rotation=45, ha='right', fontsize=6)
            ax4.set_yticks(range(len(frames_list)))
            ax4.set_yticklabels([FRAME_NAMES[f] for f in frames_list], fontsize=7)
            ax4.set_xlabel('Year', fontsize=8)
            ax4.set_title('Network centrality evolution (PageRank)', fontsize=9, pad=10)
            
            cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            cbar.set_label('Centrality', fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        
        ax4.text(-0.05, 1.02, 'd', transform=ax4.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
        
        # --- Panel E: Network analysis by decade (2 rows) ---
        # Define decades
        decades = [(1978, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2029)]
        decade_labels = ['1980s', '1990s', '2000s', '2010s', '2020s']
        
        # First row of networks (1980s, 1990s, 2000s)
        for idx, ((start, end), label) in enumerate(zip(decades[:3], decade_labels[:3])):
            ax = fig.add_subplot(gs[3, idx])
            self._draw_decade_network(ax, frames_data, yearly, start, end, label, temporal)
        
        # Second row of networks (2010s, 2020s) - centered
        # Create a subplot that spans the middle area
        ax_2010s = fig.add_subplot(gs[4, 0:2])  # 2010s spans first 2/3
        self._draw_decade_network(ax_2010s, frames_data, yearly, 2010, 2019, '2010s', temporal)
        
        ax_2020s = fig.add_subplot(gs[4, 1:3])  # 2020s spans last 2/3
        self._draw_decade_network(ax_2020s, frames_data, yearly, 2020, 2029, '2020s', temporal)
        
        # Add panel label
        fig.text(0.07, 0.38, 'e', fontsize=12, fontweight='bold', va='bottom')
        
        # Main title
        fig.suptitle('Temporal evolution of climate coverage paradigms', 
                    fontsize=11, fontweight='bold', y=0.97)
        
        # Save
        output_path = RESULTS_DIR / "Figure2_temporal_evolution.pdf"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Figure 2 saved to: {output_path}")
    
    def _draw_decade_network(self, ax, frames_data, yearly, start, end, label, temporal):
        """Helper function to draw network for a specific decade."""
        # Get data for this decade
        decade_data = frames_data[(frames_data['date'].dt.year >= start) & 
                                (frames_data['date'].dt.year <= end)]
        
        print(f"\n{label}: {start}-{end}")
        print(f"  Data points: {len(decade_data)}")
        print(f"  Yearly data points: {len(yearly[(yearly['year'] >= start) & (yearly['year'] <= end)])}")
        
        if len(decade_data) > 0:
            # Get yearly data for this decade
            decade_yearly = yearly[(yearly['year'] >= start) & (yearly['year'] <= end)]
            
            # Count how many times each frame appears in top 3
            top3_counts = {frame: 0 for frame in FRAME_NAMES.keys()}
            
            for _, row in decade_yearly.iterrows():
                # Check ranks 1, 2, and 3
                for rank in [1, 2, 3]:
                    rank_col = f'rank_{rank}_frame'
                    if rank_col in row and row[rank_col] in top3_counts:
                        top3_counts[row[rank_col]] += 1
            
            # Get the 3 frames with most appearances in top 3
            top3_frames = sorted(top3_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_frames = [f[0] for f in top3_frames if f[1] > 0]  # Only include if appeared at least once
            
            # Build network
            G = nx.Graph()
            for frame in FRAME_NAMES.keys():
                is_top3 = frame in top3_frames
                G.add_node(frame, top3=is_top3)
            
            # Calculate correlations
            corr = decade_data[list(FRAME_NAMES.keys())].corr()
            
            # Add edges
            for i, f1 in enumerate(FRAME_NAMES.keys()):
                for j, f2 in enumerate(FRAME_NAMES.keys()):
                    if i < j and corr.loc[f1, f2] > 0.3:
                        G.add_edge(f1, f2, weight=corr.loc[f1, f2])
            
            # Layout - use spring layout with more space
            pos = nx.spring_layout(G, k=3.5, iterations=100, seed=42)  # Increased k for more spread
            
            # Scale up positions to fill the subplot better
            scale_factor = 0.88  # Use most of the available space
            pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}
            
            # Draw edges with thicker lines
            edges = G.edges()
            for edge in edges:
                weight = G[edge[0]][edge[1]]['weight']
                ax.plot([pos[edge[0]][0], pos[edge[1]][0]],
                       [pos[edge[0]][1], pos[edge[1]][1]],
                       color='gray', alpha=weight*0.6, 
                       linewidth=weight*4, zorder=1)  # Increased linewidth
            
            # Draw nodes - much larger
            for node in G.nodes():
                x, y = pos[node]
                is_top3 = G.nodes[node]['top3']
                
                # Much larger radius
                circle = Circle((x, y), radius=0.15,  # Increased from 0.18 to 0.15 (still larger overall due to scale)
                              facecolor=COLORS[node],
                              edgecolor='black' if is_top3 else 'gray',
                              linewidth=3 if is_top3 else 1.5,  # Thicker edges
                              alpha=0.9 if is_top3 else 0.6,
                              zorder=3)
                ax.add_patch(circle)
                
                # Larger font
                ax.text(x, y, FRAME_NAMES[node][:4], ha='center', va='center',
                       fontsize=9, fontweight='bold' if is_top3 else 'normal',  # Increased font size
                       color='white' if is_top3 else 'black')
            
            # Tighter limits to maximize network visibility
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(label, fontsize=8, pad=5)
        else:
            # No data for this decade - show message
            ax.text(0.5, 0.5, f'No data\navailable\nfor {label}', 
                   ha='center', va='center', fontsize=8, color='gray')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.axis('off')
            ax.set_title(label, fontsize=8, pad=5)
    
    def create_all_figures(self):
        """Create all publication-quality figures."""
        # Load data
        data = self.load_data()
        
        # Create figures
        self.create_figure1_method_comparison(data)
        self.create_figure2_temporal_evolution(data)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print("\nGenerated figures:")
        print(f"  - {RESULTS_DIR}/Figure1_dominance_analysis.pdf")
        print(f"  - {RESULTS_DIR}/Figure2_temporal_evolution.pdf")
        print("\nStatistical report:")
        print(f"  - {RESULTS_DIR}/paradigm_statistical_analysis.txt")


def main():
    """Main execution."""
    visualizer = PublicationVisualizer()
    visualizer.create_all_figures()


if __name__ == "__main__":
    main()