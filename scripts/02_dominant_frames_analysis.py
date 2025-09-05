"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
02_dominant_frames_analysis.py

MAIN OBJECTIVE:
---------------
This script performs comprehensive frame dominance analysis using multiple
scientific methods and generates all necessary data for visualization.

Dependencies:
-------------
- pandas, numpy, matplotlib, seaborn, networkx, sklearn, statsmodels, scipy, tqdm

MAIN FEATURES:
--------------
1) Multi-method dominance analysis with equal weights
2) Temporal paradigm evolution tracking
3) Network analysis across time periods
4) Comprehensive statistical report generation
5) All data export for visualization

Author:
-------
Antoine Lemor
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import custom modules
from src.database_access.db_connector import DatabaseConnector
from src.database_access.data_processor import FrameDataProcessor
from src.paradigm_dominance import ParadigmDominanceAnalyzer
from src.smoothing_utils import TimeSeriesSmoother

# Frame colors and names
FRAME_COLORS = {
    "Cult": "#8B4513",
    "Eco": "#2E8B57",
    "Envt": "#4682B4",
    "Pbh": "#FF6347",
    "Just": "#FFD700",
    "Pol": "#8A2BE2",
    "Sci": "#00CED1",
    "Secu": "#DC143C",
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

# Configure paths
DATA_DIR = project_root / "data" / "02_dominant_frames"
RESULTS_DIR = project_root / "results" / "02_dominant_frames"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ComprehensiveParadigmAnalyzer:
    """Comprehensive analysis of frame dominance and paradigm composition."""
    
    def __init__(self, n_workers: int = None):
        """Initialize analyzer."""
        self.n_workers = n_workers or min(cpu_count(), 16)
        self.dominance_analyzer = ParadigmDominanceAnalyzer(n_workers=self.n_workers)
        self.results = {}
        self.temporal_results = None
        self.weekly_paradigm = None
        self.network_evolution = {}
        logger.info(f"Initialized analyzer with {self.n_workers} workers")
    
    def load_data(self, force_reload: bool = False):
        """Load and prepare frame data."""
        print("\n" + "="*60)
        print("LOADING FRAME DATA")
        print("="*60)
        
        processed_path = DATA_DIR / "processed_frame_data.csv"
        
        if processed_path.exists() and not force_reload:
            print("✓ Using existing processed data...")
            with tqdm(total=1, desc="Loading processed data") as pbar:
                df = pd.read_csv(processed_path)
                df['date'] = pd.to_datetime(df['date'])
                pbar.update(1)
        else:
            print("Loading from database...")
            
            with tqdm(total=4, desc="Database operations") as pbar:
                pbar.set_description("Connecting to database")
                connector = DatabaseConnector()
                pbar.update(1)
                
                pbar.set_description("Fetching frame data")
                df = connector.get_frame_data()
                pbar.update(1)
                
                pbar.set_description("Closing connection")
                connector.close()
                pbar.update(1)
                
                pbar.set_description("Processing data")
                processor = FrameDataProcessor(n_workers=self.n_workers)
                df = processor.process_frame_data(df)
                weekly_data = processor.prepare_diversity_data(df)
                pbar.update(1)
            
            print("\nCalculating frame proportions...")
            frame_cols = list(FRAME_NAMES.keys())
            
            with tqdm(total=len(frame_cols)+2, desc="Calculating proportions") as pbar:
                total_counts = weekly_data[frame_cols].sum(axis=1)
                pbar.update(1)
                
                for frame in frame_cols:
                    weekly_data[f"{frame}_prop"] = weekly_data[frame] / total_counts
                    pbar.update(1)
                
                weekly_data['date'] = weekly_data['week']
                df = weekly_data
                pbar.update(1)
            
            df.to_csv(processed_path, index=False)
            print(f"✓ Saved processed data to {processed_path}")
        
        # Ensure proportions
        frame_cols = list(FRAME_NAMES.keys())
        for frame in frame_cols:
            if f"{frame}_prop" in df.columns:
                df[frame] = df[f"{frame}_prop"]
            elif frame in df.columns:
                total = df[frame_cols].sum(axis=1)
                df[frame] = df[frame] / total
        
        print(f"✓ Loaded {len(df):,} rows of data")
        return df
    
    def analyze_overall_paradigm(self, df):
        """Analyze overall paradigm composition."""
        print("\n" + "="*60)
        print("OVERALL PARADIGM COMPOSITION")
        print("="*60)
        
        start_time = time.time()
        
        with tqdm(total=1, desc="Overall paradigm analysis") as pbar:
            overall_analysis = self.dominance_analyzer.analyze_paradigm_composition(
                df, 'overall', show_progress=True
            )
            pbar.update(1)
        
        self.results['overall'] = overall_analysis
        
        # Save dominance scores
        dominance_scores_path = DATA_DIR / "overall_dominance_scores.csv"
        overall_analysis['dominance_scores'].to_csv(dominance_scores_path)
        
        print(f"\nOverall Paradigm Composition:")
        print(f"  Number of dominant frames: {overall_analysis['n_dominant_frames']}")
        print(f"  Dominant frames: {', '.join([FRAME_NAMES[f] for f in overall_analysis['dominant_frames']])}")
        print(f"  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        return overall_analysis
    
    def analyze_temporal_paradigm(self, df):
        """Analyze paradigm composition over time."""
        print("\n" + "="*60)
        print("TEMPORAL PARADIGM EVOLUTION")
        print("="*60)
        
        start_time = time.time()
        
        # Define analysis periods
        periods = {
            '1980-1990': ('1980-01-01', '1989-12-31'),
            '1990-2000': ('1990-01-01', '1999-12-31'),
            '2000-2010': ('2000-01-01', '2009-12-31'),
            '2010-2020': ('2010-01-01', '2019-12-31'),
            '2020-present': ('2020-01-01', '2030-12-31')
        }
        
        # Overall result first
        overall = self.results['overall']
        overall_result = {
            'period': 'Overall',
            'start_date': df['date'].min(),
            'end_date': df['date'].max(),
            'n_weeks': len(df),
            'n_dominant_frames': overall['n_dominant_frames'],
            'dominant_frames': ', '.join(overall['dominant_frames']),
            'paradigm_type': overall['paradigm_type'],
            'shannon_diversity': overall['shannon_diversity'],
            'paradigm_concentration': overall['paradigm_concentration']
        }
        
        # Add dominance ranking
        for i, frame in enumerate(overall['dominant_frames']):
            overall_result[f'rank_{i+1}_frame'] = frame
            overall_result[f'rank_{i+1}_score'] = overall['dominance_scores'].loc[frame, 'dominance_score']
        
        # Analyze periods
        period_results = []
        period_tasks = []
        for period_name, (start, end) in periods.items():
            period_data = df[(df['date'] >= start) & (df['date'] <= end)]
            if len(period_data) > 0:
                period_tasks.append((period_name, start, end, period_data))
        
        print(f"\nAnalyzing {len(period_tasks)} time periods...")
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            with tqdm(total=len(period_tasks), desc="Analyzing periods") as pbar:
                futures = {}
                
                for period_name, start, end, period_data in period_tasks:
                    future = executor.submit(self._analyze_period_with_ranking, 
                                           period_name, start, end, period_data)
                    futures[future] = period_name
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            period_results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error analyzing period {futures[future]}: {e}")
                        pbar.update(1)
        
        # Analyze individual years
        df['year'] = pd.to_datetime(df['date']).dt.year
        unique_years = sorted(df['year'].unique())
        
        year_tasks = []
        for year in unique_years:
            year_data = df[df['year'] == year]
            if len(year_data) > 0:
                year_tasks.append((year, year_data))
        
        print(f"\nAnalyzing {len(year_tasks)} individual years...")
        year_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            with tqdm(total=len(year_tasks), desc="Analyzing years") as pbar:
                futures = {}
                
                for year, year_data in year_tasks:
                    future = executor.submit(self._analyze_year_with_ranking, year, year_data)
                    futures[future] = year
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            year_results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error analyzing year {futures[future]}: {e}")
                        pbar.update(1)
        
        # Combine all results
        all_results = [overall_result] + period_results + year_results
        all_results.sort(key=lambda x: self._get_sort_key(x['period']))
        
        self.temporal_results = pd.DataFrame(all_results)
        
        # Save temporal results
        temporal_path = DATA_DIR / "temporal_paradigm_analysis.csv"
        self.temporal_results.to_csv(temporal_path, index=False)
        
        print(f"\nTemporal analysis complete. Time elapsed: {time.time() - start_time:.1f} seconds")
        
        return self.temporal_results
    
    def _analyze_period_with_ranking(self, period_name: str, start: str, end: str, 
                                   period_data: pd.DataFrame) -> Dict:
        """Analyze a single period with frame ranking."""
        try:
            analyzer = ParadigmDominanceAnalyzer(n_workers=1)
            period_analysis = analyzer.analyze_paradigm_composition(
                period_data, 'period', show_progress=False
            )
            
            result = {
                'period': period_name,
                'start_date': start,
                'end_date': end,
                'n_weeks': len(period_data),
                'n_dominant_frames': period_analysis['n_dominant_frames'],
                'dominant_frames': ', '.join(period_analysis['dominant_frames']),
                'paradigm_type': period_analysis['paradigm_type'],
                'shannon_diversity': period_analysis['shannon_diversity'],
                'paradigm_concentration': period_analysis['paradigm_concentration']
            }
            
            # Add dominance ranking
            for i, frame in enumerate(period_analysis['dominant_frames']):
                result[f'rank_{i+1}_frame'] = frame
                result[f'rank_{i+1}_score'] = period_analysis['dominance_scores'].loc[frame, 'dominance_score']
            
            return result
        except Exception as e:
            logger.error(f"Error in _analyze_period_with_ranking: {e}")
            return None
    
    def _analyze_year_with_ranking(self, year: int, year_data: pd.DataFrame) -> Dict:
        """Analyze a single year with frame ranking."""
        try:
            analyzer = ParadigmDominanceAnalyzer(n_workers=1)
            year_analysis = analyzer.analyze_paradigm_composition(
                year_data, 'yearly', show_progress=False
            )
            
            result = {
                'period': str(year),
                'start_date': year_data['date'].min(),
                'end_date': year_data['date'].max(),
                'n_weeks': len(year_data),
                'n_dominant_frames': year_analysis['n_dominant_frames'],
                'dominant_frames': ', '.join(year_analysis['dominant_frames']),
                'paradigm_type': year_analysis['paradigm_type'],
                'shannon_diversity': year_analysis['shannon_diversity'],
                'paradigm_concentration': year_analysis['paradigm_concentration']
            }
            
            # Add dominance ranking
            for i, frame in enumerate(year_analysis['dominant_frames']):
                result[f'rank_{i+1}_frame'] = frame
                result[f'rank_{i+1}_score'] = year_analysis['dominance_scores'].loc[frame, 'dominance_score']
            
            return result
        except Exception as e:
            logger.error(f"Error in _analyze_year_with_ranking: {e}")
            return None
    
    def analyze_network_evolution(self, df):
        """Analyze network evolution across time."""
        print("\n" + "="*60)
        print("NETWORK EVOLUTION ANALYSIS")
        print("="*60)
        
        # Select years for network analysis
        df['year'] = pd.to_datetime(df['date']).dt.year
        min_year = df['year'].min()
        max_year = df['year'].max()
        
        # More granular time points for smooth evolution
        analysis_years = []
        current_year = min_year
        while current_year <= max_year:
            analysis_years.append(current_year)
            # Smaller steps in recent years for finer detail
            if current_year >= 2000:
                current_year += 2
            elif current_year >= 1990:
                current_year += 3
            else:
                current_year += 5
        
        print(f"Analyzing network evolution for {len(analysis_years)} time points...")
        
        with tqdm(total=len(analysis_years), desc="Building networks") as pbar:
            for year in analysis_years:
                # Use 3-year window for stability
                year_data = df[(df['year'] >= year-1) & (df['year'] <= year+1)]
                
                if len(year_data) > 5:
                    # Build network
                    G = nx.Graph()
                    
                    # Add nodes
                    for frame in FRAME_NAMES.keys():
                        G.add_node(frame)
                    
                    # Calculate co-occurrence
                    threshold = 0.1
                    for i, frame1 in enumerate(FRAME_NAMES.keys()):
                        for j, frame2 in enumerate(FRAME_NAMES.keys()):
                            if i < j:
                                frame1_present = year_data[frame1] > threshold
                                frame2_present = year_data[frame2] > threshold
                                cooccurrence = (frame1_present & frame2_present).sum()
                                
                                if cooccurrence > 5:
                                    weight = cooccurrence / len(year_data)
                                    G.add_edge(frame1, frame2, weight=weight)
                    
                    # Get dominance for this period
                    period_analysis = self.dominance_analyzer.analyze_paradigm_composition(
                        year_data, 'period', show_progress=False
                    )
                    
                    # Calculate network metrics
                    if G.number_of_edges() > 0:
                        pagerank = nx.pagerank(G, weight='weight')
                        betweenness = nx.betweenness_centrality(G, weight='weight')
                        clustering = nx.clustering(G, weight='weight')
                    else:
                        pagerank = {node: 0 for node in G.nodes()}
                        betweenness = {node: 0 for node in G.nodes()}
                        clustering = {node: 0 for node in G.nodes()}
                    
                    self.network_evolution[year] = {
                        'graph': G,
                        'dominant_frames': period_analysis['dominant_frames'],
                        'pagerank': pagerank,
                        'betweenness': betweenness,
                        'clustering': clustering,
                        'n_edges': G.number_of_edges(),
                        'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                    }
                
                pbar.update(1)
        
        # Save network evolution data
        network_path = DATA_DIR / "network_evolution.json"
        network_export = {}
        for year, data in self.network_evolution.items():
            network_export[str(year)] = {
                'dominant_frames': data['dominant_frames'],
                'pagerank': data['pagerank'],
                'betweenness': data['betweenness'],
                'clustering': data['clustering'],
                'n_edges': data['n_edges'],
                'avg_degree': data['avg_degree']
            }
        
        with open(network_path, 'w') as f:
            json.dump(network_export, f, indent=2)
        
        print(f"✓ Network evolution analyzed for {len(self.network_evolution)} time points")
        
        return self.network_evolution
    
    def analyze_weekly_paradigm(self, df):
        """Analyze paradigm evolution weekly."""
        print("\n" + "="*60)
        print("WEEKLY PARADIGM EVOLUTION")
        print("="*60)
        
        start_time = time.time()
        
        print("Calculating paradigm evolution for each week...")
        self.weekly_paradigm = self.dominance_analyzer.calculate_weekly_paradigm_evolution(
            df, show_progress=True
        )
        
        # Save weekly results
        weekly_path = DATA_DIR / "weekly_paradigm_evolution.csv"
        self.weekly_paradigm.to_csv(weekly_path, index=False)
        
        print(f"Analyzed {len(self.weekly_paradigm)} weeks of paradigm evolution")
        print(f"Time elapsed: {time.time() - start_time:.1f} seconds")
        
        return self.weekly_paradigm
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical report."""
        print("\n" + "="*60)
        print("GENERATING STATISTICAL REPORT")
        print("="*60)
        
        report_path = RESULTS_DIR / "paradigm_statistical_analysis.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLIMATE COVERAGE PARADIGM ANALYSIS - STATISTICAL REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            overall = self.results['overall']
            f.write(f"Overall paradigm type: {overall['paradigm_type']}\n")
            f.write(f"Number of dominant frames: {overall['n_dominant_frames']}\n")
            f.write(f"Dominant frames: {', '.join([FRAME_NAMES[f] for f in overall['dominant_frames']])}\n")
            f.write(f"Overall Shannon diversity: {overall['shannon_diversity']:.3f}\n")
            f.write(f"Overall paradigm concentration: {overall['paradigm_concentration']:.3f}\n\n")
            
            # Methodology
            f.write("METHODOLOGY\n")
            f.write("-"*40 + "\n")
            f.write("This analysis employs four equally-weighted approaches:\n")
            f.write("1. Information Theory (25%): Mutual information, conditional entropy, information content\n")
            f.write("2. Network Analysis (25%): PageRank, eigenvector centrality with temporal dynamics\n")
            f.write("3. Causality Analysis (25%): Granger causality, transfer entropy\n")
            f.write("4. Proportional Analysis (25%): Mean proportion, elevation above median, consistency\n\n")
            
            # Dominance Determination
            f.write("DOMINANCE DETERMINATION\n")
            f.write("-"*40 + "\n")
            f.write("Dominance is determined using multiple objective criteria:\n")
            f.write("- Median threshold method\n")
            f.write("- Natural break detection using gradient analysis\n")
            f.write("- Improved elbow method with cumulative variance\n")
            f.write("- Statistical significance (mean + 0.5*std)\n")
            f.write("Consensus: A frame is dominant if selected by ≥2 methods\n")
            f.write(f"Final dominance threshold: {overall['analysis_details']['dominance_threshold']:.3f}\n\n")
            
            # Overall Analysis Details
            f.write("OVERALL ANALYSIS DETAILS\n")
            f.write("-"*40 + "\n")
            f.write("Frame Dominance Scores:\n")
            scores = overall['dominance_scores'].sort_values('dominance_score', ascending=False)
            f.write(f"{'Frame':<15} {'Score':<10} {'Status':<15} {'Rank':<5}\n")
            f.write("-"*45 + "\n")
            for i, (frame, row) in enumerate(scores.iterrows()):
                status = "DOMINANT" if frame in overall['dominant_frames'] else "non-dominant"
                f.write(f"{FRAME_NAMES[frame]:<15} {row['dominance_score']:<10.3f} {status:<15} {i+1:<5}\n")
            
            f.write("\nMethod-Specific Scores:\n")
            for method in ['information_theory_score', 'network_analysis_score', 
                         'causality_score', 'proportional_score']:
                f.write(f"\n{method.replace('_', ' ').title()}:\n")
                for frame in scores.index[:5]:  # Top 5
                    f.write(f"  {FRAME_NAMES[frame]:<15}: {scores.loc[frame, method]:.3f}\n")
            
            # Temporal Evolution Statistics
            f.write("\n\nTEMPORAL EVOLUTION STATISTICS\n")
            f.write("-"*40 + "\n")
            
            yearly_data = self.temporal_results[
                self.temporal_results['period'].str.match(r'^\d{4}$', na=False)
            ].copy()
            
            # Add year column from period
            yearly_data['year'] = yearly_data['period'].astype(int)
            
            f.write(f"Total years analyzed: {len(yearly_data)}\n")
            f.write(f"Years with data: {yearly_data['year'].min()} - {yearly_data['year'].max()}\n")
            
            # Paradigm size distribution
            size_dist = yearly_data['n_dominant_frames'].value_counts().sort_index()
            f.write("\nParadigm Size Distribution:\n")
            for size, count in size_dist.items():
                pct = (count / len(yearly_data)) * 100
                f.write(f"  {size} frames: {count} years ({pct:.1f}%)\n")
            
            # Frame dominance frequency
            f.write("\nFrame Dominance Frequency:\n")
            frame_dominance_count = {frame: 0 for frame in FRAME_NAMES.keys()}
            for _, row in yearly_data.iterrows():
                if row['dominant_frames']:
                    frames = row['dominant_frames'].split(', ')
                    for frame in frames:
                        if frame in frame_dominance_count:
                            frame_dominance_count[frame] += 1
            
            sorted_frames = sorted(frame_dominance_count.items(), 
                                 key=lambda x: x[1], reverse=True)
            for frame, count in sorted_frames:
                pct = (count / len(yearly_data)) * 100
                f.write(f"  {FRAME_NAMES[frame]:<15}: {count} years ({pct:.1f}%)\n")
            
            # Paradigm transitions
            f.write("\n\nPARADIGM TRANSITIONS\n")
            f.write("-"*40 + "\n")
            
            transitions = []
            transition_types = {'expansion': 0, 'contraction': 0, 'substitution': 0}
            
            for i in range(1, len(yearly_data)):
                prev = set(yearly_data.iloc[i-1]['dominant_frames'].split(', '))
                curr = set(yearly_data.iloc[i]['dominant_frames'].split(', '))
                
                if prev != curr:
                    year = yearly_data.iloc[i]['period']
                    added = curr - prev
                    removed = prev - curr
                    
                    # Classify transition
                    if len(curr) > len(prev):
                        transition_types['expansion'] += 1
                    elif len(curr) < len(prev):
                        transition_types['contraction'] += 1
                    else:
                        transition_types['substitution'] += 1
                    
                    transitions.append({
                        'year': year,
                        'added': added,
                        'removed': removed,
                        'prev_size': len(prev),
                        'curr_size': len(curr)
                    })
            
            f.write(f"Total transitions: {len(transitions)}\n")
            f.write(f"Transition types:\n")
            for t_type, count in transition_types.items():
                f.write(f"  {t_type.capitalize()}: {count}\n")
            
            f.write("\nDetailed Transitions (first 20):\n")
            for trans in transitions[:20]:
                f.write(f"\n{trans['year']} ({trans['prev_size']}→{trans['curr_size']} frames):\n")
                if trans['added']:
                    f.write(f"  Added: {', '.join([FRAME_NAMES[f] for f in trans['added']])}\n")
                if trans['removed']:
                    f.write(f"  Removed: {', '.join([FRAME_NAMES[f] for f in trans['removed']])}\n")
            
            # Network Evolution Statistics
            f.write("\n\nNETWORK EVOLUTION STATISTICS\n")
            f.write("-"*40 + "\n")
            
            if self.network_evolution:
                years = sorted(self.network_evolution.keys())
                f.write(f"Network analyzed for {len(years)} time points\n")
                f.write(f"Time range: {years[0]} - {years[-1]}\n\n")
                
                # Average network metrics over time
                avg_edges = np.mean([data['n_edges'] for data in self.network_evolution.values()])
                avg_degree = np.mean([data['avg_degree'] for data in self.network_evolution.values()])
                
                f.write(f"Average number of edges: {avg_edges:.1f}\n")
                f.write(f"Average degree: {avg_degree:.2f}\n")
                
                # Most central frames over time
                centrality_scores = {frame: [] for frame in FRAME_NAMES.keys()}
                for year, data in self.network_evolution.items():
                    for frame, score in data['pagerank'].items():
                        centrality_scores[frame].append(score)
                
                avg_centrality = {frame: np.mean(scores) for frame, scores in centrality_scores.items()}
                sorted_centrality = sorted(avg_centrality.items(), key=lambda x: x[1], reverse=True)
                
                f.write("\nAverage PageRank Centrality:\n")
                for frame, score in sorted_centrality:
                    f.write(f"  {FRAME_NAMES[frame]:<15}: {score:.3f}\n")
            
            # Statistical Tests
            f.write("\n\nSTATISTICAL TESTS\n")
            f.write("-"*40 + "\n")
            
            # Diversity trend test
            from scipy import stats
            years = yearly_data['year'].values
            diversity = yearly_data['shannon_diversity'].values
            
            if len(years) > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, diversity)
                f.write(f"Shannon Diversity Trend:\n")
                f.write(f"  Slope: {slope:.6f} per year\n")
                f.write(f"  R²: {r_value**2:.3f}\n")
                f.write(f"  p-value: {p_value:.4f}\n")
                f.write(f"  Trend: {'Increasing' if slope > 0 else 'Decreasing'} ")
                f.write(f"({'Significant' if p_value < 0.05 else 'Not significant'})\n")
            
            # Key Insights
            f.write("\n\nKEY INSIGHTS\n")
            f.write("-"*40 + "\n")
            
            # Most stable paradigm period
            paradigm_lengths = []
            current_paradigm = yearly_data.iloc[0]['dominant_frames']
            current_length = 1
            start_year = int(yearly_data.iloc[0]['period'])
            
            for i in range(1, len(yearly_data)):
                if yearly_data.iloc[i]['dominant_frames'] == current_paradigm:
                    current_length += 1
                else:
                    paradigm_lengths.append((current_paradigm, current_length, start_year))
                    current_paradigm = yearly_data.iloc[i]['dominant_frames']
                    current_length = 1
                    start_year = int(yearly_data.iloc[i]['period'])
            paradigm_lengths.append((current_paradigm, current_length, start_year))
            
            longest_paradigm = max(paradigm_lengths, key=lambda x: x[1])
            f.write(f"Most stable paradigm period:\n")
            f.write(f"  Frames: {longest_paradigm[0]}\n")
            f.write(f"  Duration: {longest_paradigm[1]} years\n")
            f.write(f"  Period: {longest_paradigm[2]}-{longest_paradigm[2]+longest_paradigm[1]-1}\n")
            
            # Current trajectory
            recent_years = yearly_data.tail(5)
            f.write(f"\nRecent trajectory (last 5 years):\n")
            f.write(f"  Average paradigm size: {recent_years['n_dominant_frames'].mean():.1f} frames\n")
            f.write(f"  Diversity trend: {'Increasing' if recent_years['shannon_diversity'].iloc[-1] > recent_years['shannon_diversity'].iloc[0] else 'Decreasing'}\n")
            
            # Most volatile period
            volatility = []
            for i in range(1, len(yearly_data)-1):
                prev_frames = set(yearly_data.iloc[i-1]['dominant_frames'].split(', '))
                curr_frames = set(yearly_data.iloc[i]['dominant_frames'].split(', '))
                next_frames = set(yearly_data.iloc[i+1]['dominant_frames'].split(', '))
                
                changes = len(prev_frames.symmetric_difference(curr_frames)) + \
                         len(curr_frames.symmetric_difference(next_frames))
                volatility.append((int(yearly_data.iloc[i]['period']), changes))
            
            if volatility:
                most_volatile = max(volatility, key=lambda x: x[1])
                f.write(f"\nMost volatile year: {most_volatile[0]} ({most_volatile[1]} frame changes)\n")
        
        print(f"✓ Statistical report saved to: {report_path}")
    
    def _get_sort_key(self, period):
        """Get sort key for period string."""
        if period == 'Overall':
            return (0, 0)
        elif '-' in period and period != '2020-present':
            return (1, int(period.split('-')[0]))
        elif period == '2020-present':
            return (1, 2020)
        else:
            try:
                return (2, int(period))
            except ValueError:
                return (3, 0)
    
    def run_analysis(self, force_recalculate: bool = False):
        """Run complete paradigm analysis."""
        print("="*80)
        print("COMPREHENSIVE CLIMATE COVERAGE PARADIGM ANALYSIS")
        print(f"Using {self.n_workers} parallel workers")
        print("="*80)
        
        total_start_time = time.time()
        
        # Load data
        df = self.load_data(force_reload=force_recalculate)
        
        # Run analyses
        overall_analysis = self.analyze_overall_paradigm(df)
        temporal_results = self.analyze_temporal_paradigm(df)
        network_evolution = self.analyze_network_evolution(df)
        weekly_paradigm = self.analyze_weekly_paradigm(df)
        
        # Generate statistical report
        self.generate_statistical_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total time elapsed: {time.time() - total_start_time:.1f} seconds")
        print(f"\nOutput files:")
        print(f"  Data directory: {DATA_DIR}")
        print(f"  Results directory: {RESULTS_DIR}")
        
        return self.results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze climate coverage paradigms comprehensively'
    )
    parser.add_argument('--force-recalculate', action='store_true',
                       help='Force recalculation of all metrics')
    args = parser.parse_args()
    
    analyzer = ComprehensiveParadigmAnalyzer()
    results = analyzer.run_analysis(force_recalculate=args.force_recalculate)
    
    print("\nAnalysis Summary:")
    print(f"Overall paradigm: {results['overall']['paradigm_type']}")
    print(f"Dominant frames: {', '.join([FRAME_NAMES[f] for f in results['overall']['dominant_frames']])}")


if __name__ == "__main__":
    # Fix for macOS multiprocessing with Python 3.13
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()