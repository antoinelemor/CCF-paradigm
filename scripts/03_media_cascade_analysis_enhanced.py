"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
03_media_cascade_analysis_enhanced.py

MAIN OBJECTIVE:
---------------
Enhanced media cascade analysis with comprehensive improvements:
1. Fine-grained source/messenger proportion analysis
2. EMD-based adoption detection without arbitrary thresholds  
3. Adaptive correlation thresholds with co-occurrence and temporal proximity
4. NER entity analysis for epistemic authority detection
5. Multi-level cascade indicators with equitable sub-indices

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- networkx
- tqdm
- All dependencies from cascade_analysis_enhanced module

MAIN FEATURES:
--------------
1) Enhanced source citation network analysis
2) Proportion-based adoption patterns using EMD
3) Adaptive thresholds for correlation networks
4) NER entity extraction and epistemic authority identification
5) Multi-index cascade scoring with sub-indices
6) Comprehensive data export for validation
7) Detailed reporting with all new metrics

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
import networkx as nx
import traceback
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.database_access.db_connector import DatabaseConnector
from src.database_access.data_processor import FrameDataProcessor
from src.cascade_analysis_enhanced import EnhancedCascadeDetector, EnhancedMediaCascade

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
TREND_DATA_DIR = project_root / "data" / "03_events_effects" / "trends_data"
CASCADE_DATA_DIR = project_root / "data" / "03_events_effects" / "cascade_enhanced"
CASCADE_DATA_DIR.mkdir(parents=True, exist_ok=True)


class EnhancedCascadeAnalyzer:
    """Enhanced cascade analyzer with comprehensive improvements."""
    
    def __init__(self):
        """Initialize enhanced analyzer."""
        self.db_connector = DatabaseConnector()
        self.processor = FrameDataProcessor()
        self.detector = EnhancedCascadeDetector(similarity_threshold=85)
        
        # Storage
        self.frame_data = None
        self.critical_points = None
        self.critical_cascades = []
        self.spontaneous_cascades = []
        self.all_cascades = []
        
    def load_trend_data(self) -> pd.DataFrame:
        """Load critical points from trend detection."""
        print("Loading trend detection results...")
        
        critical_points_path = TREND_DATA_DIR / "critical_points_emd.csv"
        if not critical_points_path.exists():
            alt_path = project_root / "data" / "03_events_effects" / "critical_trend_dates.csv"
            if alt_path.exists():
                critical_points_path = alt_path
            else:
                raise FileNotFoundError(
                    "Critical points data not found. Run trend detection analysis first."
                )
        
        critical_points = pd.read_csv(critical_points_path)
        critical_points['date'] = pd.to_datetime(critical_points['date'])
        
        print(f"Loaded {len(critical_points)} critical points")
        
        if 'type' in critical_points.columns:
            self.critical_points = critical_points[critical_points['type'].str.contains('start')]
        else:
            self.critical_points = critical_points
            
        print(f"Using {len(self.critical_points)} trend start points for cascade analysis")
        
        return self.critical_points
    
    def load_frame_data(self) -> pd.DataFrame:
        """Load complete frame detection data."""
        print("Loading frame detection data from database...")
        
        # Get data
        df = self.db_connector.get_frame_data(show_progress=True)
        
        # Process data
        print("Processing frame data...")
        df = self.processor.process_frame_data(df, show_progress=True)
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Add week and month columns
        df['week'] = df['date'].dt.to_period('W').dt.to_timestamp()
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        
        self.frame_data = df
        print(f"Loaded {len(df):,} sentences from {df['doc_id'].nunique():,} articles")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Check for NER entities column
        if 'ner_entities' in df.columns:
            print("✓ NER entities column found - will perform entity analysis")
        else:
            print("⚠ NER entities column not found - entity analysis will be limited")
        
        return df
    
    def analyze_critical_cascades(self) -> List[EnhancedMediaCascade]:
        """Analyze cascades at critical trend points with enhanced metrics."""
        print("\n" + "="*80)
        print("ANALYZING CASCADES AT CRITICAL POINTS (ENHANCED)")
        print("="*80)
        
        cascades = []
        errors = []
        
        # Group critical points by frame
        for frame in FRAME_NAMES:
            if 'frame' in self.critical_points.columns:
                frame_points = self.critical_points[self.critical_points['frame'] == frame]
            else:
                frame_points = self.critical_points
            
            if len(frame_points) == 0:
                continue
            
            print(f"\nAnalyzing {frame} frame ({len(frame_points)} critical points)...")
            print("  Using enhanced detection with:")
            print("    - Source proportion analysis")
            print("    - EMD-based adoption patterns")
            print("    - Adaptive correlation thresholds")
            print("    - NER entity networks")
            
            for idx, point in tqdm(frame_points.iterrows(), total=len(frame_points),
                                desc=f"Processing {frame}"):
                try:
                    # Check data availability
                    date_window_start = point['date'] - pd.Timedelta(days=90)
                    date_window_end = point['date'] + pd.Timedelta(days=90)
                    
                    window_data = self.frame_data[
                        (self.frame_data['date'] >= date_window_start) & 
                        (self.frame_data['date'] <= date_window_end)
                    ]
                    
                    if len(window_data) < 100:
                        print(f"  ⚠ Skipping {point['date'].strftime('%Y-%m-%d')}: "
                            f"insufficient data ({len(window_data)} rows)")
                        continue
                    
                    frame_col = f"{frame}_Detection"
                    if frame_col not in window_data.columns:
                        print(f"  ⚠ Skipping {point['date'].strftime('%Y-%m-%d')}: "
                            f"frame column {frame_col} not found")
                        continue
                    
                    frame_detections = pd.to_numeric(
                        window_data[frame_col], errors='coerce'
                    ).fillna(0).sum()
                    
                    if frame_detections == 0:
                        print(f"  ⚠ Skipping {point['date'].strftime('%Y-%m-%d')}: "
                            f"no {frame} detections in window")
                        continue
                    
                    # Detect cascade with enhanced metrics
                    cascade = self.detector.detect_cascade(
                        self.frame_data, 
                        frame, 
                        point['date'],
                        cascade_type='critical_point',
                        window_days=180
                    )
                    
                    if cascade:
                        cascades.append(cascade)
                        
                        if cascade.cascade_strength != 'weak':
                            # Report enhanced metrics
                            print(f"  ✓ Enhanced cascade detected at {point['date'].strftime('%Y-%m-%d')}:")
                            print(f"      Strength: {cascade.cascade_strength} (score={cascade.total_score:.3f})")
                            print(f"      EMD adoption: {cascade.metrics.adoption_emd_score:.3f}")
                            print(f"      Source network density: {cascade.metrics.source_network_density:.3f}")
                            print(f"      Entity convergence: {cascade.metrics.entity_convergence:.3f}")
                            print(f"      Adaptive threshold: {cascade.metrics.adaptive_threshold:.3f}")
                            
                            # Report epistemic authorities if found
                            if cascade.metrics.epistemic_authorities:
                                top_authorities = cascade.metrics.epistemic_authorities[:3]
                                print(f"      Top epistemic authorities:")
                                for auth, score in top_authorities:
                                    print(f"        - {auth}: {score:.2f}")
                
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    print(f"  ✗ Error analyzing {frame} at {point['date'].strftime('%Y-%m-%d')}: {error_msg}")
                    errors.append((frame, point['date'], error_msg))
                    continue
        
        if errors:
            print(f"\n⚠ Total errors encountered: {len(errors)}")
        
        self.critical_cascades = cascades
        
        print(f"\n" + "="*40)
        print(f"ENHANCED CASCADE DETECTION SUMMARY")
        print(f"="*40)
        print(f"Total cascades at critical points: {len(cascades)}")
        print(f"Strong cascades: {sum(1 for c in cascades if c.cascade_strength == 'strong')}")
        print(f"Moderate cascades: {sum(1 for c in cascades if c.cascade_strength == 'moderate')}")
        print(f"With significant EMD adoption (>0.5): {sum(1 for c in cascades if c.metrics.adoption_emd_score > 0.5)}")
        print(f"With entity convergence (>0.3): {sum(1 for c in cascades if c.metrics.entity_convergence > 0.3)}")
        print(f"Consensus significant: {sum(1 for c in cascades if c.metrics.consensus_significant)}")
        
        return cascades
    
    def analyze_spontaneous_cascades(self) -> List[EnhancedMediaCascade]:
        """Detect spontaneous cascades with enhanced metrics."""
        print("\n" + "="*80)
        print("DETECTING SPONTANEOUS CASCADES (ENHANCED)")
        print("="*80)
        
        spontaneous = []
        
        # Create exclusion periods
        exclude_periods = []
        for cascade in self.critical_cascades:
            exclude_periods.append((
                cascade.metrics.onset_date,
                cascade.metrics.end_date
            ))
        
        print(f"Excluding {len(exclude_periods)} critical cascade periods")
        
        for frame in FRAME_NAMES:
            print(f"\nSearching for spontaneous cascades in {frame} frame...")
            
            frame_spontaneous = self.detector.detect_spontaneous_cascades(
                self.frame_data,
                frame,
                exclude_periods,
                min_threshold=0.05
            )
            
            if frame_spontaneous:
                print(f"  Found {len(frame_spontaneous)} spontaneous cascades")
                for cascade in frame_spontaneous:
                    print(f"    - {cascade.metrics.onset_date.strftime('%Y-%m-%d')} to "
                        f"{cascade.metrics.end_date.strftime('%Y-%m-%d')}: "
                        f"{cascade.cascade_strength} (score={cascade.total_score:.3f}, "
                        f"EMD={cascade.metrics.adoption_emd_score:.3f})")
                
                spontaneous.extend(frame_spontaneous)
        
        self.spontaneous_cascades = spontaneous
        print(f"\nTotal spontaneous cascades: {len(spontaneous)}")
        
        return spontaneous
    
    def export_enhanced_cascade_data(self):
        """Export enhanced cascade analysis results."""
        print("\n" + "="*80)
        print("EXPORTING ENHANCED CASCADE DATA")
        print("="*80)
        
        self.all_cascades = self.critical_cascades + self.spontaneous_cascades
        
        # 1. Main cascade results with enhanced metrics
        cascade_data = []
        for cascade in self.all_cascades:
            cascade_dict = {
                'frame': cascade.frame,
                'cascade_type': cascade.cascade_type,
                'reference_date': cascade.reference_date,
                'onset_date': cascade.metrics.onset_date,
                'inflection_date': cascade.metrics.inflection_date,
                'peak_date': cascade.metrics.peak_date,
                'end_date': cascade.metrics.end_date,
                'duration_days': cascade.metrics.duration_days,
                'duration_days_transformed': cascade.metrics.duration_days_transformed,
                
                # Velocity metrics
                'initial_velocity': cascade.metrics.initial_velocity,
                'initial_velocity_robust': cascade.metrics.initial_velocity_robust,
                'peak_velocity': cascade.metrics.peak_velocity,
                'acceleration': cascade.metrics.acceleration,
                'momentum': cascade.metrics.momentum,
                
                # Component scores
                'journalist_score': cascade.journalist_score,
                'media_score': cascade.media_score,
                'intensity_score': cascade.intensity_score,
                'network_score': cascade.network_score,
                'virality_score': cascade.virality_score,
                'total_score': cascade.total_score,
                'cascade_strength': cascade.cascade_strength,
                
                # Enhanced network metrics
                'network_density': cascade.metrics.network_density,
                'network_density_transformed': cascade.metrics.network_density_transformed,
                'clustering_coefficient': cascade.metrics.clustering_coefficient,
                'avg_path_length': cascade.metrics.path_length,
                'correlation_density': cascade.metrics.correlation_density,
                'correlation_density_transformed': cascade.metrics.correlation_density_transformed,
                'mean_correlation': cascade.metrics.mean_correlation,
                'co_occurrence_density': cascade.metrics.co_occurrence_density,
                'temporal_proximity_score': cascade.metrics.temporal_proximity_score,
                'adaptive_threshold': cascade.metrics.adaptive_threshold,
                
                # Enhanced source metrics
                'source_diversity': cascade.metrics.source_diversity,
                'source_convergence': cascade.metrics.source_convergence,
                'source_network_density': cascade.metrics.source_network_density,
                'n_epistemic_authorities': len(cascade.metrics.epistemic_authorities),
                
                # Entity metrics
                'entity_convergence': cascade.metrics.entity_convergence,
                'n_shared_entities': len(cascade.metrics.shared_entities),
                
                # EMD adoption metrics
                'adoption_emd_score': cascade.metrics.adoption_emd_score,
                'proportion_change_rate': cascade.metrics.proportion_change_rate,
                
                # Homogenization metrics
                'homogenization_score': cascade.metrics.homogenization_score,
                'homogenization_shannon': cascade.metrics.homogenization_shannon,
                'entropy_trend': cascade.metrics.entropy_trend,
                'entropy_trend_robust': cascade.metrics.entropy_trend_robust,
                
                # Other metrics
                'n_journalists': len(cascade.metrics.top_journalists),
                'n_media': len(cascade.metrics.top_media),
                'emotional_intensity': cascade.metrics.emotional_intensity,
                'viral_coefficient': cascade.metrics.viral_coefficient,
                'consensus_significant': cascade.metrics.consensus_significant,
                'confidence': cascade.confidence
            }
            
            cascade_data.append(cascade_dict)
        
        # Save main results
        cascade_df = pd.DataFrame(cascade_data)
        cascade_path = CASCADE_DATA_DIR / "enhanced_cascade_results.csv"
        cascade_df.to_csv(cascade_path, index=False)
        print(f"  ✓ Saved enhanced cascade results to {cascade_path}")
        
        # 2. Sub-indices breakdown
        sub_indices_data = []
        for cascade in self.all_cascades:
            for component, indices in cascade.sub_indices.items():
                for sub_index, value in indices.items():
                    sub_indices_data.append({
                        'frame': cascade.frame,
                        'cascade_type': cascade.cascade_type,
                        'cascade_date': cascade.reference_date,
                        'component': component,
                        'sub_index': sub_index,
                        'value': value,
                        'cascade_strength': cascade.cascade_strength
                    })
        
        if sub_indices_data:
            sub_indices_df = pd.DataFrame(sub_indices_data)
            sub_indices_path = CASCADE_DATA_DIR / "cascade_sub_indices.csv"
            sub_indices_df.to_csv(sub_indices_path, index=False)
            print(f"  ✓ Saved sub-indices to {sub_indices_path}")
        
        # 3. Epistemic authorities
        authority_data = []
        for cascade in self.all_cascades:
            for authority, score in cascade.metrics.epistemic_authorities:
                entity_type, entity_name = authority.split(':', 1)
                authority_data.append({
                    'frame': cascade.frame,
                    'cascade_type': cascade.cascade_type,
                    'cascade_date': cascade.reference_date,
                    'entity_type': entity_type,
                    'entity_name': entity_name,
                    'authority_score': score,
                    'cascade_strength': cascade.cascade_strength
                })
        
        if authority_data:
            authority_df = pd.DataFrame(authority_data)
            authority_path = CASCADE_DATA_DIR / "epistemic_authorities.csv"
            authority_df.to_csv(authority_path, index=False)
            print(f"  ✓ Saved epistemic authorities to {authority_path}")
        
        # 4. Source proportions
        source_prop_data = []
        for cascade in self.all_cascades:
            for source_type, proportion in cascade.metrics.source_proportions_by_article.items():
                source_prop_data.append({
                    'frame': cascade.frame,
                    'cascade_type': cascade.cascade_type,
                    'cascade_date': cascade.reference_date,
                    'source_type': source_type,
                    'mean_proportion': proportion,
                    'cascade_strength': cascade.cascade_strength
                })
        
        if source_prop_data:
            source_prop_df = pd.DataFrame(source_prop_data)
            source_prop_path = CASCADE_DATA_DIR / "source_proportions.csv"
            source_prop_df.to_csv(source_prop_path, index=False)
            print(f"  ✓ Saved source proportions to {source_prop_path}")
        
        # 5. Shared entities
        entity_data = []
        for cascade in self.all_cascades:
            for entity, count in cascade.metrics.shared_entities.items():
                entity_data.append({
                    'frame': cascade.frame,
                    'cascade_type': cascade.cascade_type,
                    'cascade_date': cascade.reference_date,
                    'entity': entity,
                    'shared_count': count,
                    'cascade_strength': cascade.cascade_strength
                })
        
        if entity_data:
            entity_df = pd.DataFrame(entity_data)
            entity_path = CASCADE_DATA_DIR / "shared_entities.csv"
            entity_df.to_csv(entity_path, index=False)
            print(f"  ✓ Saved shared entities to {entity_path}")
        
        # 6. Network graphs (enhanced)
        self.export_enhanced_networks()
    
    def export_enhanced_networks(self):
        """Export enhanced network graphs."""
        print("\n  Exporting enhanced network graphs...")
        
        networks_dir = CASCADE_DATA_DIR / "networks"
        networks_dir.mkdir(exist_ok=True)
        
        for cascade in self.all_cascades:
            if cascade.cascade_strength in ['strong', 'moderate']:
                date_str = cascade.reference_date.strftime('%Y%m%d')
                
                # Helper function to convert timestamps
                def convert_timestamps_in_graph(G):
                    G_copy = G.copy()
                    
                    for node, attrs in G_copy.nodes(data=True):
                        for key, value in attrs.items():
                            if isinstance(value, pd.Timestamp):
                                G_copy.nodes[node][key] = value.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(value, (pd.Period, datetime)):
                                G_copy.nodes[node][key] = str(value)
                    
                    for u, v, attrs in G_copy.edges(data=True):
                        for key, value in attrs.items():
                            if isinstance(value, pd.Timestamp):
                                G_copy.edges[u, v][key] = value.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(value, (pd.Period, datetime)):
                                G_copy.edges[u, v][key] = str(value)
                    
                    return G_copy
                
                # Export adoption network
                if len(cascade.network_graph) > 0:
                    filename = f"adoption_{cascade.frame}_{cascade.cascade_type}_{date_str}.gexf"
                    filepath = networks_dir / filename
                    network_to_export = convert_timestamps_in_graph(cascade.network_graph)
                    nx.write_gexf(network_to_export, str(filepath))
                
                # Export correlation network
                if len(cascade.correlation_network) > 0:
                    filename = f"correlation_{cascade.frame}_{cascade.cascade_type}_{date_str}.gexf"
                    filepath = networks_dir / filename
                    network_to_export = convert_timestamps_in_graph(cascade.correlation_network)
                    nx.write_gexf(network_to_export, str(filepath))
                
                # Export source citation network
                if hasattr(cascade, 'source_citation_network') and len(cascade.source_citation_network) > 0:
                    filename = f"source_{cascade.frame}_{cascade.cascade_type}_{date_str}.gexf"
                    filepath = networks_dir / filename
                    network_to_export = convert_timestamps_in_graph(cascade.source_citation_network)
                    nx.write_gexf(network_to_export, str(filepath))
                
                # Export entity network
                if hasattr(cascade, 'entity_network') and len(cascade.entity_network) > 0:
                    filename = f"entity_{cascade.frame}_{cascade.cascade_type}_{date_str}.gexf"
                    filepath = networks_dir / filename
                    network_to_export = convert_timestamps_in_graph(cascade.entity_network)
                    nx.write_gexf(network_to_export, str(filepath))
        
        print(f"    ✓ Exported network graphs to {networks_dir}")
    
    def generate_enhanced_summary(self):
        """Generate enhanced summary statistics."""
        print("\n  Generating enhanced summary statistics...")
        
        summary = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_type': 'enhanced_comprehensive',
            'data_period': {
                'start': str(self.frame_data['date'].min()),
                'end': str(self.frame_data['date'].max())
            },
            'total_cascades': len(self.all_cascades),
            'critical_cascades': len(self.critical_cascades),
            'spontaneous_cascades': len(self.spontaneous_cascades),
            'by_strength': {
                'strong': sum(1 for c in self.all_cascades if c.cascade_strength == 'strong'),
                'moderate': sum(1 for c in self.all_cascades if c.cascade_strength == 'moderate'),
                'weak': sum(1 for c in self.all_cascades if c.cascade_strength == 'weak')
            },
            'consensus_significant': sum(1 for c in self.all_cascades if c.metrics.consensus_significant),
            'enhanced_metrics': {
                'avg_emd_adoption': np.mean([c.metrics.adoption_emd_score for c in self.all_cascades]),
                'avg_source_network_density': np.mean([c.metrics.source_network_density for c in self.all_cascades]),
                'avg_entity_convergence': np.mean([c.metrics.entity_convergence for c in self.all_cascades]),
                'avg_adaptive_threshold': np.mean([c.metrics.adaptive_threshold for c in self.all_cascades]),
                'total_epistemic_authorities': sum(len(c.metrics.epistemic_authorities) for c in self.all_cascades),
                'total_shared_entities': sum(len(c.metrics.shared_entities) for c in self.all_cascades)
            },
            'by_frame': {}
        }
        
        # Frame-specific statistics
        for frame in FRAME_NAMES:
            frame_cascades = [c for c in self.all_cascades if c.frame == frame]
            if frame_cascades:
                summary['by_frame'][frame] = {
                    'total': len(frame_cascades),
                    'critical': sum(1 for c in frame_cascades if c.cascade_type == 'critical_point'),
                    'spontaneous': sum(1 for c in frame_cascades if c.cascade_type == 'spontaneous'),
                    'strong': sum(1 for c in frame_cascades if c.cascade_strength == 'strong'),
                    'avg_score': np.mean([c.total_score for c in frame_cascades]),
                    'avg_duration': np.mean([c.metrics.duration_days for c in frame_cascades]),
                    'avg_emd_adoption': np.mean([c.metrics.adoption_emd_score for c in frame_cascades]),
                    'avg_source_density': np.mean([c.metrics.source_network_density for c in frame_cascades]),
                    'avg_entity_convergence': np.mean([c.metrics.entity_convergence for c in frame_cascades]),
                    'avg_journalists': np.mean([len(c.metrics.top_journalists) for c in frame_cascades]),
                    'avg_media': np.mean([len(c.metrics.top_media) for c in frame_cascades])
                }
        
        # Save summary
        summary_path = CASCADE_DATA_DIR / "enhanced_cascade_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"    ✓ Saved enhanced summary to {summary_path}")
        
        return summary
    
    def generate_enhanced_report(self):
        """Generate comprehensive enhanced analysis report."""
        report_path = CASCADE_DATA_DIR / "enhanced_cascade_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED MEDIA CASCADE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Period: {self.frame_data['date'].min()} to {self.frame_data['date'].max()}\n\n")
            
            f.write("ENHANCED METHODOLOGY\n")
            f.write("-"*40 + "\n")
            f.write("1. Fine-grained source/messenger proportion analysis\n")
            f.write("   - Average proportion per article per source type\n")
            f.write("   - Correlation analysis between journalists/media\n")
            f.write("   - Source citation network construction\n\n")
            
            f.write("2. EMD-based adoption detection without thresholds\n")
            f.write("   - Earth Mover's Distance for proportion changes\n")
            f.write("   - No arbitrary adoption thresholds\n")
            f.write("   - Continuous measurement of adoption strength\n\n")
            
            f.write("3. Adaptive correlation thresholds\n")
            f.write("   - Data-driven threshold determination (25th percentile)\n")
            f.write("   - Co-occurrence analysis (same article)\n")
            f.write("   - Temporal proximity edges (consecutive days)\n\n")
            
            f.write("4. NER entity analysis\n")
            f.write("   - Extraction of persons and organizations\n")
            f.write("   - 95% similarity matching for entity correspondence\n")
            f.write("   - Epistemic authority identification\n")
            f.write("   - Entity-based network construction\n\n")
            
            f.write("5. Multi-level cascade indicators\n")
            f.write("   - 5 main components (20% each)\n")
            f.write("   - Multiple sub-indices per component\n")
            f.write("   - Equitable distribution of weights\n\n")
            
            f.write("OVERALL RESULTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total cascades detected: {len(self.all_cascades)}\n")
            f.write(f"  At critical points: {len(self.critical_cascades)}\n")
            f.write(f"  Spontaneous: {len(self.spontaneous_cascades)}\n\n")
            
            if self.all_cascades:
                f.write("Enhanced Metrics Summary:\n")
                f.write(f"  Average EMD adoption score: {np.mean([c.metrics.adoption_emd_score for c in self.all_cascades]):.3f}\n")
                f.write(f"  Average source network density: {np.mean([c.metrics.source_network_density for c in self.all_cascades]):.3f}\n")
                f.write(f"  Average entity convergence: {np.mean([c.metrics.entity_convergence for c in self.all_cascades]):.3f}\n")
                f.write(f"  Average adaptive threshold: {np.mean([c.metrics.adaptive_threshold for c in self.all_cascades]):.3f}\n")
                f.write(f"  Total epistemic authorities identified: {sum(len(c.metrics.epistemic_authorities) for c in self.all_cascades)}\n")
                f.write(f"  Total shared entities: {sum(len(c.metrics.shared_entities) for c in self.all_cascades)}\n\n")
            
            f.write("Cascade Strength Distribution:\n")
            strong = sum(1 for c in self.all_cascades if c.cascade_strength == 'strong')
            moderate = sum(1 for c in self.all_cascades if c.cascade_strength == 'moderate')
            weak = sum(1 for c in self.all_cascades if c.cascade_strength == 'weak')
            
            if self.all_cascades:
                f.write(f"  Strong: {strong} ({strong/len(self.all_cascades)*100:.1f}%)\n")
                f.write(f"  Moderate: {moderate} ({moderate/len(self.all_cascades)*100:.1f}%)\n")
                f.write(f"  Weak: {weak} ({weak/len(self.all_cascades)*100:.1f}%)\n\n")
            
            # Frame-specific results
            f.write("FRAME-SPECIFIC RESULTS\n")
            f.write("-"*40 + "\n")
            
            for frame in FRAME_NAMES:
                frame_cascades = [c for c in self.all_cascades if c.frame == frame]
                if not frame_cascades:
                    continue
                
                f.write(f"\n{frame} Frame:\n")
                f.write("="*20 + "\n")
                f.write(f"Total cascades: {len(frame_cascades)}\n")
                
                critical = [c for c in frame_cascades if c.cascade_type == 'critical_point']
                spontaneous = [c for c in frame_cascades if c.cascade_type == 'spontaneous']
                f.write(f"  Critical point cascades: {len(critical)}\n")
                f.write(f"  Spontaneous cascades: {len(spontaneous)}\n")
                
                # Find strongest cascade
                if frame_cascades:
                    strongest = max(frame_cascades, key=lambda c: c.total_score)
                    f.write(f"\nStrongest cascade:\n")
                    f.write(f"  Date: {strongest.reference_date.strftime('%Y-%m-%d')}\n")
                    f.write(f"  Type: {strongest.cascade_type}\n")
                    f.write(f"  Score: {strongest.total_score:.3f}\n")
                    f.write(f"  Duration: {strongest.metrics.duration_days} days\n")
                    f.write(f"\n  Enhanced Metrics:\n")
                    f.write(f"    EMD adoption score: {strongest.metrics.adoption_emd_score:.3f}\n")
                    f.write(f"    Source network density: {strongest.metrics.source_network_density:.3f}\n")
                    f.write(f"    Entity convergence: {strongest.metrics.entity_convergence:.3f}\n")
                    f.write(f"    Co-occurrence density: {strongest.metrics.co_occurrence_density:.3f}\n")
                    f.write(f"    Temporal proximity: {strongest.metrics.temporal_proximity_score:.3f}\n")
                    f.write(f"    Adaptive threshold used: {strongest.metrics.adaptive_threshold:.3f}\n")
                    
                    # Report sub-indices
                    f.write(f"\n  Sub-indices breakdown:\n")
                    for component, indices in strongest.sub_indices.items():
                        f.write(f"    {component.capitalize()}:\n")
                        for sub_index, value in indices.items():
                            f.write(f"      - {sub_index}: {value:.3f}\n")
                    
                    # Top epistemic authorities
                    if strongest.metrics.epistemic_authorities:
                        f.write(f"\n  Top epistemic authorities:\n")
                        for auth, score in strongest.metrics.epistemic_authorities[:5]:
                            f.write(f"    - {auth}: {score:.2f}\n")
                    
                    # Top journalists and media
                    if strongest.metrics.top_journalists:
                        f.write(f"\n  Top journalists (by centrality):\n")
                        for journalist, score in strongest.metrics.top_journalists[:5]:
                            f.write(f"    - {journalist}: {score:.3f}\n")
                    
                    if strongest.metrics.top_media:
                        f.write(f"\n  Top media outlets:\n")
                        for media, score in strongest.metrics.top_media[:5]:
                            f.write(f"    - {media}: {score:.3f}\n")
            
            # Key findings
            f.write("\n\nKEY FINDINGS (ENHANCED ANALYSIS)\n")
            f.write("-"*40 + "\n")
            
            if self.all_cascades:
                # Most cascade-prone frame
                frame_counts = {}
                for frame in FRAME_NAMES:
                    frame_counts[frame] = len([c for c in self.all_cascades if c.frame == frame])
                
                if frame_counts:
                    most_active = max(frame_counts, key=frame_counts.get)
                    f.write(f"- Most cascade-prone frame: {most_active} ({frame_counts[most_active]} cascades)\n")
                
                # Cascades with highest EMD adoption
                high_emd = [c for c in self.all_cascades if c.metrics.adoption_emd_score > 0.7]
                f.write(f"- Cascades with high EMD adoption (>0.7): {len(high_emd)}\n")
                
                # Cascades with entity convergence
                high_entity = [c for c in self.all_cascades if c.metrics.entity_convergence > 0.5]
                f.write(f"- Cascades with high entity convergence (>0.5): {len(high_entity)}\n")
                
                # Average metrics
                avg_duration = np.mean([c.metrics.duration_days for c in self.all_cascades])
                avg_journalists = np.mean([len(c.metrics.top_journalists) for c in self.all_cascades])
                avg_media = np.mean([len(c.metrics.top_media) for c in self.all_cascades])
                avg_authorities = np.mean([len(c.metrics.epistemic_authorities) for c in self.all_cascades])
                
                f.write(f"- Average cascade duration: {avg_duration:.1f} days\n")
                f.write(f"- Average journalists involved: {avg_journalists:.1f}\n")
                f.write(f"- Average media outlets: {avg_media:.1f}\n")
                f.write(f"- Average epistemic authorities per cascade: {avg_authorities:.1f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF ENHANCED REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"    ✓ Generated enhanced report: {report_path}")
    
    def run_analysis(self):
        """Run complete enhanced cascade analysis pipeline."""
        print("="*80)
        print("ENHANCED MEDIA CASCADE ANALYSIS")
        print("="*80)
        print("\nThis analysis includes:")
        print("  • Fine-grained source/messenger proportion tracking")
        print("  • EMD-based adoption without arbitrary thresholds")
        print("  • Adaptive correlation thresholds with co-occurrence")
        print("  • NER entity extraction and epistemic authority detection")
        print("  • Multi-level cascade indicators with sub-indices")
        print()
        
        # Load data
        self.load_trend_data()
        self.load_frame_data()
        
        # Analyze cascades
        self.analyze_critical_cascades()
        self.analyze_spontaneous_cascades()
        
        # Export results
        self.export_enhanced_cascade_data()
        
        # Generate summary and report
        summary = self.generate_enhanced_summary()
        self.generate_enhanced_report()
        
        # Close database connection
        self.db_connector.close()
        
        print("\n" + "="*80)
        print("ENHANCED ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total cascades detected: {summary['total_cascades']}")
        print(f"  Critical point cascades: {summary['critical_cascades']}")
        print(f"  Spontaneous cascades: {summary['spontaneous_cascades']}")
        print(f"  Consensus significant: {summary['consensus_significant']}")
        print(f"\nEnhanced metrics summary:")
        for metric, value in summary['enhanced_metrics'].items():
            print(f"  {metric}: {value:.3f}")
        print(f"\nResults saved to: {CASCADE_DATA_DIR}")


def main():
    """Main execution function."""
    analyzer = EnhancedCascadeAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()