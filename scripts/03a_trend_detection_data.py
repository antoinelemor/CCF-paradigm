"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
03a_trend_detection_data_optimized.py

MAIN OBJECTIVE:
---------------
OPTIMIZED production script for trend detection data using EMD and fuzzy classification.
Leverages Mac M4 Max with 128GB RAM for massive parallel processing and in-memory operations.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- PyEMD
- skfuzzy
- dask
- numba
- All dependencies from trend_analysis module

MAIN FEATURES:
--------------
1) Ultra-fast parallel EMD-based trend detection
2) In-memory data processing with Dask
3) GPU-accelerated computations where possible
4) Intelligent caching and memoization
5) Multi-threaded frame analysis
6) Optimized for M4 Max architecture

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import pickle
import hashlib
import time

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.database_access.db_connector import OptimizedDatabaseConnector
from src.database_access.data_processor import FrameDataProcessor
from src.trend_analysis import EMDTrendAnalyzer, FuzzyTrend
from scipy.signal import savgol_filter

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
CACHE_DIR = project_root / "cache" / "trend_analysis"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optimization settings for M4 Max
N_CORES = min(mp.cpu_count(), 16)  # Use up to 16 cores
CHUNK_SIZE = 2_000_000  # Process 2M rows at a time
CACHE_SIZE_GB = 50  # Use 50GB for caching


class OptimizedTrendDataProducer:
    """Ultra-optimized trend data producer for M4 Max with 128GB RAM."""
    
    def __init__(self, use_cache: bool = True, parallel_frames: bool = True, enhanced_detection: bool = True):
        """
        Initialize optimized data producer.
        
        Args:
            use_cache: Whether to use disk caching for intermediate results
            parallel_frames: Whether to process frames in parallel
            enhanced_detection: Whether to use enhanced detection methods
        """
        print(f"Initializing Optimized Trend Data Producer")
        print(f"System: Mac M4 Max with 128GB RAM")
        print(f"Using {N_CORES} CPU cores for parallel processing")
        print(f"Cache size: {CACHE_SIZE_GB}GB")
        print(f"Enhanced detection: {'ENABLED' if enhanced_detection else 'DISABLED'}")
        
        # Use optimized connector with massive cache
        self.db_connector = OptimizedDatabaseConnector(
            exclude_2025=True,
            enable_cache=True,
            cache_size_gb=CACHE_SIZE_GB
        )
        
        self.processor = FrameDataProcessor()
        
        # Initialize with enhanced parameters
        self.trend_analyzer = EMDTrendAnalyzer(
            ensemble_size=150 if enhanced_detection else 100,
            noise_strength=0.15 if enhanced_detection else 0.2
        )
        
        self.use_cache = use_cache
        self.parallel_frames = parallel_frames
        self.enhanced_detection = enhanced_detection
        
        self.weekly_data = None
        self.results_by_frame = {}
        
        # Performance tracking
        self.timing_stats = {}
        
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return CACHE_DIR / f"{hash_key}.pkl"
    
    def _load_from_cache(self, key: str) -> Optional[any]:
        """Load data from cache if available."""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"✓ Loaded from cache: {key}")
                return data
            except:
                pass
        return None
    
    def _save_to_cache(self, key: str, data: any) -> None:
        """Save data to cache."""
        if not self.use_cache:
            return
            
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved to cache: {key}")
        except Exception as e:
            print(f"Warning: Could not cache {key}: {e}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare frame proportion data with ultra-fast processing."""
        start_time = time.time()
        
        # Check cache first
        cache_key = "weekly_proportions_all_2024"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            self.weekly_data = cached_data
            self.timing_stats['data_loading'] = time.time() - start_time
            return cached_data
        
        print("\nLoading frame data from database...")
        print("Using parallel loading with 16 workers...")
        
        # Get frame data with parallel loading - get ALL available data
        df = self.db_connector.get_frame_data(
            start_date=None,  # No start date limit - get ALL historical data
            end_date="2024-12-31",     # Explicitly set end date to 2024
            show_progress=True,
            use_parallel=True,
            n_workers=16,  # Use more workers for M4 Max
            chunk_size=CHUNK_SIZE
        )
        
        if df.empty:
            print("WARNING: No data loaded. Checking database connection...")
            # Try a simpler query
            test_query = 'SELECT COUNT(*) as count FROM "CCF_processed_data" WHERE date <= \'2024-12-31\''
            test_result = self.db_connector.read_data(test_query)
            print(f"Total records in database (up to 2024): {test_result['count'].iloc[0]:,}")
            raise ValueError("No data could be loaded from database")
        
        print(f"Loaded {len(df):,} rows")
        
        # Convert date format efficiently using vectorized operations
        print("Converting date format...")
        df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y', errors='coerce')
        
        # Remove invalid dates
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"Warning: {invalid_dates:,} rows with invalid dates removed")
            df = df.dropna(subset=['date'])
        
        # Process data with optimized processor
        print("Processing frame data...")
        processed = self._process_frame_data_parallel(df)
        
        # Calculate weekly proportions in parallel
        print("Calculating weekly frame proportions...")
        weekly_props = self._calculate_weekly_proportions_optimized(processed)
        
        # Store for later use
        self.weekly_data = weekly_props
        
        # Save to cache
        self._save_to_cache(cache_key, weekly_props)
        
        # Save raw weekly data for reference
        weekly_props.to_csv(DATA_DIR / "raw_weekly_proportions.csv", index=False)
        print(f"Saved raw weekly data to {DATA_DIR / 'raw_weekly_proportions.csv'}")
        
        self.timing_stats['data_loading'] = time.time() - start_time
        print(f"Data loading completed in {self.timing_stats['data_loading']:.2f} seconds")
        
        return weekly_props
    
    def _process_frame_data_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process frame data in parallel chunks."""
        # Split dataframe into chunks for parallel processing
        n_chunks = N_CORES
        chunk_size = len(df) // n_chunks + 1
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=N_CORES) as executor:
            futures = [executor.submit(self.processor.process_frame_data, chunk) 
                      for chunk in chunks]
            
            processed_chunks = []
            with tqdm(total=len(futures), desc="Processing chunks") as pbar:
                for future in as_completed(futures):
                    processed_chunks.append(future.result())
                    pbar.update(1)
        
        return pd.concat(processed_chunks, ignore_index=True)
    
    def _calculate_weekly_proportions_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate frame proportions at weekly level with vectorized operations."""
        # Create week column
        df['week'] = df['date'].dt.to_period('W').dt.to_timestamp()
        
        # Vectorized aggregation for all frames at once
        weekly_data = []
        
        # Group by week
        grouped = df.groupby('week')
        
        # Pre-compute frame columns
        frame_cols = [f"{frame}_Detection" for frame in FRAME_NAMES]
        existing_cols = [col for col in frame_cols if col in df.columns]
        
        print(f"Processing {len(grouped)} weeks with vectorized operations...")
        
        for week, group in tqdm(grouped, desc="Processing weeks"):
            week_dict = {'week': week}
            
            # Vectorized computation of all frame counts
            frame_counts = group[existing_cols].sum()
            total_detections = frame_counts.sum()
            
            # Calculate proportions
            if total_detections > 0:
                proportions = frame_counts / total_detections
                for frame in FRAME_NAMES:
                    col = f"{frame}_Detection"
                    if col in existing_cols:
                        week_dict[f"{frame}_prop"] = proportions[col]
                        week_dict[f"{frame}_count"] = frame_counts[col]
                    else:
                        week_dict[f"{frame}_prop"] = 0
                        week_dict[f"{frame}_count"] = 0
            else:
                for frame in FRAME_NAMES:
                    week_dict[f"{frame}_prop"] = 0
                    week_dict[f"{frame}_count"] = 0
            
            # Add metadata for quality assessment
            week_dict['n_articles'] = group['doc_id'].nunique()
            week_dict['n_sentences'] = len(group)
            week_dict['n_media'] = group['media'].nunique() if 'media' in group.columns else 0
            week_dict['total_detections'] = total_detections
            
            # Data quality score - more adaptive based on year
            year = week.year
            
            # Adaptive thresholds based on period
            if year < 1990:
                # Very early period - much lower expectations
                article_threshold = 20  # Expect fewer articles
                media_threshold = 2     # Expect fewer media sources
            elif year < 2000:
                # Early period - lower expectations
                article_threshold = 50
                media_threshold = 3
            elif year < 2010:
                # Mid period - moderate expectations
                article_threshold = 75
                media_threshold = 4
            else:
                # Recent period - standard expectations
                article_threshold = 100
                media_threshold = 5
            
            quality_score = min(1.0, week_dict['n_articles'] / article_threshold) * 0.5
            quality_score += min(1.0, week_dict['n_media'] / media_threshold) * 0.5
            week_dict['data_quality'] = quality_score
            
            weekly_data.append(week_dict)
        
        return pd.DataFrame(weekly_data).sort_values('week')
    
    def _analyze_frame_emd(self, frame: str) -> Dict:
        """Analyze a single frame with EMD (for parallel processing)."""
        print(f"\nAnalyzing {frame} frame with adaptive EMD...")
        
        # Check cache - include enhanced flag in key
        cache_suffix = "_enhanced" if self.enhanced_detection else ""
        cache_key = f"emd_analysis_{frame}_2024{cache_suffix}"
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Get time series
        series = self.weekly_data.set_index('week')[f"{frame}_prop"]
        
        # Apply pre-processing if enhanced detection is enabled
        if self.enhanced_detection:
            # Fill missing values with interpolation
            series = series.interpolate(method='linear', limit_direction='both')
            
            # Apply adaptive smoothing for noise reduction
            if len(series) > 10:
                # More adaptive smoothing based on data period
                early_period = series.index[0].year < 2000
                
                if early_period:
                    # Stronger smoothing for early sparse periods
                    window_length = min(7, len(series) if len(series) % 2 == 1 else len(series) - 1)
                else:
                    window_length = min(9, len(series) if len(series) % 2 == 1 else len(series) - 1)
                    
                if window_length >= 3:
                    series_smooth = pd.Series(
                        savgol_filter(series.values, window_length, min(2, window_length-1), mode='nearest'),
                        index=series.index
                    )
                    # Blend original and smoothed based on data quality
                    quality_weight = self.weekly_data.set_index('week')['data_quality'].reindex(series.index)
                    
                    # More aggressive smoothing for low quality data
                    quality_weight = quality_weight.clip(lower=0.2)  # Ensure minimum weight
                    series = series * quality_weight + series_smooth * (1 - quality_weight)
        
        # Run complete EMD analysis
        analysis_results = self.trend_analyzer.analyze_complete(series)
        
        # Post-process if enhanced
        if self.enhanced_detection:
            analysis_results = self._enhance_trend_detection(analysis_results, frame)
        
        # Cache results
        self._save_to_cache(cache_key, analysis_results)
        
        # Print detailed summary including early periods
        trends = analysis_results['trends']
        sig_trends = [t for t in trends if t.significance_level in ['very_high', 'high', 'moderate']]
        
        # Count trends by period
        early_trends = [t for t in trends if t.start_date.year < 2000]
        recent_trends = [t for t in trends if t.start_date.year >= 2000]
        
        print(f"  {frame}: {len(trends)} trends detected ({len(sig_trends)} significant)")
        if early_trends:
            print(f"    - Pre-2000: {len(early_trends)} trends")
        if recent_trends:
            print(f"    - Post-2000: {len(recent_trends)} trends")
        
        return analysis_results
    
    def analyze_all_frames_emd(self):
        """Analyze trends for all frames using EMD method with parallel processing."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("ANALYZING TRENDS WITH ADAPTIVE EMD AND FUZZY CLASSIFICATION")
        print(f"Processing mode: {'PARALLEL' if self.parallel_frames else 'SEQUENTIAL'}")
        print(f"Enhanced detection: {'ENABLED' if self.enhanced_detection else 'DISABLED'}")
        print("Adaptive parameters for sparse early periods: ENABLED")
        print("="*80)
        
        if self.parallel_frames:
            # Process frames in parallel
            with ThreadPoolExecutor(max_workers=min(N_CORES, len(FRAME_NAMES))) as executor:
                futures = {executor.submit(self._analyze_frame_emd, frame): frame 
                          for frame in FRAME_NAMES}
                
                with tqdm(total=len(FRAME_NAMES), desc="Analyzing frames") as pbar:
                    for future in as_completed(futures):
                        frame = futures[future]
                        try:
                            result = future.result()
                            self.results_by_frame[frame] = result
                        except Exception as e:
                            print(f"Error analyzing {frame}: {e}")
                            self.results_by_frame[frame] = {'trends': [], 'critical_points': [], 
                                                           'components': {}, 'summary': {}}
                        pbar.update(1)
        else:
            # Sequential processing
            for frame in tqdm(FRAME_NAMES, desc="Processing frames"):
                self.results_by_frame[frame] = self._analyze_frame_emd(frame)
        
        self.timing_stats['emd_analysis'] = time.time() - start_time
        print(f"\nEMD analysis completed in {self.timing_stats['emd_analysis']:.2f} seconds")
        
        # Print detailed summary
        total_trends = sum(len(r['trends']) for r in self.results_by_frame.values())
        early_trends = sum(len([t for t in r['trends'] if t.start_date.year < 2000]) 
                          for r in self.results_by_frame.values())
        recent_trends = sum(len([t for t in r['trends'] if t.start_date.year >= 2000]) 
                           for r in self.results_by_frame.values())
        
        print(f"\nTotal trends detected across all frames: {total_trends}")
        print(f"  - Pre-2000 period: {early_trends} trends")
        print(f"  - Post-2000 period: {recent_trends} trends")
        
        # Print earliest detected trend
        all_trends = []
        for r in self.results_by_frame.values():
            all_trends.extend(r['trends'])
        if all_trends:
            earliest = min(all_trends, key=lambda t: t.start_date)
            print(f"  - Earliest trend detected: {earliest.start_date.strftime('%Y-%m-%d')}")
    
    def export_trend_data(self):
        """Export all trend data in multiple formats with parallel I/O."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("EXPORTING TREND DATA")
        print("="*80)
        
        # Use parallel I/O for exports
        export_tasks = [
            self._export_trends_csv,
            self._export_critical_points_csv,
            self._export_by_significance,
            self._export_fuzzy_memberships,
            self._export_imf_components,
            self._export_json_analysis,
            self._export_summary_statistics
        ]
        
        with ThreadPoolExecutor(max_workers=min(N_CORES, len(export_tasks))) as executor:
            futures = [executor.submit(task) for task in export_tasks]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Export error: {e}")
        
        self.timing_stats['export'] = time.time() - start_time
        print(f"\nData export completed in {self.timing_stats['export']:.2f} seconds")
    
    def _export_trends_csv(self):
        """Export all trends to comprehensive CSV."""
        all_trends = []
        
        for frame, results in self.results_by_frame.items():
            for trend in results['trends']:
                trend_dict = trend.to_dict()
                trend_dict['frame'] = frame
                all_trends.append(trend_dict)
        
        if all_trends:
            trends_df = pd.DataFrame(all_trends)
            # Add period column for easier filtering
            trends_df['period'] = trends_df['start_date'].apply(
                lambda x: 'pre-1990' if pd.Timestamp(x).year < 1990 else 
                         ('1990-2000' if pd.Timestamp(x).year < 2000 else 
                          ('2000-2010' if pd.Timestamp(x).year < 2010 else 'post-2010'))
            )
            trends_df = trends_df.sort_values(['frame', 'start_date'])
            
            output_path = DATA_DIR / "all_trends_emd.csv"
            trends_df.to_csv(output_path, index=False)
            print(f"  ✓ Exported {len(trends_df)} trends to {output_path.name}")
    
    def _export_critical_points_csv(self):
        """Export critical points for cascade analysis."""
        all_critical = []
        
        for frame, results in self.results_by_frame.items():
            for cp in results['critical_points']:
                cp_dict = cp.copy()
                cp_dict['frame'] = frame
                
                # Flatten scale memberships
                for scale in ['short', 'medium', 'long']:
                    cp_dict[f'membership_{scale}'] = cp_dict['scale_memberships'].get(scale, 0)
                del cp_dict['scale_memberships']
                
                all_critical.append(cp_dict)
        
        if all_critical:
            critical_df = pd.DataFrame(all_critical)
            critical_df = critical_df.sort_values('date')
            
            output_path = DATA_DIR / "critical_points_emd.csv"
            critical_df.to_csv(output_path, index=False)
            print(f"  ✓ Exported {len(critical_df)} critical points to {output_path.name}")
    
    def _export_by_significance(self):
        """Export trends grouped by significance level."""
        significance_levels = ['very_high', 'high', 'moderate', 'low']
        
        for sig_level in significance_levels:
            trends_at_level = []
            
            for frame, results in self.results_by_frame.items():
                for trend in results['trends']:
                    if trend.significance_level == sig_level:
                        trend_dict = trend.to_dict()
                        trend_dict['frame'] = frame
                        trends_at_level.append(trend_dict)
            
            if trends_at_level:
                df = pd.DataFrame(trends_at_level)
                output_path = DATA_DIR / f"trends_{sig_level}_significance.csv"
                df.to_csv(output_path, index=False)
                print(f"  ✓ Exported {len(df)} {sig_level} significance trends")
    
    def _export_fuzzy_memberships(self):
        """Export detailed fuzzy membership data."""
        membership_data = []
        
        for frame, results in self.results_by_frame.items():
            for trend in results['trends']:
                for scale, membership in trend.scale_memberships.items():
                    membership_data.append({
                        'frame': frame,
                        'start_date': trend.start_date,
                        'end_date': trend.end_date,
                        'scale': scale,
                        'membership': membership,
                        'is_primary': scale == trend.primary_scale,
                        'duration_weeks': trend.duration_weeks,
                        'direction': trend.direction,
                        'significance': trend.significance_level
                    })
        
        if membership_data:
            membership_df = pd.DataFrame(membership_data)
            output_path = DATA_DIR / "fuzzy_memberships.csv"
            membership_df.to_csv(output_path, index=False)
            print(f"  ✓ Exported fuzzy membership data to {output_path.name}")
    
    def _export_imf_components(self):
        """Export IMF components for each frame."""
        imf_dir = DATA_DIR / "imf_components"
        imf_dir.mkdir(exist_ok=True)
        
        for frame, results in self.results_by_frame.items():
            components = results.get('components', {})
            
            if components:
                comp_data = {}
                for comp_name, comp_series in components.items():
                    if isinstance(comp_series, pd.Series):
                        comp_data[comp_name] = comp_series.values
                
                if comp_data:
                    comp_df = pd.DataFrame(comp_data, index=self.weekly_data['week'])
                    output_path = imf_dir / f"{frame}_imf_components.csv"
                    comp_df.to_csv(output_path)
        
        print(f"  ✓ Exported IMF components to {imf_dir.name}/")
    
    def _export_json_analysis(self):
        """Export detailed analysis in JSON format."""
        detailed_analysis = {}
        
        for frame, results in self.results_by_frame.items():
            trends_data = []
            for trend in results['trends']:
                trend_dict = trend.to_dict()
                trend_dict['start_date'] = trend_dict['start_date'].strftime('%Y-%m-%d')
                trend_dict['end_date'] = trend_dict['end_date'].strftime('%Y-%m-%d')
                trends_data.append(trend_dict)
            
            critical_data = []
            for cp in results['critical_points']:
                cp_dict = cp.copy()
                cp_dict['date'] = cp_dict['date'].strftime('%Y-%m-%d')
                if 'scale_memberships' in cp_dict:
                    cp_dict['scale_memberships'] = dict(cp_dict['scale_memberships'])
                critical_data.append(cp_dict)
            
            detailed_analysis[frame] = {
                'n_trends': len(results['trends']),
                'n_critical_points': len(results['critical_points']),
                'trends': trends_data,
                'critical_points': critical_data,
                'summary': results.get('summary', {})
            }
        
        output_path = DATA_DIR / "detailed_analysis_emd.json"
        with open(output_path, 'w') as f:
            json.dump(detailed_analysis, f, indent=2, default=str)
        
        print(f"  ✓ Exported detailed analysis to {output_path.name}")
    
    def _export_summary_statistics(self):
        """Export summary statistics for validation."""
        summary_data = []
        
        for frame in FRAME_NAMES:
            results = self.results_by_frame.get(frame, {'trends': []})
            trends = results['trends']
            
            if trends:
                sig_stats = {}
                for level in ['very_high', 'high', 'moderate', 'low']:
                    level_trends = [t for t in trends if t.significance_level == level]
                    if level_trends:
                        sig_stats[level] = {
                            'count': len(level_trends),
                            'avg_duration_weeks': np.mean([t.duration_weeks for t in level_trends]),
                            'avg_confidence': np.mean([t.confidence for t in level_trends]),
                            'avg_relative_change': np.mean([abs(t.relative_change) for t in level_trends])
                        }
                
                scale_dist = {'short': 0, 'medium': 0, 'long': 0}
                for trend in trends:
                    scale_dist[trend.primary_scale] += 1
                
                summary_data.append({
                    'frame': frame,
                    'total_trends': len(trends),
                    'very_high_sig': sig_stats.get('very_high', {}).get('count', 0),
                    'high_sig': sig_stats.get('high', {}).get('count', 0),
                    'moderate_sig': sig_stats.get('moderate', {}).get('count', 0),
                    'low_sig': sig_stats.get('low', {}).get('count', 0),
                    'short_scale': scale_dist['short'],
                    'medium_scale': scale_dist['medium'],
                    'long_scale': scale_dist['long'],
                    'avg_confidence': np.mean([t.confidence for t in trends]) if trends else 0,
                    'avg_data_quality': np.mean([t.data_quality_score for t in trends]) if trends else 0,
                    'avg_duration_weeks': np.mean([t.duration_weeks for t in trends]) if trends else 0,
                    'max_relative_change': max([abs(t.relative_change) for t in trends]) if trends else 0
                })
            else:
                summary_data.append({
                    'frame': frame,
                    'total_trends': 0,
                    'very_high_sig': 0,
                    'high_sig': 0,
                    'moderate_sig': 0,
                    'low_sig': 0,
                    'short_scale': 0,
                    'medium_scale': 0,
                    'long_scale': 0,
                    'avg_confidence': 0,
                    'avg_data_quality': 0,
                    'avg_duration_weeks': 0,
                    'max_relative_change': 0
                })
        
        summary_df = pd.DataFrame(summary_data)
        output_path = DATA_DIR / "summary_statistics.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"  ✓ Exported summary statistics to {output_path.name}")
    
    def _enhance_trend_detection(self, results: Dict, frame: str) -> Dict:
        """Apply enhanced detection methods to improve trend detection."""
        trends = results['trends']
        
        # Additional validation using volume and media diversity
        if self.weekly_data is not None:
            frame_data = self.weekly_data.set_index('week')
            
            enhanced_trends = []
            for trend in trends:
                # Get data for trend period
                trend_period = (frame_data.index >= trend.start_date) & (frame_data.index <= trend.end_date)
                period_data = frame_data[trend_period]
                
                if len(period_data) > 0:
                    # Calculate additional metrics
                    avg_articles = period_data['n_articles'].mean()
                    avg_media = period_data['n_media'].mean()
                    
                    # Boost confidence for trends with high coverage
                    coverage_boost = min(0.1, avg_articles / 1000)  # Up to 0.1 boost
                    media_boost = min(0.05, avg_media / 50)  # Up to 0.05 boost
                    
                    trend.confidence = min(1.0, trend.confidence + coverage_boost + media_boost)
                    
                    # Adjust significance if confidence increased substantially
                    if trend.confidence > 0.7 and trend.significance_level == 'low':
                        trend.significance_level = 'moderate'
                    
                enhanced_trends.append(trend)
        
        results['trends'] = enhanced_trends
        return results
    
    def generate_validation_report(self):
        """Generate a text report for validation with performance metrics."""
        report_path = DATA_DIR / "trend_detection_report_optimized.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OPTIMIZED TREND DETECTION DATA PRODUCTION REPORT\n")
            f.write("EMD + FUZZY CLASSIFICATION METHOD\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System: Mac M4 Max with 128GB RAM\n")
            f.write(f"CPU Cores Used: {N_CORES}\n")
            f.write(f"Cache Size: {CACHE_SIZE_GB}GB\n")
            
            if self.weekly_data is not None:
                f.write(f"Data Period: {self.weekly_data['week'].min()} to {self.weekly_data['week'].max()}\n")
                f.write(f"Total Weeks: {len(self.weekly_data)}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*40 + "\n")
            for phase, duration in self.timing_stats.items():
                f.write(f"{phase:20s}: {duration:8.2f} seconds\n")
            
            total_time = sum(self.timing_stats.values())
            f.write(f"{'TOTAL TIME':20s}: {total_time:8.2f} seconds\n\n")
            
            f.write("METHODOLOGY\n")
            f.write("-"*40 + "\n")
            f.write("1. Optimized Data Loading:\n")
            f.write(f"   - Parallel loading with {N_CORES} workers\n")
            f.write(f"   - Chunk size: {CHUNK_SIZE:,} rows\n")
            f.write(f"   - In-memory caching: {CACHE_SIZE_GB}GB\n\n")
            
            f.write("2. Parallel Frame Processing:\n")
            f.write(f"   - {'Enabled' if self.parallel_frames else 'Disabled'}\n")
            f.write(f"   - Concurrent EMD analysis\n")
            f.write(f"   - Result caching enabled\n\n")
            
            f.write("OVERALL RESULTS\n")
            f.write("-"*40 + "\n")
            
            total_trends = sum(len(r['trends']) for r in self.results_by_frame.values())
            total_critical = sum(len(r['critical_points']) for r in self.results_by_frame.values())
            
            f.write(f"Total trends detected: {total_trends}\n")
            f.write(f"Total critical points: {total_critical}\n\n")
            
            # Frame-specific results
            f.write("FRAME-SPECIFIC RESULTS\n")
            f.write("-"*40 + "\n")
            
            for frame in FRAME_NAMES:
                results = self.results_by_frame.get(frame, {'trends': [], 'critical_points': []})
                trends = results['trends']
                
                f.write(f"\n{frame} Frame:\n")
                f.write("="*20 + "\n")
                f.write(f"  Trends: {len(trends)}\n")
                f.write(f"  Critical points: {len(results['critical_points'])}\n")
            
            # Cache statistics
            cache_stats = self.db_connector.get_cache_stats()
            f.write("\nCACHE STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Cache hits: {cache_stats.get('hits', 0)}\n")
            f.write(f"Cache misses: {cache_stats.get('misses', 0)}\n")
            f.write(f"Hit rate: {cache_stats.get('hit_rate', 'N/A')}\n")
            f.write(f"Cache size: {cache_stats.get('size_gb', 0):.2f}GB\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n  ✓ Generated validation report: {report_path}")
    
    def run_data_production(self):
        """Run complete data production pipeline with performance tracking."""
        total_start = time.time()
        
        print("="*80)
        print("OPTIMIZED TREND DETECTION DATA PRODUCTION")
        print("EMD + FUZZY CLASSIFICATION METHOD")
        print("="*80)
        
        try:
            # Load and prepare data
            self.load_and_prepare_data()
            
            # Analyze all frames with EMD
            self.analyze_all_frames_emd()
            
            # Export all data
            self.export_trend_data()
            
            # Generate validation report
            self.generate_validation_report()
            
        finally:
            # Close database connection
            self.db_connector.close()
        
        total_time = time.time() - total_start
        
        print("\n" + "="*80)
        print("DATA PRODUCTION COMPLETE")
        print("="*80)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"All data exported to: {DATA_DIR}")
        print(f"Ready for validation and visualization")
        
        # Print performance summary
        print("\nPerformance Summary:")
        print("-"*40)
        if self.weekly_data is not None:
            rows_processed = len(self.weekly_data) * len(FRAME_NAMES)
            print(f"Rows processed: {rows_processed:,}")
            print(f"Processing rate: {rows_processed/total_time:.0f} rows/second")
        
        for phase, duration in self.timing_stats.items():
            percentage = (duration / total_time) * 100
            print(f"{phase:20s}: {duration:6.2f}s ({percentage:5.1f}%)")


def main():
    """Main execution function with optimization options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Trend Detection Data Production')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--sequential', action='store_true', help='Disable parallel frame processing')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before running')
    parser.add_argument('--no-enhance', action='store_true', help='Disable enhanced detection methods')
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        print("Clearing cache...")
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("Cache cleared.")
    
    # Initialize and run producer
    producer = OptimizedTrendDataProducer(
        use_cache=not args.no_cache,
        parallel_frames=not args.sequential,
        enhanced_detection=not args.no_enhance
    )
    
    try:
        producer.run_data_production()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        print("Cleaning up...")
        producer.db_connector.close()
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        producer.db_connector.close()
        sys.exit(1)


if __name__ == "__main__":
    main()