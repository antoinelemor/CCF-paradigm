"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
01_multiframe_paradigm_analysis_fast.py

MAIN OBJECTIVE:
---------------
Ultra-fast version using server-side cursors and streaming for MacBook M4 Max.
Avoids the OFFSET/LIMIT bottleneck by using proper database streaming.

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from scipy import stats
import time
import logging
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor
import gc
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import custom modules
from src.diversity_metrics import DiversityMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MULTIFRAME_THRESHOLD = np.log(2) / np.log(8)  # ≈ 0.333
DATA_DIR = project_root / "data" / "01_paradigm_composition"
RESULTS_DIR = project_root / "results" / "01_paradigm_composition"
FRAME_CODES = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class FastMultiFrameAnalyzer:
    """Ultra-fast analyzer using database streaming."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.diversity_metrics = DiversityMetrics()
        self.db_params = {
            "host": "localhost",
            "port": 5432,
            "dbname": "CCF",
            "user": "antoine",
            "password": ""
        }
        self.results = {}
    
    @contextmanager
    def get_streaming_cursor(self, name="stream_cursor"):
        """Create a server-side cursor for streaming large datasets."""
        conn = None
        cursor = None
        try:
            # Connect with specific options for streaming
            conn = psycopg2.connect(
                **self.db_params,
                cursor_factory=RealDictCursor
            )
            conn.autocommit = False
            
            # Create server-side cursor
            cursor = conn.cursor(name=name)
            # Set fetch size for optimal performance
            cursor.itersize = 10000  # Fetch 10k rows at a time
            
            yield cursor
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def load_and_aggregate_data(self, start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load and aggregate data directly in the database.
        This is MUCH faster than loading raw data and processing in Python.
        """
        logger.info("Loading and aggregating data directly in database...")
        start_time = time.time()
        
        # Build optimized aggregation query
        # This does all the heavy lifting in PostgreSQL which is much faster
        query = """
        WITH weekly_aggregates AS (
            SELECT 
                DATE_TRUNC('week', TO_DATE(date, 'MM-DD-YYYY')) as week,
                AVG("Cult_Detection") as cult_avg,
                AVG("Eco_Detection") as eco_avg,
                AVG("Envt_Detection") as envt_avg,
                AVG("Pbh_Detection") as pbh_avg,
                AVG("Just_Detection") as just_avg,
                AVG("Pol_Detection") as pol_avg,
                AVG("Sci_Detection") as sci_avg,
                AVG("Secu_Detection") as secu_avg,
                COUNT(DISTINCT doc_id) as doc_count,
                COUNT(*) as sentence_count
            FROM "CCF_processed_data"
            WHERE sentences IS NOT NULL 
                AND sentences != ''
                AND TO_DATE(date, 'MM-DD-YYYY') <= '2024-12-31'::date
        """
        
        if start_date:
            query += f" AND TO_DATE(date, 'MM-DD-YYYY') >= '{start_date}'::date"
        
        query += """
            GROUP BY DATE_TRUNC('week', TO_DATE(date, 'MM-DD-YYYY'))
        )
        SELECT 
            week,
            cult_avg, eco_avg, envt_avg, pbh_avg, 
            just_avg, pol_avg, sci_avg, secu_avg,
            doc_count, sentence_count
        FROM weekly_aggregates
        ORDER BY week
        """
        
        # Execute query with streaming
        weekly_data = []
        
        with self.get_streaming_cursor("aggregate_cursor") as cursor:
            logger.info("Executing aggregation query...")
            cursor.execute(query)
            
            logger.info("Streaming results...")
            # Stream results in batches
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                weekly_data.extend(rows)
        
        # Convert to DataFrame
        df = pd.DataFrame(weekly_data)
        
        # Convert week to datetime (handle timezone-aware timestamps)
        df['week'] = pd.to_datetime(df['week'], utc=True).dt.tz_localize(None)
        
        # Convert all numeric columns from Decimal to float
        numeric_cols = df.columns.difference(['week'])
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        load_time = time.time() - start_time
        logger.info(f"Data loaded and aggregated in {load_time:.2f} seconds")
        logger.info(f"Loaded {len(df)} weekly aggregates")
        
        return df
    
    def calculate_metrics_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics using vectorized operations.
        """
        logger.info("Calculating metrics...")
        start_time = time.time()
        
        # Frame columns
        frame_cols = ['cult_avg', 'eco_avg', 'envt_avg', 'pbh_avg', 
                     'just_avg', 'pol_avg', 'sci_avg', 'secu_avg']
        
        # Convert to numpy array for faster computation
        frame_values = df[frame_cols].values
        
        # Normalize rows to get proportions (vectorized)
        row_sums = frame_values.sum(axis=1, keepdims=True)
        proportions = np.divide(frame_values, row_sums, 
                               out=np.zeros_like(frame_values), 
                               where=row_sums != 0)
        
        # Calculate Shannon entropy for each week (vectorized)
        # H = -sum(p * log(p)) / log(8)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_proportions = np.log(proportions)
            log_proportions[~np.isfinite(log_proportions)] = 0
        
        entropy_values = -np.sum(proportions * log_proportions, axis=1) / np.log(8)
        
        # Create result DataFrame
        result = df[['week', 'doc_count', 'sentence_count']].copy()
        
        # Add frame proportions
        for i, frame in enumerate(FRAME_CODES):
            result[frame] = proportions[:, i]
        
        # Add calculated metrics
        result['shannon_entropy'] = entropy_values
        result['effective_frames'] = np.exp(entropy_values * np.log(8))
        result['is_multiframe'] = entropy_values > MULTIFRAME_THRESHOLD
        result['year'] = df['week'].dt.year
        
        calc_time = time.time() - start_time
        logger.info(f"Metrics calculated in {calc_time:.2f} seconds")
        
        return result
    
    def calculate_statistics_fast(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics with optimized algorithms.
        """
        logger.info("Calculating statistics...")
        
        # Use numpy arrays for speed
        entropy_values = df['shannon_entropy'].values
        
        # Overall statistics
        overall_stats = {
            'n_weeks': len(df),
            'median_entropy': float(np.median(entropy_values)),
            'mean_entropy': float(np.mean(entropy_values)),
            'std_entropy': float(np.std(entropy_values)),
            'min_entropy': float(np.min(entropy_values)),
            'max_entropy': float(np.max(entropy_values)),
            'pct_multiframe': float(np.mean(df['is_multiframe'].values) * 100),
            'median_effective_frames': float(np.median(df['effective_frames'].values)),
            'is_multiframe_overall': bool(np.median(entropy_values) > MULTIFRAME_THRESHOLD)
        }
        
        # Statistical test
        differences = entropy_values - MULTIFRAME_THRESHOLD
        
        if np.all(differences > 0) or np.all(differences < 0):
            # Use sign test
            n_positive = np.sum(differences > 0)
            n_total = len(differences)
            p_value = 1 - stats.binom.cdf(n_positive - 1, n_total, 0.5)
        else:
            # Use Wilcoxon test
            statistic, p_value = stats.wilcoxon(differences, alternative='greater')
        
        overall_stats['p_value'] = float(p_value)
        overall_stats['significant'] = bool(p_value < 0.05)
        
        # Fast bootstrap for confidence intervals
        n_bootstrap = 1000
        np.random.seed(42)
        bootstrap_indices = np.random.randint(0, len(entropy_values), 
                                             size=(n_bootstrap, len(entropy_values)))
        bootstrap_medians = np.median(entropy_values[bootstrap_indices], axis=1)
        
        overall_stats['ci_lower'] = float(np.percentile(bootstrap_medians, 2.5))
        overall_stats['ci_upper'] = float(np.percentile(bootstrap_medians, 97.5))
        
        return overall_stats
    
    def identify_paradigm_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify paradigm periods using vectorized operations.
        """
        logger.info("Identifying paradigm periods...")
        
        # Create smoothed version
        window = 12  # 3 months
        df = df.copy()
        df['shannon_smooth'] = df['shannon_entropy'].rolling(
            window=window, center=True, min_periods=window//2
        ).median()
        
        # Determine paradigm status
        df['paradigm_smooth'] = df['shannon_smooth'] > MULTIFRAME_THRESHOLD
        
        # Identify period changes
        df['period_change'] = df['paradigm_smooth'].ne(df['paradigm_smooth'].shift())
        df['period_id'] = df['period_change'].cumsum()
        
        # Aggregate periods
        periods = []
        for period_id, group in df.groupby('period_id'):
            if len(group) < 4:
                continue
            
            period_info = {
                'period_id': int(period_id),
                'start_date': group['week'].min(),
                'end_date': group['week'].max(),
                'n_weeks': len(group),
                'is_multiframe': bool(group['paradigm_smooth'].iloc[0]),
                'median_entropy': float(group['shannon_entropy'].median()),
                'mean_effective_frames': float(group['effective_frames'].mean())
            }
            periods.append(period_info)
        
        return pd.DataFrame(periods)
    
    def save_results(self, weekly_df: pd.DataFrame, periods_df: pd.DataFrame, stats: Dict):
        """Save results efficiently."""
        logger.info("Saving results...")
        
        # Save weekly metrics
        columns_to_save = ['week', 'shannon_entropy', 'effective_frames', 
                          'is_multiframe', 'year'] + FRAME_CODES
        
        # Try Parquet format if available (much faster and smaller)
        try:
            weekly_path_parquet = DATA_DIR / "weekly_metrics.parquet"
            weekly_df[columns_to_save].to_parquet(weekly_path_parquet, 
                                                  compression='snappy', 
                                                  index=False)
            logger.info(f"Saved to Parquet format: {weekly_path_parquet}")
        except ImportError:
            logger.info("Parquet not available, using CSV format")
        
        # Always save as CSV for compatibility
        weekly_path_csv = DATA_DIR / "weekly_metrics.csv"
        weekly_df[columns_to_save].to_csv(weekly_path_csv, index=False)
        
        # Save periods
        periods_path = DATA_DIR / "paradigm_periods.csv"
        periods_df.to_csv(periods_path, index=False)
        
        # Save statistics
        stats_path = DATA_DIR / "summary_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Results saved to {DATA_DIR}")
    
    def generate_report(self, weekly_df: pd.DataFrame, stats: Dict, periods_df: pd.DataFrame):
        """Generate text report."""
        logger.info("Generating report...")
        
        report_path = RESULTS_DIR / "paradigm_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-FRAME PARADIGM ANALYSIS (FAST VERSION)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Period: {weekly_df['week'].min()} to {weekly_df['week'].max()}\n")
            f.write(f"Total Weeks: {stats['n_weeks']}\n")
            f.write(f"Multi-frame Threshold: {MULTIFRAME_THRESHOLD:.4f}\n\n")
            
            f.write("OVERALL RESULTS\n")
            f.write("-"*40 + "\n")
            paradigm_type = "MULTI-FRAME" if stats['is_multiframe_overall'] else "MONO-FRAME DOMINANT"
            f.write(f"Paradigm Classification: {paradigm_type}\n")
            f.write(f"Median Shannon Entropy: {stats['median_entropy']:.4f}\n")
            f.write(f"95% CI for Median: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")
            f.write(f"Median Effective Frames: {stats['median_effective_frames']:.2f}\n")
            f.write(f"Percentage Multi-frame Weeks: {stats['pct_multiframe']:.1f}%\n\n")
            
            f.write("STATISTICAL TEST\n")
            f.write("-"*40 + "\n")
            f.write(f"H0: Median entropy ≤ {MULTIFRAME_THRESHOLD:.4f}\n")
            f.write(f"H1: Median entropy > {MULTIFRAME_THRESHOLD:.4f}\n")
            f.write(f"p-value: {stats['p_value']:.6f}\n")
            f.write(f"Result: {'Reject H0 (Multi-frame)' if stats['significant'] else 'Fail to reject H0'}\n\n")
            
            # Paradigm periods
            if not periods_df.empty:
                f.write("PARADIGM PERIODS\n")
                f.write("-"*40 + "\n")
                for _, period in periods_df.iterrows():
                    period_type = "Multi-frame" if period['is_multiframe'] else "Mono-frame"
                    f.write(f"\nPeriod {period['period_id']}: {period_type}\n")
                    f.write(f"  Duration: {period['start_date'].strftime('%Y-%m-%d')} to ")
                    f.write(f"{period['end_date'].strftime('%Y-%m-%d')} ({period['n_weeks']} weeks)\n")
                    f.write(f"  Median entropy: {period['median_entropy']:.3f}\n")
                    f.write(f"  Mean effective frames: {period['mean_effective_frames']:.2f}\n")
            
            # Yearly summary
            f.write("\n\nYEARLY SUMMARY\n")
            f.write("-"*40 + "\n")
            yearly_summary = weekly_df.groupby('year').agg({
                'shannon_entropy': ['median', 'mean', 'std'],
                'is_multiframe': 'mean'
            })
            
            f.write(f"{'Year':<6} {'Median':<8} {'Mean':<8} {'Std':<8} {'% Multi':<8}\n")
            f.write("-"*40 + "\n")
            
            for year in yearly_summary.index:
                median = yearly_summary.loc[year, ('shannon_entropy', 'median')]
                mean = yearly_summary.loc[year, ('shannon_entropy', 'mean')]
                std = yearly_summary.loc[year, ('shannon_entropy', 'std')]
                pct_multi = yearly_summary.loc[year, ('is_multiframe', 'mean')] * 100
                
                f.write(f"{year:<6} {median:<8.3f} {mean:<8.3f} {std:<8.3f} {pct_multi:<8.1f}\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def run_analysis(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run fast analysis pipeline.
        """
        total_start = time.time()
        
        logger.info("="*80)
        logger.info("MULTI-FRAME PARADIGM ANALYSIS (FAST VERSION)")
        logger.info("="*80)
        
        # Performance tracking
        timings = {}
        
        # Step 1: Load and aggregate data directly in database
        step_start = time.time()
        weekly_df = self.load_and_aggregate_data(start_date, end_date)
        timings['data_loading'] = time.time() - step_start
        
        # Step 2: Calculate metrics
        step_start = time.time()
        weekly_df = self.calculate_metrics_vectorized(weekly_df)
        timings['metrics_calculation'] = time.time() - step_start
        
        # Step 3: Calculate statistics
        step_start = time.time()
        stats = self.calculate_statistics_fast(weekly_df)
        timings['statistics'] = time.time() - step_start
        
        # Step 4: Identify paradigm periods
        step_start = time.time()
        periods_df = self.identify_paradigm_periods(weekly_df)
        timings['period_identification'] = time.time() - step_start
        
        # Step 5: Generate report
        step_start = time.time()
        self.generate_report(weekly_df, stats, periods_df)
        timings['report_generation'] = time.time() - step_start
        
        # Step 6: Save results
        step_start = time.time()
        self.save_results(weekly_df, periods_df, stats)
        timings['data_saving'] = time.time() - step_start
        
        # Calculate total time
        total_time = time.time() - total_start
        timings['total'] = total_time
        
        # Performance report
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE REPORT")
        logger.info("="*80)
        for step, duration in timings.items():
            if step != 'total':
                percentage = (duration / total_time) * 100
                logger.info(f"{step:25s}: {duration:6.2f}s ({percentage:5.1f}%)")
        logger.info("-"*40)
        logger.info(f"{'Total time':25s}: {total_time:6.2f}s")
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Paradigm: {'MULTI-FRAME' if stats['is_multiframe_overall'] else 'MONO-FRAME DOMINANT'}")
        logger.info(f"Median Entropy: {stats['median_entropy']:.4f}")
        
        return weekly_df, stats


def main():
    """Main execution function."""
    analyzer = FastMultiFrameAnalyzer()
    weekly_df, results = analyzer.run_analysis()


if __name__ == "__main__":
    main()