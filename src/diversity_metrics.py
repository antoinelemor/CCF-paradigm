"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
diversity_metrics.py

MAIN OBJECTIVE:
---------------
This script provides diversity metrics calculations for frame analysis, with
an objective threshold for multi-frame classification based on information theory.

Dependencies:
-------------
- numpy
- pandas
- scipy

MAIN FEATURES:
--------------
1) Shannon entropy calculation with normalization
2) Effective number of frames (Hill numbers)
3) Multi-frame paradigm detection using objective threshold
4) Time series diversity analysis
5) Statistical tests for paradigm classification

Author:
-------
Antoine Lemor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Define objective threshold
MULTIFRAME_THRESHOLD = np.log(2) / np.log(8)  # ≈ 0.333


class DiversityMetrics:
    """Calculate diversity metrics for frame analysis."""
    
    def __init__(self, frame_names: Optional[List[str]] = None):
        """
        Initialize diversity metrics calculator.
        
        Args:
            frame_names: List of frame names to analyze
        """
        self.frame_names = frame_names or ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]
        self.n_frames = len(self.frame_names)
        self.multiframe_threshold = MULTIFRAME_THRESHOLD
    
    def shannon_entropy(self, proportions: np.ndarray, normalize: bool = True) -> float:
        """
        Calculate Shannon entropy for a set of proportions.
        
        Args:
            proportions: Array of frame proportions
            normalize: Whether to normalize by maximum entropy
            
        Returns:
            Shannon entropy value
        """
        proportions = np.asarray(proportions)
        
        # Remove zero values and normalize
        proportions = proportions[proportions > 0]
        
        if len(proportions) == 0:
            return 0.0
            
        proportions = proportions / proportions.sum()
        
        # Calculate Shannon entropy
        entropy = -np.sum(proportions * np.log(proportions))
        
        # Normalize if requested
        if normalize and self.n_frames > 1:
            max_entropy = np.log(self.n_frames)
            entropy = entropy / max_entropy
        
        return entropy
    
    def effective_frames(self, proportions: np.ndarray) -> float:
        """
        Calculate effective number of frames (exp(Shannon entropy)).
        
        Args:
            proportions: Array of frame proportions
            
        Returns:
            Effective number of frames
        """
        proportions = np.asarray(proportions)
        
        # Filter positive values and normalize
        proportions = proportions[proportions > 0]
        
        if len(proportions) == 0:
            return 0.0
            
        proportions = proportions / proportions.sum()
        
        # Calculate Shannon entropy (not normalized)
        entropy = -np.sum(proportions * np.log(proportions))
        
        # Return exponential (effective number)
        return np.exp(entropy)
    
    def is_multiframe(self, entropy_value: float) -> bool:
        """
        Determine if a single entropy value indicates multi-frame.
        
        Args:
            entropy_value: Normalized Shannon entropy
            
        Returns:
            True if multi-frame, False otherwise
        """
        return entropy_value > self.multiframe_threshold
    
    def calculate_time_series_diversity(self, df: pd.DataFrame, 
                                      metric: str = 'shannon') -> pd.DataFrame:
        """
        Calculate diversity metrics over time.
        
        Args:
            df: DataFrame with time period and frame columns
            metric: Diversity metric to use ('shannon', 'effective')
            
        Returns:
            DataFrame with time and diversity values
        """
        # Identify time column
        time_col = None
        for col in ['week', 'period', 'year_month', 'date']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            raise ValueError("No time column found in DataFrame")
        
        # Get frame columns
        frame_cols = [col for col in self.frame_names if col in df.columns]
        
        if not frame_cols:
            raise ValueError("No frame columns found in DataFrame")
        
        # Calculate diversity for each time period
        results = []
        
        for idx, row in df.iterrows():
            # Get proportions
            proportions = row[frame_cols].values
            
            # Convert to numeric array
            try:
                proportions = np.asarray(proportions, dtype=np.float64)
            except (ValueError, TypeError):
                proportions = np.array([float(x) if pd.notna(x) else 0.0 for x in proportions])
            
            # Skip if all values are NaN or zero
            if np.all(np.isnan(proportions)) or np.all(proportions == 0):
                continue
            
            # Replace NaN with 0
            proportions = np.nan_to_num(proportions, nan=0.0)
            
            if metric == 'shannon':
                diversity = self.shannon_entropy(proportions)
            elif metric == 'effective':
                diversity = self.effective_frames(proportions)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            results.append({
                time_col: row[time_col],
                'diversity': diversity,
                'metric': metric
            })
        
        return pd.DataFrame(results)
    
    def test_multiframe_paradigm(self, entropy_series: pd.Series) -> Dict[str, Union[bool, float, str]]:
        """
        Test whether the paradigm is multi-frame based on median entropy.
        
        Args:
            entropy_series: Series of Shannon entropy values
            
        Returns:
            Dictionary with test results
        """
        # Calculate statistics
        median_entropy = entropy_series.median()
        mean_entropy = entropy_series.mean()
        std_entropy = entropy_series.std()
        
        # Determine multi-frame status
        is_multiframe = median_entropy > self.multiframe_threshold
        
        # Calculate percentage of time above threshold
        pct_above_threshold = (entropy_series > self.multiframe_threshold).mean() * 100
        
        # Calculate effective frames at median
        effective_frames_median = np.exp(median_entropy * np.log(self.n_frames))
        
        # Perform statistical test
        # H0: median <= threshold, H1: median > threshold
        differences = entropy_series - self.multiframe_threshold
        
        # Handle case where all differences might be the same sign
        if np.all(differences > 0) or np.all(differences < 0):
            # Use sign test
            n_positive = np.sum(differences > 0)
            n_total = len(differences)
            p_value = 1 - stats.binom.cdf(n_positive - 1, n_total, 0.5)
            test_name = "Sign test"
        else:
            # Use Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(differences, alternative='greater')
            test_name = "Wilcoxon signed-rank test"
        
        # Bootstrap confidence interval for median
        n_bootstrap = 1000
        bootstrap_medians = []
        for _ in range(n_bootstrap):
            sample = entropy_series.sample(n=len(entropy_series), replace=True)
            bootstrap_medians.append(sample.median())
        
        ci_lower = np.percentile(bootstrap_medians, 2.5)
        ci_upper = np.percentile(bootstrap_medians, 97.5)
        
        # Classification
        if is_multiframe:
            if median_entropy > 0.5:
                classification = "Established multi-frame paradigm"
            else:
                classification = "Emerging multi-frame paradigm"
        else:
            classification = "Mono-frame dominant paradigm"
        
        results = {
            'is_multiframe': is_multiframe,
            'median_entropy': median_entropy,
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy,
            'threshold': self.multiframe_threshold,
            'pct_above_threshold': pct_above_threshold,
            'effective_frames_median': effective_frames_median,
            'test_name': test_name,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'paradigm_classification': classification
        }
        
        return results
    
    def calculate_yearly_multiframe_status(self, df: pd.DataFrame, 
                                         time_col: str = 'week',
                                         entropy_col: str = 'shannon_entropy') -> pd.DataFrame:
        """
        Calculate multi-frame status for each year.
        
        Args:
            df: DataFrame with time and entropy columns
            time_col: Name of time column
            entropy_col: Name of entropy column
            
        Returns:
            DataFrame with yearly status
        """
        # Extract year
        df['year'] = pd.to_datetime(df[time_col]).dt.year
        
        yearly_results = []
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            
            # Test for multi-frame
            test_results = self.test_multiframe_paradigm(year_data[entropy_col])
            
            yearly_results.append({
                'year': year,
                'n_weeks': len(year_data),
                'median_entropy': test_results['median_entropy'],
                'is_multiframe': test_results['is_multiframe'],
                'pct_weeks_multiframe': test_results['pct_above_threshold'],
                'effective_frames': test_results['effective_frames_median'],
                'classification': test_results['paradigm_classification']
            })
        
        return pd.DataFrame(yearly_results)


def analyze_paradigm_diversity(df: pd.DataFrame, 
                             frame_names: Optional[List[str]] = None) -> Dict:
    """
    Convenience function to analyze paradigm diversity.
    
    Args:
        df: DataFrame with frame data
        frame_names: List of frame names
        
    Returns:
        Dictionary with analysis results
    """
    metrics = DiversityMetrics(frame_names)
    
    # Calculate Shannon entropy
    entropy_df = metrics.calculate_time_series_diversity(df, metric='shannon')
    
    # Test for multi-frame paradigm
    return metrics.test_multiframe_paradigm(entropy_df['diversity'])