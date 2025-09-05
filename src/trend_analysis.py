"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
trend_analysis.py

MAIN OBJECTIVE:
---------------
Advanced multi-resolution trend analysis module using EMD (Empirical Mode Decomposition)
and fuzzy classification for adaptive trend detection without arbitrary thresholds.
Designed to identify trend periods and critical transition points for event impact analysis.

Dependencies:
-------------
- numpy
- pandas
- scipy
- sklearn
- statsmodels
- PyEMD
- skfuzzy

MAIN FEATURES:
--------------
1) EMD-based adaptive decomposition without fixed temporal thresholds
2) Fuzzy classification for trend scale assignment
3) Dynamic significance level validation
4) Robust statistical testing with data volume adjustment
5) Critical point identification with multi-level significance
6) Adaptive smoothing for early period variability

Author:
-------
Antoine Lemor
"""

import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.multitest import multipletests
from PyEMD import EMD, EEMD, CEEMDAN
import skfuzzy as fuzz
import ruptures as rpt
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FuzzyTrend:
    """Data class for trend with fuzzy scale membership."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_idx: int
    end_idx: int
    direction: str  # 'increasing' or 'decreasing'
    primary_scale: str  # Primary scale based on highest membership
    scale_memberships: Dict[str, float]  # Fuzzy memberships for each scale
    duration_weeks: int
    slope: float
    relative_change: float
    absolute_change: float
    raw_p_value: float
    corrected_p_value: float
    significance_level: str  # 'very_high', 'high', 'moderate', 'low'
    strength: str  # 'strong', 'moderate', 'weak'
    method: str
    confidence: float
    data_quality_score: float  # Quality score based on data availability
    imf_index: Optional[int] = None  # Which IMF this trend comes from
    
    @property
    def duration_years(self) -> float:
        return self.duration_weeks / 52.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'direction': self.direction,
            'primary_scale': self.primary_scale,
            'scale_membership_short': self.scale_memberships.get('short', 0),
            'scale_membership_medium': self.scale_memberships.get('medium', 0),
            'scale_membership_long': self.scale_memberships.get('long', 0),
            'duration_weeks': self.duration_weeks,
            'duration_years': self.duration_years,
            'slope': self.slope,
            'relative_change': self.relative_change,
            'absolute_change': self.absolute_change,
            'raw_p_value': self.raw_p_value,
            'corrected_p_value': self.corrected_p_value,
            'significance_level': self.significance_level,
            'strength': self.strength,
            'method': self.method,
            'confidence': self.confidence,
            'data_quality_score': self.data_quality_score,
            'imf_index': self.imf_index
        }


class EMDTrendAnalyzer:
    """Advanced trend analyzer using EMD and fuzzy classification."""
    
    def __init__(self, ensemble_size: int = 100, noise_strength: float = 0.2):
        """
        Initialize EMD analyzer.
        
        Args:
            ensemble_size: Number of ensemble members for EEMD
            noise_strength: Noise amplitude for EEMD
        """
        self.ensemble_size = ensemble_size
        self.noise_strength = noise_strength
        
        # Fuzzy membership function parameters (will be adapted dynamically)
        self.fuzzy_params = {
            'short': {'center': 8, 'width': 4},     # ~2 months center
            'medium': {'center': 26, 'width': 13},  # ~6 months center
            'long': {'center': 78, 'width': 26}     # ~1.5 years center
        }
        
        # Significance levels
        self.significance_levels = {
            'very_high': 0.001,
            'high': 0.01,
            'moderate': 0.05,
            'low': 0.10
        }
        
        # Enhanced detection parameters - adaptive based on data density
        self.base_min_trend_weeks = 4  # Base minimum weeks for a trend
        self.overlap_threshold = 0.5  # More aggressive merging
        self.quality_weight_power = 0.3  # Even less aggressive quality weighting for early periods
        
        # Adaptive parameters for different data density periods
        self.density_thresholds = {
            'very_sparse': 0.3,  # < 30% data coverage
            'sparse': 0.5,       # 30-50% data coverage  
            'moderate': 0.7,     # 50-70% data coverage
            'dense': 0.85        # > 85% data coverage
        }
        
        # Adaptive confidence thresholds
        self.adaptive_confidence = {
            'very_sparse': 0.35,  # Lower threshold for very sparse data
            'sparse': 0.40,       # Lower threshold for sparse data
            'moderate': 0.45,     # Standard threshold
            'dense': 0.50         # Higher threshold for dense data  
        }
        
        # Keep compatibility attribute
        self.min_trend_weeks = self.base_min_trend_weeks
        
    def decompose_with_emd(self, series: pd.Series, method: str = 'ceemdan') -> Dict[str, pd.Series]:
        """
        Decompose time series using EMD variants.
        
        Args:
            series: Time series data
            method: 'emd', 'eemd', or 'ceemdan'
            
        Returns:
            Dictionary with IMFs and residual
        """
        components = {}
        data = series.values
        
        # Handle data quality issues in early periods
        data_quality = self._assess_data_quality(series)
        
        try:
            if method == 'ceemdan':
                # CEEMDAN for most robust decomposition
                ceemdan = CEEMDAN(ensemble_size=self.ensemble_size, 
                                 noise_strength=self.noise_strength)
                IMFs = ceemdan(data)
            elif method == 'eemd':
                # Ensemble EMD
                eemd = EEMD(ensemble_size=self.ensemble_size,
                           noise_strength=self.noise_strength)
                IMFs = eemd(data)
            else:
                # Standard EMD
                emd = EMD()
                IMFs = emd(data)
            
            # Store IMFs
            for i, imf in enumerate(IMFs):
                # Apply adaptive smoothing based on data quality
                if data_quality['early_period_sparse'] and i < 2:
                    # Apply stronger smoothing to high-frequency IMFs in sparse periods
                    imf = gaussian_filter1d(imf, sigma=2, mode='nearest')
                
                components[f'IMF_{i}'] = pd.Series(imf, index=series.index)
            
            # Also compute traditional decomposition for comparison
            if len(series) >= 104:
                try:
                    stl = STL(series, seasonal=53, trend=105)
                    result = stl.fit()
                    components['stl_trend'] = result.trend
                    components['stl_seasonal'] = result.seasonal
                except:
                    pass
            
            # Add quality-adjusted raw series
            components['raw'] = series
            components['quality_scores'] = pd.Series(data_quality['point_quality'], 
                                                    index=series.index)
            
        except Exception as e:
            print(f"EMD decomposition failed: {e}")
            # Fallback to simple smoothing
            components['smoothed'] = pd.Series(
                gaussian_filter1d(data, sigma=4, mode='nearest'),
                index=series.index
            )
            components['raw'] = series
        
        return components
    
    def get_adaptive_min_trend_weeks(self, data_density: float) -> int:
        """Get adaptive minimum trend duration based on data density."""
        if data_density < self.density_thresholds['very_sparse']:
            return max(2, self.base_min_trend_weeks - 2)  # Allow 2-week trends for very sparse
        elif data_density < self.density_thresholds['sparse']:
            return max(3, self.base_min_trend_weeks - 1)  # Allow 3-week trends for sparse
        elif data_density < self.density_thresholds['moderate']:
            return self.base_min_trend_weeks  # Standard 4 weeks
        else:
            return self.base_min_trend_weeks + 1  # Require 5 weeks for dense data
    
    def get_data_density_category(self, density: float) -> str:
        """Categorize data density."""
        if density < self.density_thresholds['very_sparse']:
            return 'very_sparse'
        elif density < self.density_thresholds['sparse']:
            return 'sparse'
        elif density < self.density_thresholds['moderate']:
            return 'moderate'
        else:
            return 'dense'
    
    def _assess_data_quality(self, series: pd.Series) -> Dict[str, Any]:
        """
        Assess data quality and density over time.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with quality metrics
        """
        # Dynamic assessment of data density across the entire time series
        # No hard threshold at year 2000 - assess density continuously
        total_possible_weeks = (series.index[-1] - series.index[0]).days // 7 + 1
        actual_weeks = len(series)
        overall_density = actual_weeks / total_possible_weeks if total_possible_weeks > 0 else 1.0
        
        # Assess density in different periods dynamically
        # Split series into quarters for assessment
        quarter_size = len(series) // 4
        densities = []
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else len(series)
            quarter_data = series.iloc[start_idx:end_idx]
            if len(quarter_data) > 0:
                quarter_start = quarter_data.index[0]
                quarter_end = quarter_data.index[-1]
                expected_weeks = (quarter_end - quarter_start).days // 7 + 1
                quarter_density = len(quarter_data) / expected_weeks if expected_weeks > 0 else 1.0
                densities.append(quarter_density)
        
        # Identify if early periods are sparse (first quarter)
        early_sparse = densities[0] < self.density_thresholds['sparse'] if densities else False
        
        # Calculate rolling data density with adaptive window
        window = min(26, len(series) // 4)  # Adaptive window, max 6 months
        rolling_count = pd.Series(1, index=series.index).rolling(
            window=window, center=True, min_periods=1
        ).count()
        
        # Normalize to [0, 1] with minimum baseline
        max_possible = min(window, len(series))
        point_quality = (rolling_count / max_possible).values
        
        # Apply sigmoid transformation for smoother quality scores
        point_quality = 1 / (1 + np.exp(-10 * (point_quality - 0.5)))
        
        # Ensure minimum quality for non-empty periods
        point_quality = np.maximum(point_quality, 0.3)
        
        return {
            'early_period_sparse': early_sparse,
            'point_quality': point_quality,
            'overall_density': overall_density,
            'quarter_densities': densities,
            'density_category': self.get_data_density_category(overall_density),
            'adaptive_min_weeks': self.get_adaptive_min_trend_weeks(overall_density),
            'mean_quality': np.mean(point_quality)
        }
    
    def _calculate_fuzzy_membership(self, duration: float, amplitude: float, 
                                   persistence: float) -> Dict[str, float]:
        """
        Calculate fuzzy membership for trend scales.
        
        Args:
            duration: Trend duration in weeks
            amplitude: Relative change magnitude
            persistence: Trend persistence score
            
        Returns:
            Dictionary with membership degrees for each scale
        """
        # Adapt fuzzy parameters based on observed data distribution
        self._adapt_fuzzy_parameters(duration, amplitude)
        
        memberships = {}
        
        # Define membership functions for each scale
        for scale in ['short', 'medium', 'long']:
            center = self.fuzzy_params[scale]['center']
            width = self.fuzzy_params[scale]['width']
            
            # Gaussian membership function
            membership = np.exp(-0.5 * ((duration - center) / width) ** 2)
            
            # Adjust for amplitude and persistence
            if scale == 'short' and amplitude > 0.1:
                membership *= 1.2  # Short trends with high amplitude
            elif scale == 'long' and persistence > 0.8:
                membership *= 1.3  # Long trends with high persistence
            
            memberships[scale] = min(membership, 1.0)
        
        # Normalize so sum equals 1
        total = sum(memberships.values())
        if total > 0:
            memberships = {k: v/total for k, v in memberships.items()}
        
        return memberships
    
    def _adapt_fuzzy_parameters(self, duration: float, amplitude: float):
        """
        Dynamically adapt fuzzy parameters based on observed data.
        
        Args:
            duration: Current trend duration
            amplitude: Current trend amplitude
        """
        # Exponential moving average update
        alpha = 0.1
        
        # Update centers based on observed durations
        if duration < 20:
            self.fuzzy_params['short']['center'] = (
                (1 - alpha) * self.fuzzy_params['short']['center'] + 
                alpha * duration
            )
        elif duration < 52:
            self.fuzzy_params['medium']['center'] = (
                (1 - alpha) * self.fuzzy_params['medium']['center'] + 
                alpha * duration
            )
        else:
            self.fuzzy_params['long']['center'] = (
                (1 - alpha) * self.fuzzy_params['long']['center'] + 
                alpha * duration
            )
    
    def detect_trends_in_imf(self, imf: pd.Series, original: pd.Series, 
                            imf_index: int, quality_scores: np.ndarray) -> List[FuzzyTrend]:
        """
        Detect trends in a single IMF.
        
        Args:
            imf: IMF component
            original: Original series
            imf_index: Index of the IMF
            quality_scores: Data quality scores
            
        Returns:
            List of detected trends
        """
        trends = []
        
        # Skip if IMF is too noisy
        if self._is_noise_imf(imf):
            return trends
        
        # Find extrema (local maxima and minima) for better segmentation
        from scipy.signal import argrelextrema
        
        # Smooth IMF slightly for extrema detection
        smoothed_imf = gaussian_filter1d(imf.values, sigma=1, mode='nearest')
        
        # Find local extrema
        local_max = argrelextrema(smoothed_imf, np.greater, order=5)[0]
        local_min = argrelextrema(smoothed_imf, np.less, order=5)[0]
        
        # Combine and sort extrema
        extrema = np.sort(np.concatenate([local_max, local_min]))
        
        if len(extrema) < 2:
            # Entire IMF might be a single trend
            segment_quality = np.mean(quality_scores)
            adaptive_min = self.get_adaptive_min_trend_weeks(segment_quality)
            if len(imf) >= adaptive_min:
                trend = self._analyze_segment_fuzzy(
                    original, 0, len(original)-1, 
                    imf_index, quality_scores, 'emd_full'
                )
                if trend:
                    trends.append(trend)
        else:
            # Add boundaries
            segments = np.concatenate([[0], extrema, [len(imf)-1]])
            segments = np.unique(segments)  # Remove duplicates
            
            # Analyze each segment
            for i in range(len(segments)-1):
                start_idx = int(segments[i])
                end_idx = int(min(segments[i+1], len(original) - 1))
                
                # Ensure indices are valid
                if start_idx >= len(original) or end_idx >= len(original):
                    continue
                    
                # Use adaptive minimum duration based on data quality
                segment_quality = np.mean(quality_scores[start_idx:min(end_idx+1, len(quality_scores))])
                adaptive_min = self.get_adaptive_min_trend_weeks(segment_quality)
                
                if end_idx > start_idx and end_idx - start_idx >= adaptive_min:
                    trend = self._analyze_segment_fuzzy(
                        original, start_idx, end_idx,
                        imf_index, quality_scores, 'emd_segment'
                    )
                    if trend:
                        trends.append(trend)
        
        # Also detect using change point detection for comparison
        if len(original) > 20:
            try:
                change_points = self._detect_change_points(original.values)
                for i in range(len(change_points)-1):
                    start_idx = int(change_points[i])
                    end_idx = int(min(change_points[i+1] - 1, len(original) - 1))
                    
                    # Ensure indices are valid
                    if start_idx >= len(original) or end_idx >= len(original):
                        continue
                    
                    # Use adaptive minimum duration based on data quality
                    segment_quality = np.mean(quality_scores[start_idx:min(end_idx+1, len(quality_scores))])
                    adaptive_min = self.get_adaptive_min_trend_weeks(segment_quality)
                    
                    if end_idx > start_idx and end_idx - start_idx >= adaptive_min:
                        trend = self._analyze_segment_fuzzy(
                            original, start_idx, end_idx,
                            imf_index, quality_scores, 'change_point'
                        )
                        if trend and trend.confidence > 0.6:
                            trends.append(trend)
            except Exception as e:
                # Skip change point detection if it fails
                pass
        
        return trends
    
    def _detect_change_points(self, signal: np.ndarray) -> List[int]:
        """
        Detect change points using PELT algorithm.
        
        Args:
            signal: Input signal
            
        Returns:
            List of change point indices
        """
        try:
            # Use Pelt with adaptive penalty
            min_size = max(3, len(signal) // 50)
            algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
            
            # Adaptive penalty based on signal length
            penalty = np.log(len(signal)) * 2
            change_points = algo.predict(pen=penalty)
            
            # Ensure all change points are within bounds
            change_points = [cp for cp in change_points if cp < len(signal)]
            
            # Add start point if not present
            if 0 not in change_points:
                change_points = [0] + change_points
            
            # Add end point if not present
            if len(signal) not in change_points:
                change_points.append(len(signal))
            
            return sorted(change_points)
        except:
            # Fallback to simple detection
            return [0, len(signal)]
    
    def _is_noise_imf(self, imf: pd.Series, threshold: float = 0.1) -> bool:
        """
        Check if IMF is primarily noise.
        
        Args:
            imf: IMF component
            threshold: Energy threshold
            
        Returns:
            True if IMF is noise
        """
        # Calculate energy ratio
        energy = np.sum(imf.values ** 2)
        max_energy = len(imf) * np.max(np.abs(imf.values)) ** 2
        
        if max_energy > 0:
            energy_ratio = energy / max_energy
            return energy_ratio < threshold
        return True
    
    def _analyze_segment_fuzzy(self, series: pd.Series, start_idx: int, end_idx: int,
                              imf_index: int, quality_scores: np.ndarray,
                              method: str) -> Optional[FuzzyTrend]:
        """
        Analyze a segment with fuzzy classification.
        
        Args:
            series: Original series
            start_idx: Start index
            end_idx: End index  
            imf_index: IMF index
            quality_scores: Data quality scores
            method: Detection method
            
        Returns:
            FuzzyTrend object or None
        """
        if end_idx <= start_idx:
            return None
        
        segment = series.iloc[start_idx:end_idx+1]
        segment_quality = quality_scores[start_idx:end_idx+1]
        
        if len(segment) < 2:
            return None
        
        # Weight by data quality
        weights = segment_quality
        
        # Weighted linear regression
        x = np.arange(len(segment))
        y = segment.values
        
        # Remove NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return None
        
        x = x[mask]
        y = y[mask]
        weights = weights[mask]
        
        # Weighted regression
        X = sm.add_constant(x)
        wls_model = sm.WLS(y, X, weights=weights)
        results = wls_model.fit()
        
        slope = results.params[1]
        p_value = results.pvalues[1]
        
        # Mann-Kendall test with adjustment for data quality
        mk_result = self._mann_kendall_with_quality(y, weights)
        
        # Combine p-values
        combined_p = self._combine_pvalues([p_value, mk_result['p_value']], weights=weights)
        
        # Calculate changes
        start_val = segment.iloc[0]
        end_val = segment.iloc[-1]
        absolute_change = end_val - start_val
        relative_change = absolute_change / abs(start_val) if start_val != 0 else 0
        
        # Duration
        duration_weeks = end_idx - start_idx
        
        # Calculate persistence
        persistence = self._calculate_persistence(segment)
        
        # Fuzzy membership
        memberships = self._calculate_fuzzy_membership(
            duration_weeks, 
            abs(relative_change),
            persistence
        )
        
        # Primary scale
        primary_scale = max(memberships, key=memberships.get)
        
        # Direction
        direction = 'increasing' if slope > 0 else 'decreasing'
        
        # Significance level
        significance_level = self._classify_significance(combined_p)
        
        # Strength (considering fuzzy membership)
        strength = self._classify_strength_fuzzy(relative_change, memberships)
        
        # Data quality score
        data_quality_score = np.mean(segment_quality)
        
        # Confidence
        confidence = self._calculate_confidence(
            combined_p, abs(results.rsquared) if hasattr(results, 'rsquared') else 0.5, 
            abs(relative_change), data_quality_score
        )
        
        # Filter by significance with truly adaptive threshold based on data density
        density_category = self.get_data_density_category(data_quality_score)
        min_confidence = self.adaptive_confidence[density_category]
        
        # Even more lenient for very early periods with very sparse data
        if segment.index[0].year < 1990 and data_quality_score < 0.3:
            min_confidence *= 0.85  # Additional 15% reduction for pre-1990 sparse data
        
        if significance_level == 'low' and confidence < min_confidence:
            return None
        
        # Ensure indices are within bounds
        start_idx = min(start_idx, len(series) - 1)
        end_idx = min(end_idx, len(series) - 1)
        
        return FuzzyTrend(
            start_date=series.index[start_idx],
            end_date=series.index[end_idx],
            start_idx=start_idx,
            end_idx=end_idx,
            direction=direction,
            primary_scale=primary_scale,
            scale_memberships=memberships,
            duration_weeks=duration_weeks,
            slope=float(slope),
            relative_change=float(relative_change),
            absolute_change=float(absolute_change),
            raw_p_value=float(p_value),
            corrected_p_value=float(combined_p),
            significance_level=significance_level,
            strength=strength,
            method=method,
            confidence=float(confidence),
            data_quality_score=float(data_quality_score),
            imf_index=imf_index
        )
    
    def _mann_kendall_with_quality(self, x: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """
        Mann-Kendall test with quality weighting.
        
        Args:
            x: Data values
            weights: Quality weights
            
        Returns:
            Test results
        """
        n = len(x)
        s = 0
        
        # Apply power transformation to weights (less aggressive)
        adjusted_weights = weights ** self.quality_weight_power
        
        # Weighted Mann-Kendall
        for i in range(n-1):
            for j in range(i+1, n):
                # Use geometric mean for weight combination
                weight = np.sqrt(adjusted_weights[i] * adjusted_weights[j])
                s += weight * np.sign(x[j] - x[i])
        
        # Adjusted variance with correction factor
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Even less aggressive weight adjustment for sparse data
        weight_factor = np.mean(adjusted_weights)
        # Adaptive adjustment - less penalty for low quality if data is sparse
        if weight_factor < 0.5:  # Very sparse data
            weight_factor = weight_factor ** 0.5  # Square root to reduce penalty
        var_s *= weight_factor  # Single factor instead of squared
        
        # Test statistic with continuity correction
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # P-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {'statistic': s, 'p_value': p_value, 'z': z}
    
    def _combine_pvalues(self, pvalues: List[float], weights: Optional[np.ndarray] = None) -> float:
        """
        Combine multiple p-values using Fisher's method with optional weighting.
        
        Args:
            pvalues: List of p-values
            weights: Optional weights
            
        Returns:
            Combined p-value
        """
        # Remove invalid p-values
        valid_p = [p for p in pvalues if 0 < p < 1]
        
        if not valid_p:
            return 0.5
        
        if weights is not None and len(weights) == len(valid_p):
            # Weighted geometric mean
            log_p = np.log(valid_p)
            weighted_sum = np.sum(weights[:len(valid_p)] * log_p)
            chi2_stat = -2 * weighted_sum
            df = 2 * np.sum(weights[:len(valid_p)])
        else:
            # Fisher's method
            chi2_stat = -2 * np.sum(np.log(valid_p))
            df = 2 * len(valid_p)
        
        return stats.chi2.sf(chi2_stat, df)
    
    def _calculate_persistence(self, segment: pd.Series) -> float:
        """
        Calculate trend persistence score.
        
        Args:
            segment: Time series segment
            
        Returns:
            Persistence score [0, 1]
        """
        if len(segment) < 3:
            return 0.5
        
        # Count direction changes
        diff = segment.diff().dropna()
        direction_changes = np.sum(np.diff(np.sign(diff.values)) != 0)
        
        # Normalize
        max_changes = len(diff) - 1
        if max_changes > 0:
            persistence = 1 - (direction_changes / max_changes)
        else:
            persistence = 1.0
        
        return persistence
    
    def _classify_significance(self, p_value: float) -> str:
        """
        Classify significance level.
        
        Args:
            p_value: P-value
            
        Returns:
            Significance level string
        """
        for level, threshold in self.significance_levels.items():
            if p_value <= threshold:
                return level
        return 'low'
    
    def _classify_strength_fuzzy(self, relative_change: float, 
                                memberships: Dict[str, float]) -> str:
        """
        Classify trend strength considering fuzzy membership.
        
        Args:
            relative_change: Relative change magnitude
            memberships: Fuzzy scale memberships
            
        Returns:
            Strength classification
        """
        abs_change = abs(relative_change)
        
        # Adjust thresholds based on dominant scale
        dominant_scale = max(memberships, key=memberships.get)
        
        if dominant_scale == 'long':
            thresholds = {'strong': 0.3, 'moderate': 0.15, 'weak': 0.05}
        elif dominant_scale == 'medium':
            thresholds = {'strong': 0.2, 'moderate': 0.10, 'weak': 0.03}
        else:  # short
            thresholds = {'strong': 0.1, 'moderate': 0.05, 'weak': 0.02}
        
        if abs_change >= thresholds['strong']:
            return 'strong'
        elif abs_change >= thresholds['moderate']:
            return 'moderate'
        else:
            return 'weak'
    
    def _calculate_confidence(self, p_value: float, r_squared: float,
                            relative_change: float, quality_score: float) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            p_value: P-value
            r_squared: R-squared value
            relative_change: Relative change magnitude
            quality_score: Data quality score
            
        Returns:
            Confidence score [0, 1]
        """
        components = [
            1 - p_value,                        # Statistical significance
            r_squared,                           # Model fit
            min(abs(relative_change) * 2, 1),   # Effect size
            quality_score                        # Data quality
        ]
        
        # Weighted average with adjusted weights
        weights = [0.35, 0.15, 0.35, 0.15]  # More weight on significance and effect size
        confidence = np.sum(np.array(components) * np.array(weights))
        
        return min(max(confidence, 0), 1)
    
    def analyze_complete(self, series: pd.Series) -> Dict[str, Any]:
        """
        Complete analysis with EMD and fuzzy classification.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with all analysis results
        """
        results = {
            'trends': [],
            'components': {},
            'critical_points': [],
            'summary': {}
        }
        
        # EMD decomposition
        components = self.decompose_with_emd(series, method='ceemdan')
        results['components'] = components
        
        # Get quality scores
        quality_info = self._assess_data_quality(series)
        quality_scores = quality_info['point_quality']
        
        # Detect trends in each IMF
        all_trends = []
        
        for key, component in components.items():
            if key.startswith('IMF_'):
                imf_index = int(key.split('_')[1])
                trends = self.detect_trends_in_imf(
                    component, series, imf_index, quality_scores
                )
                all_trends.extend(trends)
        
        # Merge overlapping trends
        merged_trends = self._merge_fuzzy_trends(all_trends)
        
        # Apply multiple testing correction with adaptive method
        if merged_trends:
            p_values = [t.raw_p_value for t in merged_trends]
            
            # Use less conservative correction for smaller numbers of tests
            if len(p_values) < 20:
                # Bonferroni-Holm for small numbers
                rejected, corrected_p, _, _ = multipletests(
                    p_values, alpha=0.05, method='holm'
                )
            else:
                # FDR for larger numbers
                rejected, corrected_p, _, _ = multipletests(
                    p_values, alpha=0.05, method='fdr_bh'
                )
            
            for trend, new_p in zip(merged_trends, corrected_p):
                trend.corrected_p_value = float(new_p)
                trend.significance_level = self._classify_significance(new_p)
        
        results['trends'] = merged_trends
        
        # Identify critical points
        results['critical_points'] = self.identify_critical_points(series, merged_trends)
        
        # Summary statistics
        results['summary'] = self._create_summary(merged_trends, quality_info)
        
        return results
    
    def _merge_fuzzy_trends(self, trends: List[FuzzyTrend]) -> List[FuzzyTrend]:
        """
        Merge overlapping trends with fuzzy logic.
        
        Args:
            trends: List of trends
            
        Returns:
            Merged trends
        """
        if not trends:
            return []
        
        # Sort by start date and confidence
        trends.sort(key=lambda t: (t.start_date, -t.confidence))
        
        merged = []
        current = trends[0]
        
        for trend in trends[1:]:
            # Check overlap and compatibility
            overlap_ratio = self._calculate_overlap_ratio(current, trend)
            
            # More aggressive merging with adaptive threshold
            merge_threshold = self.overlap_threshold
            if current.primary_scale == trend.primary_scale:
                merge_threshold *= 0.8  # Lower threshold for same scale
            
            if (overlap_ratio > merge_threshold and 
                trend.direction == current.direction and
                self._are_scales_compatible(current.scale_memberships, 
                                          trend.scale_memberships)):
                
                # Merge with fuzzy logic
                current = self._merge_two_trends(current, trend)
            elif overlap_ratio > 0.1 and trend.direction == current.direction:
                # Check for continuity (small gap between trends)
                gap = trend.start_idx - current.end_idx
                if 0 < gap <= 4:  # Allow small gaps up to 4 weeks
                    current = self._merge_two_trends(current, trend)
                else:
                    merged.append(current)
                    current = trend
            else:
                merged.append(current)
                current = trend
        
        merged.append(current)
        
        # Post-process to ensure minimum duration - use most lenient threshold
        min_duration = self.get_adaptive_min_trend_weeks(0.2)  # Use very sparse threshold as minimum
        merged = [t for t in merged if t.duration_weeks >= min_duration]
        
        return merged
    
    def _calculate_overlap_ratio(self, trend1: FuzzyTrend, trend2: FuzzyTrend) -> float:
        """Calculate overlap ratio between two trends."""
        overlap_start = max(trend1.start_idx, trend2.start_idx)
        overlap_end = min(trend1.end_idx, trend2.end_idx)
        
        if overlap_end < overlap_start:
            return 0.0
        
        overlap_length = overlap_end - overlap_start + 1
        min_length = min(trend1.duration_weeks, trend2.duration_weeks)
        
        return overlap_length / min_length if min_length > 0 else 0.0
    
    def _are_scales_compatible(self, memberships1: Dict[str, float],
                              memberships2: Dict[str, float]) -> bool:
        """Check if two fuzzy memberships are compatible."""
        # Calculate similarity using cosine similarity
        scales = ['short', 'medium', 'long']
        vec1 = np.array([memberships1.get(s, 0) for s in scales])
        vec2 = np.array([memberships2.get(s, 0) for s in scales])
        
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product > 0:
            similarity = dot_product / norm_product
            # More lenient compatibility threshold
            return similarity > 0.6
        return False
    
    def _merge_two_trends(self, trend1: FuzzyTrend, trend2: FuzzyTrend) -> FuzzyTrend:
        """Merge two trends with fuzzy logic."""
        # Combine memberships
        combined_memberships = {}
        for scale in ['short', 'medium', 'long']:
            m1 = trend1.scale_memberships.get(scale, 0)
            m2 = trend2.scale_memberships.get(scale, 0)
            # Weighted average based on duration
            w1 = trend1.duration_weeks
            w2 = trend2.duration_weeks
            combined_memberships[scale] = (w1 * m1 + w2 * m2) / (w1 + w2)
        
        # Normalize
        total = sum(combined_memberships.values())
        if total > 0:
            combined_memberships = {k: v/total for k, v in combined_memberships.items()}
        
        # Create merged trend
        return FuzzyTrend(
            start_date=min(trend1.start_date, trend2.start_date),
            end_date=max(trend1.end_date, trend2.end_date),
            start_idx=min(trend1.start_idx, trend2.start_idx),
            end_idx=max(trend1.end_idx, trend2.end_idx),
            direction=trend1.direction,
            primary_scale=max(combined_memberships, key=combined_memberships.get),
            scale_memberships=combined_memberships,
            duration_weeks=max(trend1.end_idx, trend2.end_idx) - min(trend1.start_idx, trend2.start_idx),
            slope=(trend1.slope + trend2.slope) / 2,
            relative_change=trend1.relative_change,  # Will recalculate if needed
            absolute_change=trend1.absolute_change,  # Will recalculate if needed
            raw_p_value=min(trend1.raw_p_value, trend2.raw_p_value),
            corrected_p_value=min(trend1.corrected_p_value, trend2.corrected_p_value),
            significance_level=trend1.significance_level if trend1.corrected_p_value < trend2.corrected_p_value else trend2.significance_level,
            strength=trend1.strength,
            method=f"{trend1.method}+{trend2.method}",
            confidence=max(trend1.confidence, trend2.confidence),
            data_quality_score=(trend1.data_quality_score + trend2.data_quality_score) / 2,
            imf_index=trend1.imf_index  # Keep first
        )
    
    def identify_critical_points(self, series: pd.Series, trends: List[FuzzyTrend]) -> List[Dict]:
        """
        Identify critical points with multi-level significance.
        
        Args:
            series: Original series
            trends: List of trends
            
        Returns:
            List of critical points
        """
        critical_points = []
        
        for trend in trends:
            # Start point
            critical_points.append({
                'date': trend.start_date,
                'type': f'{trend.direction}_start',
                'primary_scale': trend.primary_scale,
                'scale_memberships': trend.scale_memberships,
                'significance_level': trend.significance_level,
                'strength': trend.strength,
                'trend_duration': trend.duration_weeks,
                'confidence': trend.confidence,
                'data_quality': trend.data_quality_score,
                'value': float(series.iloc[trend.start_idx]),
                'imf_index': trend.imf_index
            })
            
            # End point (if not at boundary)
            if trend.end_idx < len(series) - 2:
                critical_points.append({
                    'date': trend.end_date,
                    'type': f'{trend.direction}_end',
                    'primary_scale': trend.primary_scale,
                    'scale_memberships': trend.scale_memberships,
                    'significance_level': trend.significance_level,
                    'strength': trend.strength,
                    'trend_duration': trend.duration_weeks,
                    'confidence': trend.confidence,
                    'data_quality': trend.data_quality_score,
                    'value': float(series.iloc[trend.end_idx]),
                    'imf_index': trend.imf_index
                })
        
        # Sort by date
        critical_points.sort(key=lambda x: x['date'])
        
        return critical_points
    
    def _create_summary(self, trends: List[FuzzyTrend], quality_info: Dict) -> Dict:
        """Create summary statistics."""
        if not trends:
            return {
                'total_trends': 0,
                'data_quality': quality_info
            }
        
        # Count by significance level
        sig_counts = {}
        for level in self.significance_levels.keys():
            sig_counts[level] = sum(1 for t in trends if t.significance_level == level)
        
        # Average memberships by scale
        avg_memberships = {'short': [], 'medium': [], 'long': []}
        for trend in trends:
            for scale in avg_memberships:
                avg_memberships[scale].append(trend.scale_memberships.get(scale, 0))
        
        avg_memberships = {k: np.mean(v) if v else 0 for k, v in avg_memberships.items()}
        
        return {
            'total_trends': len(trends),
            'significance_distribution': sig_counts,
            'average_scale_memberships': avg_memberships,
            'average_confidence': np.mean([t.confidence for t in trends]),
            'average_data_quality': np.mean([t.data_quality_score for t in trends]),
            'data_quality_info': quality_info
        }