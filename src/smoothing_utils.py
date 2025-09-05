"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
smoothing_utils.py

MAIN OBJECTIVE:
---------------
This script provides advanced smoothing functions for time series data
optimized for professional scientific visualizations.

Dependencies:
-------------
- numpy
- pandas
- scipy
- statsmodels

MAIN FEATURES:
--------------
1) Rolling average with various window types
2) Savitzky-Golay filter for smooth derivatives
3) LOWESS smoothing for robust trend estimation
4) Combined smoothing methods

Author:
-------
Antoine Lemor
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate, ndimage
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Optional, Union


class TimeSeriesSmoother:
    """Advanced smoothing methods for time series data."""
    
    @staticmethod
    def rolling_average(data: pd.Series, window: int = 12, 
                       min_periods: Optional[int] = None,
                       center: bool = True,
                       win_type: Optional[str] = None) -> pd.Series:
        """
        Apply rolling average with optional window function.
        
        Args:
            data: Time series data
            window: Window size (default 12 for quarterly smoothing)
            min_periods: Minimum observations in window
            center: Center the window
            win_type: Window type ('gaussian', 'triang', 'blackman', etc.)
            
        Returns:
            Smoothed series
        """
        if min_periods is None:
            min_periods = window // 2
        
        if win_type:
            if win_type == 'gaussian':
                # For gaussian, std is required
                return data.rolling(window, center=center, min_periods=min_periods,
                                  win_type=win_type).mean(std=window/4)
            else:
                return data.rolling(window, center=center, min_periods=min_periods,
                                  win_type=win_type).mean()
        else:
            # Simple rolling average
            return data.rolling(window, center=center, min_periods=min_periods).mean()
    
    @staticmethod
    def savitzky_golay(data: pd.Series, window_length: int = 13, 
                       polyorder: int = 3) -> pd.Series:
        """
        Apply Savitzky-Golay filter for smooth differentiation.
        
        Args:
            data: Time series data
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial to fit
            
        Returns:
            Smoothed series
        """
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Handle NaN values
        mask = ~np.isnan(data.values)
        y = data.values.copy()
        
        # Apply filter only to non-NaN values
        if np.sum(mask) > window_length:
            y[mask] = signal.savgol_filter(y[mask], window_length, polyorder)
        
        return pd.Series(y, index=data.index)
    
    @staticmethod
    def lowess_smooth(data: pd.Series, frac: float = 0.15, 
                     iterations: int = 3) -> pd.Series:
        """
        Apply LOWESS (Locally Weighted Scatterplot Smoothing).
        
        Args:
            data: Time series data
            frac: Fraction of data used for smoothing at each point
            iterations: Number of residual-based reweightings
            
        Returns:
            Smoothed series
        """
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) < 3:
            return data
        
        # Create numeric x values
        x = np.arange(len(clean_data))
        y = clean_data.values
        
        # Apply LOWESS
        smoothed = lowess(y, x, frac=frac, it=iterations, return_sorted=False)
        
        # Return as series with original index
        result = pd.Series(index=data.index, dtype=float)
        result.loc[clean_data.index] = smoothed
        
        return result
    
    @staticmethod
    def gaussian_filter(data: pd.Series, sigma: float = 2.0) -> pd.Series:
        """
        Apply Gaussian filter for smoothing.
        
        Args:
            data: Time series data
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Smoothed series
        """
        # Handle NaN values
        mask = ~np.isnan(data.values)
        y = data.values.copy()
        
        if np.sum(mask) > 3:
            # Interpolate NaN values first
            x = np.arange(len(y))
            y_interpolated = np.interp(x, x[mask], y[mask])
            
            # Apply Gaussian filter
            y_smooth = ndimage.gaussian_filter1d(y_interpolated, sigma=sigma)
            
            # Put back NaN values where they were
            y[mask] = y_smooth[mask]
        
        return pd.Series(y, index=data.index)
    
    @staticmethod
    def combined_smooth(data: pd.Series, method1: str = 'rolling', 
                       method2: str = 'savitzky_golay',
                       **kwargs) -> pd.Series:
        """
        Apply two smoothing methods in sequence for better results.
        
        Args:
            data: Time series data
            method1: First smoothing method
            method2: Second smoothing method
            **kwargs: Arguments for smoothing methods
            
        Returns:
            Smoothed series
        """
        smoother = TimeSeriesSmoother()
        
        # First pass
        if method1 == 'rolling':
            smooth1 = smoother.rolling_average(data, **kwargs.get('rolling_params', {}))
        elif method1 == 'lowess':
            smooth1 = smoother.lowess_smooth(data, **kwargs.get('lowess_params', {}))
        elif method1 == 'gaussian':
            smooth1 = smoother.gaussian_filter(data, **kwargs.get('gaussian_params', {}))
        else:
            smooth1 = data
        
        # Second pass
        if method2 == 'savitzky_golay':
            smooth2 = smoother.savitzky_golay(smooth1, **kwargs.get('savgol_params', {}))
        elif method2 == 'gaussian':
            smooth2 = smoother.gaussian_filter(smooth1, **kwargs.get('gaussian_params', {}))
        elif method2 == 'lowess':
            smooth2 = smoother.lowess_smooth(smooth1, **kwargs.get('lowess_params', {}))
        else:
            smooth2 = smooth1
        
        return smooth2
    
    @staticmethod
    def optimal_smooth_for_visualization(data: pd.Series, 
                                       smoothness: str = 'medium') -> pd.Series:
        """
        Apply optimal smoothing for scientific visualization.
        
        Args:
            data: Time series data
            smoothness: Level of smoothing ('light', 'medium', 'heavy')
            
        Returns:
            Smoothed series
        """
        smoother = TimeSeriesSmoother()
        
        if smoothness == 'light':
            # Light smoothing - preserve more detail
            return smoother.rolling_average(data, window=6, center=True)
        
        elif smoothness == 'medium':
            # Medium smoothing - good balance
            return smoother.combined_smooth(
                data,
                method1='rolling',
                method2='savitzky_golay',
                rolling_params={'window': 12, 'center': True},
                savgol_params={'window_length': 13, 'polyorder': 3}
            )
        
        elif smoothness == 'heavy':
            # Heavy smoothing - very smooth curves
            return smoother.combined_smooth(
                data,
                method1='rolling',
                method2='lowess',
                rolling_params={'window': 26, 'center': True, 'win_type': 'gaussian'},
                lowess_params={'frac': 0.15, 'iterations': 3}
            )
        
        else:
            # Default to medium
            return smoother.optimal_smooth_for_visualization(data, 'medium')


def smooth_for_publication(df: pd.DataFrame, columns: list, 
                          time_col: str, smoothness: str = 'medium') -> pd.DataFrame:
    """
    Smooth multiple columns in a DataFrame for publication-quality plots.
    
    Args:
        df: DataFrame with time series data
        columns: List of column names to smooth
        time_col: Name of time column
        smoothness: Level of smoothing
        
    Returns:
        DataFrame with additional smoothed columns
    """
    df = df.copy()
    df = df.sort_values(time_col)
    
    smoother = TimeSeriesSmoother()
    
    for col in columns:
        if col in df.columns:
            df[f'{col}_smooth'] = smoother.optimal_smooth_for_visualization(
                df[col], smoothness
            )
    
    return df