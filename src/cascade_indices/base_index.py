"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
base_index.py

MAIN OBJECTIVE:
---------------
This script provides the abstract base class for all cascade sub-indices,
establishing a standardized interface and common functionality for cascade measurement.

Dependencies:
-------------
- abc
- typing
- pandas
- numpy
- dataclasses

MAIN FEATURES:
--------------
1) Abstract base class definition for cascade indices
2) Standardized result container with scores and metadata
3) Common validation and normalization methods
4) Interface for sub-index calculations
5) Confidence scoring framework

Author:
-------
Antoine Lemor
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class IndexResult:
    """Container for sub-index calculation results."""
    score: float  # Main score [0, 1]
    sub_scores: Dict[str, float]  # Individual sub-metric scores
    metadata: Dict[str, Any]  # Additional metadata (counts, lists, etc.)
    confidence: float  # Confidence level of the calculation [0, 1]
    details: str  # Human-readable description of the result


class BaseCascadeIndex(ABC):
    """Abstract base class for cascade sub-indices."""
    
    def __init__(self, name: str, weight: float = 0.2):
        """
        Initialize base index.
        
        Args:
            name: Name of the index
            weight: Weight in the overall cascade score (default 0.2 for 5 components)
        """
        self.name = name
        self.weight = weight
        self.last_result = None
        
    @abstractmethod
    def calculate(self, 
                  data: pd.DataFrame,
                  frame: str,
                  reference_date: pd.Timestamp,
                  window_days: int = 180,
                  **kwargs) -> IndexResult:
        """
        Calculate the sub-index for a given cascade window.
        
        Args:
            data: Complete frame detection data
            frame: Frame name (e.g., 'Eco', 'Pol')
            reference_date: Reference date for the cascade
            window_days: Window size in days for analysis
            **kwargs: Additional parameters specific to each index
            
        Returns:
            IndexResult with calculated scores and metadata
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate that the data has required columns and format.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    def get_window_data(self, 
                        data: pd.DataFrame,
                        reference_date: pd.Timestamp,
                        window_days: int = 180) -> pd.DataFrame:
        """
        Extract data for the analysis window.
        
        Args:
            data: Complete dataset
            reference_date: Reference date
            window_days: Window size in days
            
        Returns:
            Filtered DataFrame for the window
        """
        start_date = reference_date - pd.Timedelta(days=window_days // 2)
        end_date = reference_date + pd.Timedelta(days=window_days // 2)
        
        return data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
    
    def normalize_score(self, value: float, min_val: float = 0, max_val: float = 1) -> float:
        """
        Normalize a value to [0, 1] range.
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value in [0, 1]
        """
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    
    def calculate_confidence(self, data_points: int, min_points: int = 10) -> float:
        """
        Calculate confidence based on data availability.
        
        Args:
            data_points: Number of available data points
            min_points: Minimum points for full confidence
            
        Returns:
            Confidence score [0, 1]
        """
        if data_points >= min_points:
            return 1.0
        elif data_points > 0:
            return data_points / min_points
        else:
            return 0.0
    
    def apply_sigmoid_transformation(self, value: float, k: float = 10, x0: float = 0.5) -> float:
        """
        Apply sigmoid transformation for smooth scoring.
        
        Args:
            value: Input value [0, 1]
            k: Steepness parameter
            x0: Midpoint
            
        Returns:
            Transformed value [0, 1]
        """
        return 1 / (1 + np.exp(-k * (value - x0)))
    
    def calculate_trend(self, series: pd.Series, method: str = 'linear') -> float:
        """
        Calculate trend in a time series.
        
        Args:
            series: Time series data
            method: Trend calculation method ('linear', 'robust', 'exponential')
            
        Returns:
            Trend coefficient
        """
        if len(series) < 2:
            return 0
        
        x = np.arange(len(series))
        y = series.values
        
        if method == 'linear':
            if len(x) > 1:
                coef = np.polyfit(x, y, 1)[0]
                return coef
            return 0
        elif method == 'robust':
            # Use median-based trend
            mid = len(y) // 2
            first_half = np.median(y[:mid])
            second_half = np.median(y[mid:])
            return (second_half - first_half) / (len(y) / 2)
        elif method == 'exponential':
            # Log transform for exponential trend
            y_log = np.log(y + 1e-10)
            if len(x) > 1:
                coef = np.polyfit(x, y_log, 1)[0]
                return np.exp(coef) - 1
            return 0
        
        return 0
    
    def detect_change_points(self, series: pd.Series, sensitivity: float = 2.0) -> List[int]:
        """
        Detect change points in a time series.
        
        Args:
            series: Time series data
            sensitivity: Sensitivity parameter (lower = more sensitive)
            
        Returns:
            List of change point indices
        """
        if len(series) < 3:
            return []
        
        # Calculate rolling statistics
        window = max(3, len(series) // 10)
        rolling_mean = series.rolling(window, center=True).mean()
        rolling_std = series.rolling(window, center=True).std()
        
        # Detect points outside bounds
        upper_bound = rolling_mean + sensitivity * rolling_std
        lower_bound = rolling_mean - sensitivity * rolling_std
        
        change_points = []
        for i in range(len(series)):
            if pd.notna(upper_bound.iloc[i]) and pd.notna(lower_bound.iloc[i]):
                if series.iloc[i] > upper_bound.iloc[i] or series.iloc[i] < lower_bound.iloc[i]:
                    change_points.append(i)
        
        return change_points
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"