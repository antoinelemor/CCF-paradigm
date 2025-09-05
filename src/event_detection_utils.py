"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
event_detection_utils.py

MAIN OBJECTIVE:
---------------
This module provides utilities for rigorous event detection and impact analysis.
It implements methods to distinguish between event mentions and actual event occurrences,
and to analyze their impact on media cascades and frame changes.

Dependencies:
-------------
- pandas
- numpy
- scipy
- sklearn
- tqdm

MAIN FEATURES:
--------------
1) Rigorous event occurrence detection using temporal clustering
2) Event burst detection with adaptive thresholds
3) Multi-media validation for event confirmation
4) Impact window analysis for cascade effects
5) Event-frame co-occurrence patterns
6) Statistical validation of event impacts

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Set
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class EventDetector:
    """Detects actual event occurrences from event mentions in articles."""
    
    # Event type definitions
    EVENT_TYPES = {
        'Event_1_SUB': 'Extreme meteorological event',
        'Event_2_SUB': 'Meeting/Conference', 
        'Event_3_SUB': 'Publication',
        'Event_4_SUB': 'Election',
        'Event_5_SUB': 'Policy announcement',
        'Event_6_SUB': 'Judiciary decision',
        'Event_7_SUB': 'Cultural event',
        'Event_8_SUB': 'Protest'
    }
    
    def __init__(self, 
                 min_mentions_threshold: int = 10,
                 temporal_window_days: int = 7,
                 min_media_diversity: int = 3,
                 burst_factor: float = 2.0,
                 n_workers: Optional[int] = None):
        """
        Initialize event detector.
        
        Args:
            min_mentions_threshold: Minimum mentions to consider an event
            temporal_window_days: Window for temporal clustering
            min_media_diversity: Minimum different media outlets
            burst_factor: Factor above baseline for burst detection
            n_workers: Number of parallel workers
        """
        self.min_mentions_threshold = min_mentions_threshold
        self.temporal_window_days = temporal_window_days
        self.min_media_diversity = min_media_diversity
        self.burst_factor = burst_factor
        self.n_workers = n_workers or min(cpu_count(), 8)
    
    @staticmethod
    def _convert_to_timestamp(date_value):
        """Convert various date formats to pandas Timestamp."""
        if pd.isna(date_value):
            return pd.NaT
        elif isinstance(date_value, pd.Timestamp):
            return date_value
        elif isinstance(date_value, np.datetime64):
            return pd.Timestamp(date_value)
        elif isinstance(date_value, datetime):
            return pd.Timestamp(date_value)
        else:
            try:
                return pd.Timestamp(date_value)
            except:
                return pd.NaT
    
    def detect_event_occurrences(self, df: pd.DataFrame, 
                               show_progress: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Detect actual event occurrences from mentions.
        
        Args:
            df: DataFrame with event detection columns
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping event types to detected occurrences
        """
        detected_events = {}
        
        # Ensure all dates are pandas Timestamps
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = df['date'].apply(self._convert_to_timestamp)
        
        # Process each event type
        event_cols = [col for col in df.columns if col.startswith('Event_') and col.endswith('_SUB')]
        
        if show_progress:
            pbar = tqdm(total=len(event_cols), desc="Detecting event occurrences")
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            
            for event_col in event_cols:
                # Get mentions for this event type
                event_mentions = df[df[event_col] == 1].copy()
                
                if len(event_mentions) >= self.min_mentions_threshold:
                    future = executor.submit(
                        self._detect_single_event_type,
                        event_mentions,
                        event_col,
                        df
                    )
                    futures[future] = event_col
            
            for future in as_completed(futures):
                event_col = futures[future]
                try:
                    occurrences = future.result()
                    if occurrences is not None and len(occurrences) > 0:
                        detected_events[event_col] = occurrences
                except Exception as e:
                    logger.error(f"Error detecting {event_col}: {e}")
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return detected_events
    
    def _detect_single_event_type(self, event_mentions: pd.DataFrame, 
                                event_col: str, full_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Detect occurrences for a single event type."""
        if len(event_mentions) == 0:
            return None
        
        # Ensure dates are timestamps
        event_mentions = event_mentions.copy()
        event_mentions['date'] = event_mentions['date'].apply(self._convert_to_timestamp)
        
        # Sort by date
        event_mentions = event_mentions.sort_values('date')
        
        # 1. Temporal clustering
        temporal_clusters = self._temporal_clustering(event_mentions)
        
        # 2. Filter clusters by size and media diversity
        valid_clusters = []
        
        for cluster_idx, cluster_dates in temporal_clusters.items():
            cluster_data = event_mentions[event_mentions['date'].isin(cluster_dates)]
            
            # Check minimum mentions
            if len(cluster_data) < self.min_mentions_threshold:
                continue
                
            # Check media diversity
            unique_media = cluster_data['media'].nunique()
            if unique_media < self.min_media_diversity:
                continue
                
            # Check burst pattern
            if self._is_burst_pattern(cluster_data, full_df):
                # Calculate event metadata
                event_info = self._calculate_event_metadata(cluster_data, event_col)
                valid_clusters.append(event_info)
        
        if valid_clusters:
            return pd.DataFrame(valid_clusters)
        return None
    
    def _temporal_clustering(self, mentions: pd.DataFrame) -> Dict[int, List[pd.Timestamp]]:
        """Cluster mentions temporally using DBSCAN."""
        # Convert dates to ordinal for clustering
        dates = mentions['date'].values
        
        # Convert to ordinal, handling both numpy datetime64 and pandas Timestamp
        ordinal_dates = []
        for d in dates:
            # Convert to pandas Timestamp first
            ts = self._convert_to_timestamp(d)
            if pd.notna(ts):
                # For pandas Timestamp, use .toordinal()
                ordinal_dates.append(ts.toordinal())
            else:
                ordinal_dates.append(np.nan)
        
        # Remove NaN values
        valid_indices = [i for i, val in enumerate(ordinal_dates) if not np.isnan(val)]
        ordinal_dates_clean = np.array([ordinal_dates[i] for i in valid_indices]).reshape(-1, 1)
        
        if len(ordinal_dates_clean) == 0:
            return {}
        
        # Cluster with DBSCAN
        clustering = DBSCAN(
            eps=self.temporal_window_days,
            min_samples=max(1, self.min_mentions_threshold // 2)
        ).fit(ordinal_dates_clean)
        
        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Ignore noise points
                original_idx = valid_indices[idx]
                clusters[label].append(mentions.iloc[original_idx]['date'])
        
        return dict(clusters)
    
    def _is_burst_pattern(self, cluster_data: pd.DataFrame, 
                         full_df: pd.DataFrame) -> bool:
        """Check if cluster represents a burst pattern."""
        # Ensure dates are timestamps
        cluster_data = cluster_data.copy()
        cluster_data['date'] = cluster_data['date'].apply(self._convert_to_timestamp)
        full_df = full_df.copy()
        if 'date' in full_df.columns:
            full_df['date'] = full_df['date'].apply(self._convert_to_timestamp)
        
        # Calculate baseline frequency
        event_col = [col for col in cluster_data.columns 
                    if col.startswith('Event_') and col.endswith('_SUB')][0]
        
        # Get baseline period (3 months before cluster)
        cluster_start = cluster_data['date'].min()
        baseline_start = cluster_start - timedelta(days=90)
        baseline_end = cluster_start - timedelta(days=1)
        
        baseline_data = full_df[
            (full_df['date'] >= baseline_start) & 
            (full_df['date'] <= baseline_end)
        ]
        
        if len(baseline_data) == 0:
            return True  # No baseline, consider as burst
        
        # Calculate frequencies
        baseline_freq = (baseline_data[event_col] == 1).sum() / 90  # Per day
        cluster_duration = (cluster_data['date'].max() - cluster_data['date'].min()).days + 1
        cluster_freq = len(cluster_data) / max(1, cluster_duration)
        
        # Check if burst
        return cluster_freq > baseline_freq * self.burst_factor
    
    def _calculate_event_metadata(self, cluster_data: pd.DataFrame, 
                                event_col: str) -> Dict:
        """Calculate metadata for detected event."""
        # Ensure dates are timestamps
        cluster_data = cluster_data.copy()
        cluster_data['date'] = cluster_data['date'].apply(self._convert_to_timestamp)
        
        # Find peak date (most mentions)
        daily_counts = cluster_data.groupby('date').size()
        peak_date = daily_counts.idxmax()
        
        # Calculate intensity metrics
        total_mentions = len(cluster_data)
        unique_media = cluster_data['media'].nunique()
        unique_authors = cluster_data['author'].nunique()
        duration_days = (cluster_data['date'].max() - cluster_data['date'].min()).days + 1
        
        # Frame co-occurrence
        frame_cols = ['Cult_Detection', 'Eco_Detection', 'Envt_Detection', 'Pbh_Detection',
                     'Just_Detection', 'Pol_Detection', 'Sci_Detection', 'Secu_Detection']
        
        frame_cooccurrence = {}
        for frame_col in frame_cols:
            if frame_col in cluster_data.columns:
                frame_cooccurrence[frame_col.replace('_Detection', '')] = \
                    cluster_data[frame_col].sum() / len(cluster_data)
        
        return {
            'event_type': event_col,
            'event_name': self.EVENT_TYPES.get(event_col, event_col),
            'start_date': cluster_data['date'].min(),
            'end_date': cluster_data['date'].max(),
            'peak_date': peak_date,
            'total_mentions': int(total_mentions),
            'unique_media': int(unique_media),
            'unique_authors': int(unique_authors),
            'duration_days': int(duration_days),
            'intensity': float(total_mentions / max(1, duration_days)),
            'media_diversity_score': float(unique_media / total_mentions),
            'dominant_frame': max(frame_cooccurrence, key=frame_cooccurrence.get) if frame_cooccurrence else None,
            'frame_cooccurrence': frame_cooccurrence
        }


class EventImpactAnalyzer:
    """Analyzes the impact of detected events on cascades and frame changes."""
    
    def __init__(self, 
                 pre_event_window: int = 28,  # 4 weeks
                 post_event_window: int = 56,  # 8 weeks
                 min_cascade_strength: float = 0.3):
        """
        Initialize impact analyzer.
        
        Args:
            pre_event_window: Days before event to analyze
            post_event_window: Days after event to analyze
            min_cascade_strength: Minimum cascade strength to consider
        """
        self.pre_event_window = pre_event_window
        self.post_event_window = post_event_window
        self.min_cascade_strength = min_cascade_strength
    
    @staticmethod
    def _ensure_timestamp(date_value):
        """Ensure date value is a pandas Timestamp."""
        if pd.isna(date_value):
            return pd.NaT
        elif isinstance(date_value, pd.Timestamp):
            return date_value
        elif isinstance(date_value, (np.datetime64, datetime)):
            return pd.Timestamp(date_value)
        else:
            try:
                return pd.Timestamp(date_value)
            except:
                return pd.NaT
    
    def analyze_event_impacts(self, 
                            detected_events: Dict[str, pd.DataFrame],
                            cascade_results: pd.DataFrame,
                            trend_results: pd.DataFrame,
                            frame_data: pd.DataFrame,
                            show_progress: bool = True) -> pd.DataFrame:
        """
        Analyze impact of detected events on cascades and trends.
        
        Args:
            detected_events: Dictionary of detected event occurrences
            cascade_results: Results from cascade analysis
            trend_results: Results from trend analysis
            frame_data: Original frame data
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with event impact analysis
        """
        all_impacts = []
        
        # Ensure all dates are timestamps
        cascade_results = cascade_results.copy()
        if 'critical_date' in cascade_results.columns:
            cascade_results['critical_date'] = cascade_results['critical_date'].apply(self._ensure_timestamp)
        
        # Ensure numeric columns are properly typed
        cascade_numeric_cols = ['cascade_strength', 'media_count', 'magnitude_increase', 
                               'duration_weeks', 'start_week', 'peak_week', 'end_week']
        for col in cascade_numeric_cols:
            if col in cascade_results.columns:
                cascade_results[col] = pd.to_numeric(cascade_results[col], errors='coerce')
        
        trend_results = trend_results.copy()
        if 'date' in trend_results.columns:
            trend_results['date'] = trend_results['date'].apply(self._ensure_timestamp)
        
        frame_data = frame_data.copy()
        if 'date' in frame_data.columns:
            frame_data['date'] = frame_data['date'].apply(self._ensure_timestamp)
        
        # Flatten all events
        event_list = []
        for event_type, events_df in detected_events.items():
            for _, event in events_df.iterrows():
                # Ensure event dates are timestamps
                event_copy = event.copy()
                for date_col in ['start_date', 'end_date', 'peak_date']:
                    if date_col in event_copy.index:
                        event_copy[date_col] = self._ensure_timestamp(event_copy[date_col])
                event_list.append(event_copy)
        
        if show_progress:
            pbar = tqdm(total=len(event_list), desc="Analyzing event impacts")
        
        for event in event_list:
            try:
                # Analyze cascade impact
                cascade_impact = self._analyze_cascade_impact(
                    event, cascade_results
                )
                
                # Analyze trend impact
                trend_impact = self._analyze_trend_impact(
                    event, trend_results
                )
                
                # Analyze frame shift
                frame_shift = self._analyze_frame_shift(
                    event, frame_data
                )
                
                # Combine results
                impact_result = {
                    **event.to_dict(),
                    'cascade_impact': cascade_impact,
                    'trend_impact': trend_impact,
                    'frame_shift': frame_shift,
                    'overall_impact_score': self._calculate_overall_impact(
                        cascade_impact, trend_impact, frame_shift
                    )
                }
                
                all_impacts.append(impact_result)
                
            except Exception as e:
                logger.warning(f"Error analyzing impact for event {event.get('event_type', 'unknown')}: {e}")
                # Add minimal impact result to continue processing
                impact_result = {
                    **event.to_dict(),
                    'cascade_impact': {'triggered_cascades': 0, 'avg_cascade_strength': 0, 
                                     'cascade_lag_days': None, 'affected_frames': []},
                    'trend_impact': {'triggered_trends': 0, 'trend_scales': [], 
                                   'trend_directions': [], 'trend_lag_days': None},
                    'frame_shift': {'significant_shifts': [], 'max_shift_magnitude': 0},
                    'overall_impact_score': 0
                }
                all_impacts.append(impact_result)
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return pd.DataFrame(all_impacts)
    
    def _analyze_cascade_impact(self, event: pd.Series, 
                              cascade_results: pd.DataFrame) -> Dict:
        """Analyze impact on media cascades."""
        # Find cascades within post-event window
        event_date = self._ensure_timestamp(event['peak_date'])
        window_start = event_date
        window_end = event_date + timedelta(days=self.post_event_window)
        
        # Ensure cascade_strength is numeric
        cascade_results = cascade_results.copy()
        if 'cascade_strength' in cascade_results.columns:
            cascade_results['cascade_strength'] = pd.to_numeric(
                cascade_results['cascade_strength'], errors='coerce'
            )
        
        # Filter cascades
        nearby_cascades = cascade_results[
            (cascade_results['critical_date'] >= window_start) &
            (cascade_results['critical_date'] <= window_end) &
            (cascade_results['cascade_strength'] >= self.min_cascade_strength)
        ]
        
        if len(nearby_cascades) == 0:
            return {
                'triggered_cascades': 0,
                'avg_cascade_strength': 0,
                'cascade_lag_days': None,
                'affected_frames': []
            }
        
        # Calculate metrics
        triggered_cascades = len(nearby_cascades)
        avg_strength = nearby_cascades['cascade_strength'].mean()
        
        # Find closest cascade
        time_diffs = (nearby_cascades['critical_date'] - event_date).dt.days
        closest_idx = time_diffs.abs().idxmin()
        cascade_lag = time_diffs.loc[closest_idx]
        
        # Affected frames
        affected_frames = nearby_cascades['frame'].unique().tolist()
        
        return {
            'triggered_cascades': int(triggered_cascades),
            'avg_cascade_strength': float(avg_strength),
            'cascade_lag_days': int(cascade_lag),
            'affected_frames': affected_frames
        }
    
    def _analyze_trend_impact(self, event: pd.Series, 
                            trend_results: pd.DataFrame) -> Dict:
        """Analyze impact on trends."""
        event_date = self._ensure_timestamp(event['peak_date'])
        
        # Find trend changes near event
        window_start = event_date - timedelta(days=14)  # 2 weeks before
        window_end = event_date + timedelta(days=self.post_event_window)
        
        # Look for trend starts in window
        trend_changes = trend_results[
            (trend_results['date'] >= window_start) &
            (trend_results['date'] <= window_end) &
            (trend_results['type'].str.contains('start'))
        ]
        
        if len(trend_changes) == 0:
            return {
                'triggered_trends': 0,
                'trend_scales': [],
                'trend_directions': [],
                'trend_lag_days': None
            }
        
        # Analyze by scale
        scale_counts = trend_changes['scale'].value_counts().to_dict()
        direction_counts = trend_changes['direction'].value_counts().to_dict()
        
        # Find closest trend change
        time_diffs = (trend_changes['date'] - event_date).dt.days
        closest_idx = time_diffs.abs().idxmin()
        trend_lag = time_diffs.loc[closest_idx]
        
        return {
            'triggered_trends': int(len(trend_changes)),
            'trend_scales': list(scale_counts.keys()),
            'trend_directions': list(direction_counts.keys()),
            'trend_lag_days': int(trend_lag),
            'short_term_trends': scale_counts.get('short', 0),
            'medium_term_trends': scale_counts.get('medium', 0),
            'long_term_trends': scale_counts.get('long', 0)
        }
    
    def _analyze_frame_shift(self, event: pd.Series, 
                           frame_data: pd.DataFrame) -> Dict:
        """Analyze frame usage shifts after event."""
        event_date = self._ensure_timestamp(event['peak_date'])
        
        # Get frame proportions before and after
        pre_start = event_date - timedelta(days=self.pre_event_window)
        pre_end = event_date - timedelta(days=1)
        post_start = event_date + timedelta(days=1)
        post_end = event_date + timedelta(days=self.post_event_window)
        
        pre_data = frame_data[
            (frame_data['date'] >= pre_start) & 
            (frame_data['date'] <= pre_end)
        ]
        
        post_data = frame_data[
            (frame_data['date'] >= post_start) & 
            (frame_data['date'] <= post_end)
        ]
        
        if len(pre_data) == 0 or len(post_data) == 0:
            return {'significant_shifts': [], 'max_shift_magnitude': 0}
        
        # Calculate shifts for each frame
        frame_cols = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
        shifts = {}
        
        for frame in frame_cols:
            if frame in pre_data.columns and frame in post_data.columns:
                pre_mean = pre_data[frame].mean()
                post_mean = post_data[frame].mean()
                
                if pre_mean > 0:
                    relative_change = (post_mean - pre_mean) / pre_mean
                    
                    # Test significance
                    _, p_value = stats.mannwhitneyu(
                        pre_data[frame].dropna(),
                        post_data[frame].dropna(),
                        alternative='two-sided'
                    )
                    
                    if p_value < 0.05 and abs(relative_change) > 0.1:
                        shifts[frame] = {
                            'change': float(relative_change),
                            'p_value': float(p_value),
                            'direction': 'increase' if relative_change > 0 else 'decrease'
                        }
        
        return {
            'significant_shifts': list(shifts.keys()),
            'max_shift_magnitude': float(max(abs(s['change']) for s in shifts.values())) if shifts else 0,
            'shift_details': shifts
        }
    
    def _calculate_overall_impact(self, cascade_impact: Dict, 
                                trend_impact: Dict, 
                                frame_shift: Dict) -> float:
        """Calculate overall impact score."""
        # Normalize components
        cascade_score = min(cascade_impact['triggered_cascades'] / 5, 1) * \
                       cascade_impact['avg_cascade_strength']
        
        trend_score = min(trend_impact['triggered_trends'] / 10, 1)
        
        shift_score = min(len(frame_shift['significant_shifts']) / 3, 1) * \
                     frame_shift['max_shift_magnitude']
        
        # Weight components
        weights = {'cascade': 0.4, 'trend': 0.3, 'shift': 0.3}
        
        overall = (
            weights['cascade'] * cascade_score +
            weights['trend'] * trend_score +
            weights['shift'] * shift_score
        )
        
        return float(overall)


class EventFramePatternAnalyzer:
    """Analyzes patterns between event types and frame responses."""
    
    @staticmethod
    def analyze_event_frame_relationships(impact_results: pd.DataFrame) -> Dict:
        """
        Analyze relationships between event types and frame responses.
        
        Args:
            impact_results: DataFrame with event impact analysis
            
        Returns:
            Dictionary with relationship analysis
        """
        relationships = {
            'event_frame_affinity': {},
            'event_effectiveness': {},
            'frame_responsiveness': {},
            'temporal_patterns': {}
        }
        
        # Calculate event-frame affinity matrix
        event_types = impact_results['event_type'].unique()
        frames = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
        
        affinity_matrix = pd.DataFrame(index=event_types, columns=frames, dtype=float)
        
        for event_type in event_types:
            event_data = impact_results[impact_results['event_type'] == event_type]
            
            for frame in frames:
                # Count how often this frame responds to this event type
                frame_responses = 0
                total_events = len(event_data)
                
                for _, event in event_data.iterrows():
                    if frame in event.get('cascade_impact', {}).get('affected_frames', []):
                        frame_responses += 1
                    if frame in event.get('frame_shift', {}).get('significant_shifts', []):
                        frame_responses += 1
                
                affinity = frame_responses / (2 * total_events) if total_events > 0 else 0
                affinity_matrix.loc[event_type, frame] = affinity
        
        relationships['event_frame_affinity'] = affinity_matrix.to_dict()
        
        # Calculate event effectiveness
        for event_type in event_types:
            event_data = impact_results[impact_results['event_type'] == event_type]
            relationships['event_effectiveness'][event_type] = {
                'avg_impact_score': float(event_data['overall_impact_score'].mean()),
                'cascade_trigger_rate': float((event_data['cascade_impact'].apply(
                    lambda x: x.get('triggered_cascades', 0) > 0).mean())),
                'trend_trigger_rate': float((event_data['trend_impact'].apply(
                    lambda x: x.get('triggered_trends', 0) > 0).mean())),
                'n_events': int(len(event_data))
            }
        
        # Calculate frame responsiveness
        for frame in frames:
            frame_cascades = sum(
                frame in event.get('cascade_impact', {}).get('affected_frames', [])
                for _, event in impact_results.iterrows()
            )
            frame_shifts = sum(
                frame in event.get('frame_shift', {}).get('significant_shifts', [])
                for _, event in impact_results.iterrows()
            )
            
            relationships['frame_responsiveness'][frame] = {
                'cascade_responses': int(frame_cascades),
                'shift_responses': int(frame_shifts),
                'total_responses': int(frame_cascades + frame_shifts),
                'responsiveness_score': float((frame_cascades + frame_shifts) / (2 * len(impact_results)))
            }
        
        # Temporal patterns
        cascade_lags = []
        trend_lags = []
        
        for _, event in impact_results.iterrows():
            cascade_lag = event.get('cascade_impact', {}).get('cascade_lag_days')
            if cascade_lag is not None:
                cascade_lags.append(cascade_lag)
            
            trend_lag = event.get('trend_impact', {}).get('trend_lag_days')
            if trend_lag is not None:
                trend_lags.append(trend_lag)
        
        relationships['temporal_patterns'] = {
            'avg_cascade_lag': float(np.mean(cascade_lags)) if cascade_lags else 0,
            'avg_trend_lag': float(np.mean(trend_lags)) if trend_lags else 0,
            'lag_correlation': float(np.corrcoef(cascade_lags, trend_lags)[0, 1]) 
                               if len(cascade_lags) > 1 and len(trend_lags) > 1 
                               and len(cascade_lags) == len(trend_lags) else 0
        }
        
        return relationships
    
    @staticmethod
    def identify_focusing_events(impact_results: pd.DataFrame,
                               impact_threshold: float = 0.7) -> pd.DataFrame:
        """
        Identify focusing events based on their impact.
        
        Args:
            impact_results: DataFrame with event impact analysis
            impact_threshold: Minimum impact score for focusing event
            
        Returns:
            DataFrame of focusing events
        """
        # Filter high-impact events
        focusing_events = impact_results[
            impact_results['overall_impact_score'] >= impact_threshold
        ].copy()
        
        # Add additional criteria
        focusing_events = focusing_events[
            (focusing_events['cascade_impact'].apply(
                lambda x: x.get('triggered_cascades', 0) > 0)) |
            (focusing_events['trend_impact'].apply(
                lambda x: x.get('triggered_trends', 0) > 0))
        ]
        
        # Calculate focusing strength
        focusing_events['focusing_strength'] = (
            focusing_events['overall_impact_score'] * 
            focusing_events['cascade_impact'].apply(
                lambda x: x.get('avg_cascade_strength', 0)
            )
        )
        
        # Sort by focusing strength
        focusing_events = focusing_events.sort_values(
            'focusing_strength', ascending=False
        )
        
        return focusing_events