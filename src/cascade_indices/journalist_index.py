"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
journalist_index.py

MAIN OBJECTIVE:
---------------
This script calculates journalist-related metrics for cascade detection,
analyzing adoption patterns, influence networks, and cross-media coordination.

Dependencies:
-------------
- pandas
- numpy
- networkx
- scipy
- sklearn
- collections
- base_index module

MAIN FEATURES:
--------------
1) Journalist adoption pattern analysis
2) Influence network structure mapping
3) Early adopter identification
4) Source proportion analysis
5) Cross-media journalist coordination detection

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import Counter, defaultdict
import networkx as nx
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from .base_index import BaseCascadeIndex, IndexResult


class JournalistIndex(BaseCascadeIndex):
    """
    Journalist adoption and influence patterns sub-index.
    
    This index measures how journalists adopt and spread a frame,
    including network effects, influence patterns, and coordination.
    """
    
    def __init__(self, weight: float = 0.2):
        """Initialize journalist index."""
        super().__init__(name="journalist", weight=weight)
        self.min_journalists = 3  # Minimum journalists for valid cascade
        self.adoption_threshold = 0.1  # Minimum frame usage for adoption
        self.similarity_threshold = 0.7  # Threshold for journalist similarity
        
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate required columns for journalist analysis."""
        required_columns = ['date', 'author', 'media', 'doc_id', 'sentence_id']
        missing = [col for col in required_columns if col not in data.columns]
        
        if missing:
            return False, f"Missing required columns: {missing}"
        
        # Check for frame detection columns
        frame_cols = [col for col in data.columns if col.endswith('_Detection')]
        if not frame_cols:
            return False, "No frame detection columns found"
        
        return True, ""
    
    def calculate(self,
                  data: pd.DataFrame,
                  frame: str,
                  reference_date: pd.Timestamp,
                  window_days: int = 180,
                  **kwargs) -> IndexResult:
        """
        Calculate journalist sub-index for a cascade.
        
        Args:
            data: Complete frame detection data
            frame: Frame name
            reference_date: Reference date for cascade
            window_days: Analysis window size
            
        Returns:
            IndexResult with journalist metrics
        """
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            return IndexResult(
                score=0,
                sub_scores={},
                metadata={'error': error_msg},
                confidence=0,
                details=f"Data validation failed: {error_msg}"
            )
        
        # Get window data
        window_data = self.get_window_data(data, reference_date, window_days)
        
        if len(window_data) == 0:
            return IndexResult(
                score=0,
                sub_scores={},
                metadata={'error': 'No data in window'},
                confidence=0,
                details="No data available in analysis window"
            )
        
        # Ensure frame column exists
        frame_col = f"{frame}_Detection"
        if frame_col not in window_data.columns:
            return IndexResult(
                score=0,
                sub_scores={},
                metadata={'error': f'Frame column {frame_col} not found'},
                confidence=0,
                details=f"Frame {frame} not found in data"
            )
        
        # Calculate sub-metrics
        adoption_metrics = self._calculate_adoption_pattern(window_data, frame_col, reference_date)
        network_metrics = self._calculate_influence_network(window_data, frame_col)
        diversity_metrics = self._calculate_diversity_metrics(window_data, frame_col)
        coordination_metrics = self._calculate_coordination_metrics(window_data, frame_col)
        proportion_metrics = self._calculate_proportion_metrics(window_data, frame_col)
        
        # Combine sub-scores
        sub_scores = {
            'adoption_rate': adoption_metrics['score'],
            'early_adopters': adoption_metrics['early_adopter_score'],
            'network_centrality': network_metrics['centrality_score'],
            'influence_spread': network_metrics['influence_score'],
            'diversity': diversity_metrics['diversity_score'],
            'cross_media': diversity_metrics['cross_media_score'],
            'coordination': coordination_metrics['sync_score'],
            'consistency': coordination_metrics['consistency_score'],
            'proportion_change': proportion_metrics['change_score'],
            'proportion_convergence': proportion_metrics['convergence_score']
        }
        
        # Calculate overall score (weighted average)
        weights = {
            'adoption_rate': 0.15,
            'early_adopters': 0.10,
            'network_centrality': 0.15,
            'influence_spread': 0.10,
            'diversity': 0.10,
            'cross_media': 0.10,
            'coordination': 0.10,
            'consistency': 0.05,
            'proportion_change': 0.10,
            'proportion_convergence': 0.05
        }
        
        overall_score = sum(score * weights[key] for key, score in sub_scores.items())
        
        # Compile metadata
        metadata = {
            'n_journalists': adoption_metrics['n_journalists'],
            'n_adopters': adoption_metrics['n_adopters'],
            'adoption_velocity': adoption_metrics['velocity'],
            'top_journalists': network_metrics['top_journalists'][:10],
            'early_adopters': adoption_metrics['early_adopters'][:10],
            'network_density': network_metrics['density'],
            'avg_clustering': network_metrics['clustering'],
            'media_diversity': diversity_metrics['n_media'],
            'coordination_score': coordination_metrics['coordination_score'],
            'mean_proportion': proportion_metrics['mean_proportion'],
            'proportion_trend': proportion_metrics['trend']
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            adoption_metrics['n_journalists'],
            self.min_journalists
        )
        
        # Generate details
        details = self._generate_details(sub_scores, metadata)
        
        return IndexResult(
            score=overall_score,
            sub_scores=sub_scores,
            metadata=metadata,
            confidence=confidence,
            details=details
        )
    
    def _calculate_adoption_pattern(self, 
                                   data: pd.DataFrame, 
                                   frame_col: str,
                                   reference_date: pd.Timestamp) -> Dict[str, Any]:
        """Calculate journalist adoption patterns."""
        # Convert frame column to numeric
        data[frame_col] = pd.to_numeric(data[frame_col], errors='coerce').fillna(0)
        
        # Group by author (journalist) and date
        journalist_daily = data.groupby(['author', 'date'])[frame_col].agg(['sum', 'count'])
        journalist_daily['proportion'] = journalist_daily['sum'] / journalist_daily['count'].clip(lower=1)
        
        # Identify adopters (journalists using the frame above threshold)
        adopters = journalist_daily[journalist_daily['proportion'] >= self.adoption_threshold]
        
        # Get unique journalists
        all_journalists = data['author'].nunique()
        adopting_journalists = adopters.index.get_level_values('author').nunique()
        
        # Calculate adoption rate
        adoption_rate = adopting_journalists / max(all_journalists, 1)
        
        # Identify early adopters (first quartile by date)
        if len(adopters) > 0:
            first_adoption = adopters.reset_index().groupby('author')['date'].min()
            date_threshold = first_adoption.quantile(0.25)
            early_adopters = first_adoption[first_adoption <= date_threshold]
            early_adopter_list = [(j, d) for j, d in early_adopters.items()]
            early_adopter_score = len(early_adopters) / max(len(first_adoption), 1)
        else:
            early_adopter_list = []
            early_adopter_score = 0
        
        # Calculate adoption velocity (rate of new adopters over time)
        if len(adopters) > 0:
            adoption_timeline = adopters.reset_index().groupby('date')['author'].nunique()
            adoption_cumsum = adoption_timeline.cumsum()
            
            # Calculate velocity as the slope of cumulative adoption
            if len(adoption_cumsum) > 1:
                x = np.arange(len(adoption_cumsum))
                y = adoption_cumsum.values
                velocity = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                velocity = velocity / max(all_journalists, 1)  # Normalize
            else:
                velocity = 0
        else:
            velocity = 0
        
        return {
            'score': self.apply_sigmoid_transformation(adoption_rate),
            'early_adopter_score': early_adopter_score,
            'n_journalists': all_journalists,
            'n_adopters': adopting_journalists,
            'early_adopters': early_adopter_list,
            'velocity': velocity
        }
    
    def _calculate_influence_network(self, 
                                    data: pd.DataFrame,
                                    frame_col: str) -> Dict[str, Any]:
        """Calculate journalist influence network metrics."""
        # Convert frame column to numeric
        data[frame_col] = pd.to_numeric(data[frame_col], errors='coerce').fillna(0)
        
        # Create journalist co-occurrence network
        G = nx.Graph()
        
        # Group by article to find co-occurring journalists
        article_groups = data.groupby('doc_id')['author'].apply(list)
        
        for journalists in article_groups:
            unique_journalists = list(set(journalists))
            # Add edges between co-occurring journalists
            for i, j1 in enumerate(unique_journalists):
                for j2 in unique_journalists[i+1:]:
                    if G.has_edge(j1, j2):
                        G[j1][j2]['weight'] += 1
                    else:
                        G.add_edge(j1, j2, weight=1)
        
        if len(G) == 0:
            return {
                'centrality_score': 0,
                'influence_score': 0,
                'top_journalists': [],
                'density': 0,
                'clustering': 0
            }
        
        # Calculate network metrics
        density = nx.density(G) if len(G) > 1 else 0
        
        # Calculate centrality for journalists using the frame
        frame_users = data[data[frame_col] > 0]['author'].unique()
        
        if len(frame_users) > 0:
            # Degree centrality
            degree_cent = nx.degree_centrality(G)
            frame_centralities = {j: degree_cent.get(j, 0) for j in frame_users}
            
            # Betweenness centrality (for influence paths)
            if len(G) > 2:
                between_cent = nx.betweenness_centrality(G, weight='weight')
                frame_betweenness = {j: between_cent.get(j, 0) for j in frame_users}
            else:
                frame_betweenness = frame_centralities
            
            # Average centrality scores
            avg_centrality = np.mean(list(frame_centralities.values())) if frame_centralities else 0
            avg_betweenness = np.mean(list(frame_betweenness.values())) if frame_betweenness else 0
            
            # Top journalists by centrality
            top_journalists = sorted(frame_centralities.items(), key=lambda x: x[1], reverse=True)
        else:
            avg_centrality = 0
            avg_betweenness = 0
            top_journalists = []
        
        # Calculate clustering coefficient
        clustering = nx.average_clustering(G, weight='weight') if len(G) > 2 else 0
        
        return {
            'centrality_score': self.normalize_score(avg_centrality),
            'influence_score': self.normalize_score(avg_betweenness),
            'top_journalists': top_journalists,
            'density': density,
            'clustering': clustering
        }
    
    def _calculate_diversity_metrics(self,
                                    data: pd.DataFrame,
                                    frame_col: str) -> Dict[str, Any]:
        """Calculate diversity of journalists and media outlets."""
        # Convert frame column to numeric
        data[frame_col] = pd.to_numeric(data[frame_col], errors='coerce').fillna(0)
        
        # Filter for frame users
        frame_data = data[data[frame_col] > 0]
        
        if len(frame_data) == 0:
            return {
                'diversity_score': 0,
                'cross_media_score': 0,
                'n_media': 0
            }
        
        # Calculate journalist diversity (Shannon entropy)
        journalist_counts = frame_data['author'].value_counts()
        journalist_probs = journalist_counts / journalist_counts.sum()
        journalist_entropy = entropy(journalist_probs)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(journalist_counts)) if len(journalist_counts) > 1 else 1
        diversity_score = journalist_entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate cross-media participation
        media_journalists = frame_data.groupby('media')['author'].nunique()
        n_media = len(media_journalists)
        
        # Check for journalists working across multiple media
        journalist_media = frame_data.groupby('author')['media'].nunique()
        cross_media_journalists = (journalist_media > 1).sum()
        cross_media_score = cross_media_journalists / len(journalist_media) if len(journalist_media) > 0 else 0
        
        return {
            'diversity_score': diversity_score,
            'cross_media_score': cross_media_score,
            'n_media': n_media
        }
    
    def _calculate_coordination_metrics(self,
                                       data: pd.DataFrame,
                                       frame_col: str) -> Dict[str, Any]:
        """Calculate coordination among journalists."""
        # Convert frame column to numeric
        data[frame_col] = pd.to_numeric(data[frame_col], errors='coerce').fillna(0)
        
        # Create time series for each journalist
        journalist_series = {}
        for journalist in data['author'].unique():
            j_data = data[data['author'] == journalist]
            daily = j_data.groupby('date')[frame_col].mean()
            if len(daily) > 0:
                journalist_series[journalist] = daily
        
        if len(journalist_series) < 2:
            return {
                'sync_score': 0,
                'consistency_score': 0,
                'coordination_score': 0
            }
        
        # Calculate pairwise correlations
        correlations = []
        journalists = list(journalist_series.keys())
        
        for i, j1 in enumerate(journalists):
            for j2 in journalists[i+1:]:
                # Find overlapping dates
                dates1 = set(journalist_series[j1].index)
                dates2 = set(journalist_series[j2].index)
                common_dates = sorted(dates1 & dates2)
                
                if len(common_dates) >= 3:  # Need at least 3 points for correlation
                    s1 = journalist_series[j1].loc[common_dates].values
                    s2 = journalist_series[j2].loc[common_dates].values
                    
                    if np.std(s1) > 0 and np.std(s2) > 0:
                        corr = np.corrcoef(s1, s2)[0, 1]
                        correlations.append(corr)
        
        if correlations:
            # Average correlation as synchronization score
            sync_score = np.mean(correlations)
            sync_score = (sync_score + 1) / 2  # Normalize to [0, 1]
            
            # Consistency: low variance in correlations means consistent coordination
            consistency = 1 - np.std(correlations) if len(correlations) > 1 else 1
            
            # Overall coordination score
            coordination = sync_score * consistency
        else:
            sync_score = 0
            consistency = 0
            coordination = 0
        
        return {
            'sync_score': sync_score,
            'consistency_score': consistency,
            'coordination_score': coordination
        }
    
    def _calculate_proportion_metrics(self,
                                     data: pd.DataFrame,
                                     frame_col: str) -> Dict[str, Any]:
        """Calculate frame proportion metrics for journalists."""
        # Convert frame column to numeric
        data[frame_col] = pd.to_numeric(data[frame_col], errors='coerce').fillna(0)
        
        # Calculate proportions by journalist and date
        daily_props = data.groupby(['date', 'author']).apply(
            lambda x: x[frame_col].sum() / len(x) if len(x) > 0 else 0
        ).reset_index(name='proportion')
        
        if len(daily_props) == 0:
            return {
                'change_score': 0,
                'convergence_score': 0,
                'mean_proportion': 0,
                'trend': 0
            }
        
        # Calculate overall trend
        date_means = daily_props.groupby('date')['proportion'].mean()
        if len(date_means) > 1:
            trend = self.calculate_trend(date_means, method='robust')
        else:
            trend = 0
        
        # Calculate proportion change (EMD between early and late periods)
        mid_point = len(date_means) // 2
        if mid_point > 0:
            early_dates = date_means.index[:mid_point]
            late_dates = date_means.index[mid_point:]
            
            early_props = daily_props[daily_props['date'].isin(early_dates)]['proportion'].values
            late_props = daily_props[daily_props['date'].isin(late_dates)]['proportion'].values
            
            if len(early_props) > 0 and len(late_props) > 0:
                # Use Earth Mover's Distance
                change_score = wasserstein_distance(early_props, late_props)
                change_score = self.normalize_score(change_score, 0, 1)
            else:
                change_score = 0
        else:
            change_score = 0
        
        # Calculate convergence (decreasing variance over time)
        window = max(3, len(date_means) // 5)
        rolling_std = date_means.rolling(window).std()
        if len(rolling_std.dropna()) > 1:
            convergence_trend = self.calculate_trend(rolling_std.dropna(), method='linear')
            convergence_score = 1 / (1 + np.exp(convergence_trend * 10))  # Negative trend = high score
        else:
            convergence_score = 0.5
        
        return {
            'change_score': change_score,
            'convergence_score': convergence_score,
            'mean_proportion': float(date_means.mean()),
            'trend': float(trend)
        }
    
    def _generate_details(self, sub_scores: Dict[str, float], metadata: Dict[str, Any]) -> str:
        """Generate human-readable details about the journalist index."""
        details = []
        
        # Overall assessment
        score = sum(sub_scores.values()) / len(sub_scores) if sub_scores else 0
        if score > 0.7:
            details.append("Strong journalist cascade detected")
        elif score > 0.4:
            details.append("Moderate journalist adoption observed")
        else:
            details.append("Weak journalist participation")
        
        # Key metrics
        details.append(f"Journalists involved: {metadata.get('n_journalists', 0)}")
        details.append(f"Active adopters: {metadata.get('n_adopters', 0)}")
        
        # Network characteristics
        if metadata.get('network_density', 0) > 0.3:
            details.append("High network connectivity among journalists")
        
        # Coordination
        if metadata.get('coordination_score', 0) > 0.6:
            details.append("Coordinated adoption pattern detected")
        
        # Trend
        trend = metadata.get('proportion_trend', 0)
        if trend > 0.1:
            details.append("Increasing frame adoption over time")
        elif trend < -0.1:
            details.append("Decreasing frame adoption over time")
        
        return " | ".join(details)