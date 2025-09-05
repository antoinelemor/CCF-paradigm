"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
cascade_analysis_enhanced.py

MAIN OBJECTIVE:
---------------
Enhanced version of cascade_analysis.py with major improvements:
1. Advanced network analysis with source/messenger proportion tracking
2. Improved sequential adoption using EMD and proportion changes without thresholds
3. Adaptive correlation thresholds with co-occurrence and temporal proximity
4. Integration of NER entities for epistemic authority detection
5. Comprehensive scoring with multi-index cascade indicators

Dependencies:
-------------
- pandas
- numpy
- scipy (including wasserstein_distance)
- networkx
- sklearn
- rapidfuzz
- json
- All original dependencies

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from scipy import stats, signal, interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
import networkx as nx
from rapidfuzz import fuzz, process
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import logging
import json
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')

# Set up logger
logger = logging.getLogger(__name__)

def shannon_entropy(proportions: np.ndarray) -> float:
    """Calculate Shannon entropy for diversity measurement."""
    p = proportions[proportions > 0]
    if len(p) == 0:
        return 0
    p = p / p.sum()
    return -np.sum(p * np.log(p))


def handle_outliers_iqr(data: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """Handle outliers using IQR method."""
    if len(data) == 0:
        return data
        
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    data_capped = np.clip(data, lower_bound, upper_bound)
    
    return data_capped


def apply_log_transformation(data: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Apply log transformation to handle skewed distributions."""
    data_positive = np.maximum(data, 0)
    
    if np.max(data_positive) < 1:
        transformed = np.log1p(data_positive)
    else:
        transformed = np.log(data_positive + epsilon)
    
    return transformed


def extract_entities_from_ner(ner_json: str) -> Dict[str, List[str]]:
    """
    Extract entities from NER JSON string.
    
    Args:
        ner_json: JSON string with NER entities
        
    Returns:
        Dictionary with PER and ORG entities
    """
    try:
        if pd.isna(ner_json) or ner_json == '' or ner_json == 'null':
            return {'PER': [], 'ORG': []}
        
        entities = json.loads(ner_json)
        return {
            'PER': entities.get('PER', []),
            'ORG': entities.get('ORG', [])
        }
    except (json.JSONDecodeError, TypeError):
        return {'PER': [], 'ORG': []}


def calculate_entity_similarity(entities1: List[str], entities2: List[str], 
                               threshold: float = 85) -> float:
    """
    Calculate similarity between two lists of entities using fuzzy matching.
    
    Args:
        entities1: First list of entities
        entities2: Second list of entities
        threshold: Fuzzy matching threshold
        
    Returns:
        Similarity score [0, 1]
    """
    if not entities1 or not entities2:
        return 0
    
    matches = 0
    for e1 in entities1:
        for e2 in entities2:
            if fuzz.ratio(e1.lower(), e2.lower()) >= threshold:
                matches += 1
                break
    
    return matches / max(len(entities1), len(entities2))


@dataclass
class EnhancedCascadeMetrics:
    """Enhanced metrics with comprehensive source and network analysis."""
    # Temporal metrics
    onset_date: pd.Timestamp
    inflection_date: pd.Timestamp
    peak_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int
    duration_days_transformed: float
    
    # Velocity metrics
    initial_velocity: float
    initial_velocity_robust: float
    peak_velocity: float
    acceleration: float
    momentum: float
    
    # Network metrics
    network_density: float
    network_density_transformed: float
    clustering_coefficient: float
    path_length: float
    correlation_density: float
    correlation_density_transformed: float
    mean_correlation: float
    
    # Enhanced network metrics
    co_occurrence_density: float
    temporal_proximity_score: float
    adaptive_threshold: float
    
    # Influencer metrics
    top_journalists: List[Tuple[str, float]]
    top_media: List[Tuple[str, float]]
    early_adopters: List[Tuple[str, pd.Timestamp]]
    
    # Enhanced source metrics
    source_diversity: float
    dominant_sources: Dict[str, int]
    source_concentration: float
    source_convergence: float
    source_proportions_by_article: Dict[str, float]  # Average proportions
    source_network_density: float
    epistemic_authorities: List[Tuple[str, float]]  # From NER analysis
    
    # Entity-based metrics
    entity_convergence: float
    shared_entities: Dict[str, int]
    entity_network: nx.Graph
    
    # Virality metrics
    homogenization_score: float
    homogenization_shannon: float
    entropy_trend: float
    entropy_trend_robust: float
    emotional_intensity: float
    viral_coefficient: float
    
    # EMD-based adoption metrics
    adoption_emd_score: float
    proportion_change_rate: float
    
    # Statistical validation
    statistical_significance: Dict[str, float]
    consensus_significant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'onset_date': self.onset_date,
            'inflection_date': self.inflection_date,
            'peak_date': self.peak_date,
            'end_date': self.end_date,
            'duration_days': self.duration_days,
            'duration_days_transformed': self.duration_days_transformed,
            'initial_velocity': self.initial_velocity,
            'initial_velocity_robust': self.initial_velocity_robust,
            'peak_velocity': self.peak_velocity,
            'acceleration': self.acceleration,
            'momentum': self.momentum,
            'network_density': self.network_density,
            'network_density_transformed': self.network_density_transformed,
            'clustering_coefficient': self.clustering_coefficient,
            'path_length': self.path_length,
            'correlation_density': self.correlation_density,
            'correlation_density_transformed': self.correlation_density_transformed,
            'mean_correlation': self.mean_correlation,
            'co_occurrence_density': self.co_occurrence_density,
            'temporal_proximity_score': self.temporal_proximity_score,
            'adaptive_threshold': self.adaptive_threshold,
            'n_top_journalists': len(self.top_journalists),
            'n_top_media': len(self.top_media),
            'n_early_adopters': len(self.early_adopters),
            'source_diversity': self.source_diversity,
            'source_concentration': self.source_concentration,
            'source_convergence': self.source_convergence,
            'source_network_density': self.source_network_density,
            'n_epistemic_authorities': len(self.epistemic_authorities),
            'entity_convergence': self.entity_convergence,
            'n_shared_entities': len(self.shared_entities),
            'homogenization_score': self.homogenization_score,
            'homogenization_shannon': self.homogenization_shannon,
            'entropy_trend': self.entropy_trend,
            'entropy_trend_robust': self.entropy_trend_robust,
            'emotional_intensity': self.emotional_intensity,
            'viral_coefficient': self.viral_coefficient,
            'adoption_emd_score': self.adoption_emd_score,
            'proportion_change_rate': self.proportion_change_rate,
            'consensus_significant': self.consensus_significant
        }


@dataclass
class EnhancedMediaCascade:
    """Enhanced media cascade with comprehensive metrics."""
    frame: str
    reference_date: pd.Timestamp
    cascade_type: str
    
    # Core metrics
    metrics: EnhancedCascadeMetrics
    
    # Component scores (each 0-1, weighted 20% each)
    journalist_score: float
    media_score: float
    intensity_score: float
    network_score: float
    virality_score: float
    
    # Sub-indices for multi-level scoring
    sub_indices: Dict[str, Dict[str, float]]
    
    # Composite score
    total_score: float
    cascade_strength: str
    
    # Raw data for visualization
    temporal_data: pd.DataFrame
    network_graph: nx.DiGraph
    correlation_network: nx.DiGraph
    adoption_sequence: nx.DiGraph
    source_citation_network: nx.Graph
    entity_network: nx.Graph
    
    # Metadata
    detection_method: str
    confidence: float
    notes: str = ""


class EnhancedCascadeDetector:
    """Enhanced cascade detection with comprehensive improvements."""
    
    def __init__(self, similarity_threshold: float = 85):
        """Initialize enhanced detector."""
        self.similarity_threshold = similarity_threshold
        
        # Messenger type mapping
        self.messenger_types = {
            'Messenger_1_SUB': 'health_expertise',
            'Messenger_2_SUB': 'economic_expertise',
            'Messenger_3_SUB': 'security_expertise',
            'Messenger_4_SUB': 'law_expertise',
            'Messenger_5_SUB': 'culture_expertise',
            'Messenger_6_SUB': 'hard_science',
            'Messenger_7_SUB': 'social_science',
            'Messenger_8_SUB': 'activist',
            'Messenger_9_SUB': 'public_official'
        }
    
    def analyze_source_proportions_enhanced(self, df: pd.DataFrame, frame: str,
                                           onset: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
        """
        Enhanced source analysis with fine-grained proportion tracking.
        
        Calculates average proportion of sentences per article per source,
        analyzes correlations between journalists/media citing same sources.
        """
        frame_col = f"{frame}_Detection"
        cascade_data = df[(df['date'] >= onset) & (df['date'] <= end)].copy()
        
        if frame_col in cascade_data.columns:
            cascade_data[frame_col] = pd.to_numeric(cascade_data[frame_col], errors='coerce').fillna(0)
            cascade_data = cascade_data[cascade_data[frame_col] == 1]
        
        # Calculate proportions for each messenger type per article
        article_source_profiles = []
        
        for doc_id in cascade_data['doc_id'].unique():
            article_data = cascade_data[cascade_data['doc_id'] == doc_id]
            total_sentences = len(article_data)
            
            if total_sentences == 0:
                continue
            
            # Get article metadata
            author = article_data['author'].iloc[0]
            media = article_data['media'].iloc[0]
            date = article_data['date'].iloc[0]
            
            # Calculate source proportions
            source_props = {}
            for msg_col, msg_type in self.messenger_types.items():
                if msg_col in article_data.columns:
                    article_data[msg_col] = pd.to_numeric(article_data[msg_col], errors='coerce').fillna(0)
                    prop = article_data[msg_col].sum() / total_sentences
                    source_props[msg_type] = prop
            
            # Extract NER entities if available
            entities = {'PER': [], 'ORG': []}
            if 'ner_entities' in article_data.columns:
                for _, row in article_data.iterrows():
                    row_entities = extract_entities_from_ner(row.get('ner_entities', ''))
                    entities['PER'].extend(row_entities['PER'])
                    entities['ORG'].extend(row_entities['ORG'])
            
            article_source_profiles.append({
                'doc_id': doc_id,
                'author': author,
                'media': media,
                'date': date,
                **source_props,
                'entities_PER': list(set(entities['PER'])),
                'entities_ORG': list(set(entities['ORG']))
            })
        
        if not article_source_profiles:
            return self._empty_source_analysis()
        
        profiles_df = pd.DataFrame(article_source_profiles)
        
        # Build source citation network
        source_network = self._build_source_citation_network(profiles_df)
        
        # Calculate epistemic authorities (most cited entities)
        epistemic_authorities = self._identify_epistemic_authorities(profiles_df)
        
        # Analyze correlation patterns
        source_cols = list(self.messenger_types.values())
        correlation_matrix = profiles_df[source_cols].corr()
        
        # Calculate network density of source citations
        if len(source_network) > 0:
            source_network_density = nx.density(source_network)
        else:
            source_network_density = 0
        
        # Track temporal evolution of source proportions
        temporal_evolution = self._analyze_temporal_source_evolution(profiles_df)
        
        return {
            'article_profiles': profiles_df,
            'source_network': source_network,
            'source_network_density': source_network_density,
            'epistemic_authorities': epistemic_authorities,
            'correlation_matrix': correlation_matrix,
            'mean_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'temporal_evolution': temporal_evolution,
            'source_proportions_mean': profiles_df[source_cols].mean().to_dict(),
            'source_proportions_std': profiles_df[source_cols].std().to_dict()
        }
    
    def _build_source_citation_network(self, profiles_df: pd.DataFrame) -> nx.Graph:
        """Build network of journalists/media based on shared source citations."""
        G = nx.Graph()
        
        # Add nodes for journalists and media
        for _, profile in profiles_df.iterrows():
            if pd.notna(profile['author']):
                G.add_node(f"J:{profile['author']}", type='journalist')
            if pd.notna(profile['media']):
                G.add_node(f"M:{profile['media']}", type='media')
        
        # Add edges based on shared source patterns
        source_cols = list(self.messenger_types.values())
        
        # Group by journalist
        for author1 in profiles_df['author'].dropna().unique():
            for author2 in profiles_df['author'].dropna().unique():
                if author1 >= author2:
                    continue
                
                # Get profiles for each journalist
                prof1 = profiles_df[profiles_df['author'] == author1][source_cols].mean()
                prof2 = profiles_df[profiles_df['author'] == author2][source_cols].mean()
                
                # Calculate similarity
                similarity = 1 - wasserstein_distance(prof1, prof2)
                
                if similarity > 0.7:  # High similarity threshold
                    G.add_edge(f"J:{author1}", f"J:{author2}", 
                             weight=similarity, type='source_similarity')
        
        # Similar for media outlets
        for media1 in profiles_df['media'].dropna().unique():
            for media2 in profiles_df['media'].dropna().unique():
                if media1 >= media2:
                    continue
                
                prof1 = profiles_df[profiles_df['media'] == media1][source_cols].mean()
                prof2 = profiles_df[profiles_df['media'] == media2][source_cols].mean()
                
                similarity = 1 - wasserstein_distance(prof1, prof2)
                
                if similarity > 0.7:
                    G.add_edge(f"M:{media1}", f"M:{media2}",
                             weight=similarity, type='source_similarity')
        
        return G
    
    def _identify_epistemic_authorities(self, profiles_df: pd.DataFrame) -> List[Tuple[str, float]]:
        """Identify epistemic authorities from NER entities."""
        entity_counts = Counter()
        
        for _, profile in profiles_df.iterrows():
            # Count persons
            for person in profile.get('entities_PER', []):
                entity_counts[f"PER:{person}"] += 1
            
            # Count organizations
            for org in profile.get('entities_ORG', []):
                entity_counts[f"ORG:{org}"] += 1
        
        # Calculate authority scores (frequency * diversity of sources)
        authority_scores = []
        
        for entity, count in entity_counts.most_common(20):
            # Find which journalists/media cite this entity
            citing_journalists = set()
            citing_media = set()
            
            entity_type, entity_name = entity.split(':', 1)
            
            for _, profile in profiles_df.iterrows():
                if entity_type == 'PER' and entity_name in profile.get('entities_PER', []):
                    if pd.notna(profile['author']):
                        citing_journalists.add(profile['author'])
                    if pd.notna(profile['media']):
                        citing_media.add(profile['media'])
                elif entity_type == 'ORG' and entity_name in profile.get('entities_ORG', []):
                    if pd.notna(profile['author']):
                        citing_journalists.add(profile['author'])
                    if pd.notna(profile['media']):
                        citing_media.add(profile['media'])
            
            # Authority score combines frequency and diversity
            diversity_score = len(citing_journalists) + len(citing_media)
            authority_score = count * np.log1p(diversity_score)
            
            authority_scores.append((entity, authority_score))
        
        # Sort by authority score
        authority_scores.sort(key=lambda x: x[1], reverse=True)
        
        return authority_scores[:10]  # Top 10 authorities
    
    def _analyze_temporal_source_evolution(self, profiles_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how source citation patterns evolve over time."""
        profiles_df['week'] = pd.to_datetime(profiles_df['date']).dt.to_period('W')
        
        source_cols = list(self.messenger_types.values())
        weekly_evolution = []
        
        for week in profiles_df['week'].unique():
            week_data = profiles_df[profiles_df['week'] == week]
            
            weekly_evolution.append({
                'week': week.to_timestamp(),
                'n_articles': len(week_data),
                'mean_proportions': week_data[source_cols].mean().to_dict(),
                'diversity': shannon_entropy(week_data[source_cols].mean().values)
            })
        
        return weekly_evolution
    
    def detect_adoption_with_emd(self, df: pd.DataFrame, frame: str,
                                onset: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
        """
        Detect adoption using Earth Mover's Distance on proportion distributions.
        No arbitrary thresholds - uses EMD to measure distribution changes.
        """
        # Calculate article proportions
        article_props = self.calculate_article_proportions(df, frame)
        
        # Filter to cascade window
        cascade_articles = article_props[
            (article_props['date'] >= onset) & 
            (article_props['date'] <= end)
        ].copy()
        
        if len(cascade_articles) < 2:
            return {'network': nx.DiGraph(), 'emd_scores': {}, 'adoption_rate': 0}
        
        # Sort by date
        cascade_articles = cascade_articles.sort_values('date')
        
        # Build adoption network based on EMD
        G = nx.DiGraph()
        emd_scores = {}
        
        # Track adoption patterns by journalist
        journalists = cascade_articles.groupby('author')
        
        for j_name, j_data in journalists:
            if pd.isna(j_name) or len(j_data) < 2:
                continue
            
            # Calculate proportion change over time
            j_data = j_data.sort_values('date')
            proportions = j_data['frame_proportion'].values
            
            # Calculate EMD between consecutive time points
            emd_changes = []
            for i in range(1, len(proportions)):
                # Create distributions for EMD
                dist1 = np.array([1 - proportions[i-1], proportions[i-1]])
                dist2 = np.array([1 - proportions[i], proportions[i]])
                
                emd = wasserstein_distance([0, 1], [0, 1], dist1, dist2)
                emd_changes.append(emd)
            
            if emd_changes:
                mean_emd = np.mean(emd_changes)
                max_emd = np.max(emd_changes)
                
                # Add node with EMD metrics
                G.add_node(f"J:{j_name}",
                         type='journalist',
                         mean_emd=mean_emd,
                         max_emd=max_emd,
                         first_date=j_data['date'].min(),
                         peak_proportion=proportions.max())
                
                emd_scores[j_name] = {
                    'mean_emd': mean_emd,
                    'max_emd': max_emd,
                    'adoption_strength': max_emd
                }
        
        # Similar for media
        media_groups = cascade_articles.groupby('media')
        
        for m_name, m_data in media_groups:
            if len(m_data) < 3:
                continue
            
            m_data = m_data.sort_values('date')
            proportions = m_data['frame_proportion'].values
            
            emd_changes = []
            for i in range(1, len(proportions)):
                dist1 = np.array([1 - proportions[i-1], proportions[i-1]])
                dist2 = np.array([1 - proportions[i], proportions[i]])
                emd = wasserstein_distance([0, 1], [0, 1], dist1, dist2)
                emd_changes.append(emd)
            
            if emd_changes:
                mean_emd = np.mean(emd_changes)
                max_emd = np.max(emd_changes)
                
                G.add_node(f"M:{m_name}",
                         type='media',
                         mean_emd=mean_emd,
                         max_emd=max_emd,
                         first_date=m_data['date'].min(),
                         peak_proportion=proportions.max())
        
        # Add edges based on temporal sequence and EMD similarity
        self._add_emd_based_edges(G, cascade_articles, emd_scores)
        
        # Calculate overall adoption rate
        if emd_scores:
            adoption_rate = np.mean([s['adoption_strength'] for s in emd_scores.values()])
        else:
            adoption_rate = 0
        
        return {
            'network': G,
            'emd_scores': emd_scores,
            'adoption_rate': adoption_rate,
            'n_adopters': len(G),
            'mean_emd': np.mean([s['mean_emd'] for s in emd_scores.values()]) if emd_scores else 0
        }
    
    def _add_emd_based_edges(self, G: nx.DiGraph, cascade_articles: pd.DataFrame, 
                            emd_scores: Dict) -> None:
        """Add edges to network based on EMD similarities and temporal patterns."""
        # Group by journalist pairs
        journalists = cascade_articles['author'].dropna().unique()
        
        for i, j1 in enumerate(journalists):
            if j1 not in emd_scores:
                continue
                
            for j2 in journalists[i+1:]:
                if j2 not in emd_scores:
                    continue
                
                # Check temporal relationship
                j1_data = cascade_articles[cascade_articles['author'] == j1]
                j2_data = cascade_articles[cascade_articles['author'] == j2]
                
                j1_start = j1_data['date'].min()
                j2_start = j2_data['date'].min()
                
                # Add edge from earlier to later adopter
                if j1_start < j2_start:
                    # Weight based on EMD similarity
                    weight = 1 / (1 + abs(emd_scores[j1]['mean_emd'] - emd_scores[j2]['mean_emd']))
                    G.add_edge(f"J:{j1}", f"J:{j2}", weight=weight, 
                             time_diff=(j2_start - j1_start).days)
                elif j2_start < j1_start:
                    weight = 1 / (1 + abs(emd_scores[j1]['mean_emd'] - emd_scores[j2]['mean_emd']))
                    G.add_edge(f"J:{j2}", f"J:{j1}", weight=weight,
                             time_diff=(j1_start - j2_start).days)
    
    def build_enhanced_correlation_network(self, df: pd.DataFrame, frame: str,
                                          onset: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
        """
        Build correlation network with adaptive thresholds, co-occurrence, and temporal proximity.
        """
        # Get article proportions
        article_props = self.calculate_article_proportions(df, frame)
        
        # Filter to cascade window
        cascade_articles = article_props[
            (article_props['date'] >= onset) & 
            (article_props['date'] <= end)
        ].copy()
        
        if len(cascade_articles) < 2:
            return {'network': nx.DiGraph(), 'adaptive_threshold': 0.3}
        
        # Calculate adaptive threshold based on data distribution
        all_correlations = []
        journalists = cascade_articles.groupby('author')
        
        # First pass: calculate all correlations to determine threshold
        for j1_name, j1_data in journalists:
            for j2_name, j2_data in journalists:
                if j1_name >= j2_name or pd.isna(j1_name) or pd.isna(j2_name):
                    continue
                
                if len(j1_data) < 3 or len(j2_data) < 3:
                    continue
                
                # Weekly correlation
                j1_weekly = j1_data.set_index('date').resample('W')['frame_proportion'].mean()
                j2_weekly = j2_data.set_index('date').resample('W')['frame_proportion'].mean()
                
                common_weeks = j1_weekly.index.intersection(j2_weekly.index)
                
                if len(common_weeks) >= 3:
                    corr = j1_weekly[common_weeks].corr(j2_weekly[common_weeks])
                    if not np.isnan(corr):
                        all_correlations.append(abs(corr))
        
        # Set adaptive threshold (75th percentile of correlations)
        if all_correlations:
            adaptive_threshold = np.percentile(all_correlations, 25)  # Lower percentile for less restrictive
        else:
            adaptive_threshold = 0.3
        
        # Build network with adaptive threshold
        G = nx.DiGraph()
        
        # Add correlation edges
        for j1_name, j1_data in journalists:
            if pd.isna(j1_name) or len(j1_data) < 3:
                continue
            
            G.add_node(f"J:{j1_name}", 
                      type='journalist',
                      mean_proportion=j1_data['frame_proportion'].mean(),
                      first_date=j1_data['date'].min())
            
            for j2_name, j2_data in journalists:
                if j1_name >= j2_name or pd.isna(j2_name) or len(j2_data) < 3:
                    continue
                
                # Calculate correlation
                j1_weekly = j1_data.set_index('date').resample('W')['frame_proportion'].mean()
                j2_weekly = j2_data.set_index('date').resample('W')['frame_proportion'].mean()
                
                common_weeks = j1_weekly.index.intersection(j2_weekly.index)
                
                if len(common_weeks) >= 3:
                    corr = j1_weekly[common_weeks].corr(j2_weekly[common_weeks])
                    
                    if not np.isnan(corr) and abs(corr) > adaptive_threshold:
                        # Determine direction based on who adopted first
                        if j1_data['date'].min() < j2_data['date'].min():
                            G.add_edge(f"J:{j1_name}", f"J:{j2_name}", 
                                     weight=abs(corr), correlation=corr, type='correlation')
                        else:
                            G.add_edge(f"J:{j2_name}", f"J:{j1_name}",
                                     weight=abs(corr), correlation=corr, type='correlation')
        
        # Add co-occurrence edges (same article)
        co_occurrence_pairs = self._find_co_occurrences(cascade_articles)
        
        for (j1, j2), count in co_occurrence_pairs.items():
            if f"J:{j1}" in G and f"J:{j2}" in G:
                # Add or update edge with co-occurrence weight
                if G.has_edge(f"J:{j1}", f"J:{j2}"):
                    G[f"J:{j1}"][f"J:{j2}"]['co_occurrence'] = count
                else:
                    G.add_edge(f"J:{j1}", f"J:{j2}", 
                             weight=count/10, co_occurrence=count, type='co_occurrence')
        
        # Add temporal proximity edges
        temporal_edges = self._add_temporal_proximity_edges(G, cascade_articles)
        
        # Calculate network metrics
        co_occurrence_density = len([e for e in G.edges(data=True) 
                                    if e[2].get('type') == 'co_occurrence']) / max(len(G), 1)
        
        temporal_proximity_score = len([e for e in G.edges(data=True) 
                                       if e[2].get('type') == 'temporal']) / max(len(G), 1)
        
        return {
            'network': G,
            'adaptive_threshold': adaptive_threshold,
            'co_occurrence_density': co_occurrence_density,
            'temporal_proximity_score': temporal_proximity_score,
            'n_correlation_edges': len([e for e in G.edges(data=True) 
                                       if e[2].get('type') == 'correlation']),
            'n_co_occurrence_edges': len([e for e in G.edges(data=True) 
                                        if e[2].get('type') == 'co_occurrence']),
            'n_temporal_edges': len([e for e in G.edges(data=True) 
                                   if e[2].get('type') == 'temporal'])
        }
    
    def _find_co_occurrences(self, cascade_articles: pd.DataFrame) -> Dict[Tuple[str, str], int]:
        """Find journalists who appear in the same articles."""
        co_occurrences = defaultdict(int)
        
        # Group by article
        for doc_id, article_data in cascade_articles.groupby('doc_id'):
            authors = article_data['author'].dropna().unique()
            
            # Count co-occurrences
            for i, a1 in enumerate(authors):
                for a2 in authors[i+1:]:
                    key = tuple(sorted([a1, a2]))
                    co_occurrences[key] += 1
        
        return dict(co_occurrences)
    
    def _add_temporal_proximity_edges(self, G: nx.DiGraph, 
                                     cascade_articles: pd.DataFrame) -> int:
        """Add edges based on temporal proximity."""
        edges_added = 0
        
        # Sort by date
        cascade_articles = cascade_articles.sort_values('date')
        
        # Group by date (daily)
        daily_groups = cascade_articles.groupby(cascade_articles['date'].dt.date)
        
        dates = sorted(daily_groups.groups.keys())
        
        for i, date1 in enumerate(dates[:-1]):
            date2 = dates[i+1]
            
            # Get journalists from consecutive days
            j1_set = set(daily_groups.get_group(date1)['author'].dropna())
            j2_set = set(daily_groups.get_group(date2)['author'].dropna())
            
            # Add edges between journalists from consecutive days
            for j1 in j1_set:
                for j2 in j2_set:
                    if j1 != j2 and f"J:{j1}" in G and f"J:{j2}" in G:
                        if not G.has_edge(f"J:{j1}", f"J:{j2}"):
                            G.add_edge(f"J:{j1}", f"J:{j2}",
                                     weight=0.5, type='temporal', days_apart=1)
                            edges_added += 1
        
        return edges_added
    
    def analyze_entity_networks(self, df: pd.DataFrame, frame: str,
                               onset: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
        """
        Analyze networks based on shared NER entities.
        """
        frame_col = f"{frame}_Detection"
        cascade_data = df[(df['date'] >= onset) & (df['date'] <= end)].copy()
        
        if frame_col in cascade_data.columns:
            cascade_data[frame_col] = pd.to_numeric(cascade_data[frame_col], errors='coerce').fillna(0)
            cascade_data = cascade_data[cascade_data[frame_col] == 1]
        
        if 'ner_entities' not in cascade_data.columns:
            return {'entity_network': nx.Graph(), 'entity_convergence': 0, 'shared_entities': {}}
        
        # Extract entities for each journalist/media
        entity_profiles = defaultdict(lambda: {'PER': [], 'ORG': []})
        
        for _, row in cascade_data.iterrows():
            entities = extract_entities_from_ner(row.get('ner_entities', ''))
            
            if pd.notna(row.get('author')):
                entity_profiles[f"J:{row['author']}"]['PER'].extend(entities['PER'])
                entity_profiles[f"J:{row['author']}"]['ORG'].extend(entities['ORG'])
            
            if pd.notna(row.get('media')):
                entity_profiles[f"M:{row['media']}"]['PER'].extend(entities['PER'])
                entity_profiles[f"M:{row['media']}"]['ORG'].extend(entities['ORG'])
        
        # Build entity network
        G = nx.Graph()
        
        # Add nodes
        for actor in entity_profiles:
            G.add_node(actor, type=actor.split(':')[0])
        
        # Add edges based on shared entities (95% similarity threshold)
        shared_entities = defaultdict(int)
        
        actors = list(entity_profiles.keys())
        for i, actor1 in enumerate(actors):
            for actor2 in actors[i+1:]:
                # Combine all entities for each actor
                entities1 = (entity_profiles[actor1]['PER'] + 
                            entity_profiles[actor1]['ORG'])
                entities2 = (entity_profiles[actor2]['PER'] + 
                            entity_profiles[actor2]['ORG'])
                
                if entities1 and entities2:
                    # Calculate entity similarity
                    similarity = calculate_entity_similarity(entities1, entities2, threshold=95)
                    
                    if similarity > 0:
                        G.add_edge(actor1, actor2, weight=similarity, 
                                 type='entity_similarity')
                        
                        # Track shared entities
                        for e1 in entities1:
                            for e2 in entities2:
                                if fuzz.ratio(e1.lower(), e2.lower()) >= 95:
                                    shared_entities[e1] += 1
        
        # Calculate convergence metric
        if len(G.edges()) > 0:
            entity_convergence = nx.density(G)
        else:
            entity_convergence = 0
        
        return {
            'entity_network': G,
            'entity_convergence': entity_convergence,
            'shared_entities': dict(shared_entities),
            'n_unique_entities': len(set().union(*[set(p['PER'] + p['ORG']) 
                                                   for p in entity_profiles.values()])),
            'entity_profiles': dict(entity_profiles)
        }
    
    def calculate_article_proportions(self, df: pd.DataFrame, frame: str) -> pd.DataFrame:
        """Calculate frame proportions for each article."""
        frame_col = f"{frame}_Detection"
        
        if frame_col not in df.columns:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy[frame_col] = pd.to_numeric(df_copy[frame_col], errors='coerce').fillna(0)
        
        # Group by article (doc_id)
        article_props = df_copy.groupby(['doc_id', 'date', 'media', 'author']).agg({
            frame_col: ['sum', 'count']
        })
        
        article_props.columns = ['frame_count', 'total_sentences']
        article_props['frame_proportion'] = article_props['frame_count'] / article_props['total_sentences']
        article_props = article_props.reset_index()
        
        return article_props
    
    def _empty_source_analysis(self) -> Dict[str, Any]:
        """Return empty source analysis structure."""
        return {
            'article_profiles': pd.DataFrame(),
            'source_network': nx.Graph(),
            'source_network_density': 0,
            'epistemic_authorities': [],
            'correlation_matrix': pd.DataFrame(),
            'mean_correlation': 0,
            'temporal_evolution': [],
            'source_proportions_mean': {},
            'source_proportions_std': {}
        }
    
    def detect_cascade_onset(self, series: pd.Series, reference_date: pd.Timestamp,
                            window_days: int = 180) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Dynamically detect cascade onset and inflection point."""
        # Get window around reference
        start = reference_date - pd.Timedelta(days=window_days//2)
        end = reference_date + pd.Timedelta(days=window_days//2)
        
        window_data = series[start:end].copy()
        window_data = window_data.dropna()
        
        if len(window_data) < 10:
            logger.warning(f"Insufficient data for cascade detection at {reference_date}")
            return reference_date, reference_date
        
        if window_data.std() < 1e-10:
            logger.warning(f"No variation in data for cascade detection at {reference_date}")
            return reference_date, reference_date
        
        try:
            # Smooth data
            sigma = min(2, len(window_data) // 5)
            if sigma < 1:
                sigma = 1
            smoothed = gaussian_filter1d(window_data.values, sigma=sigma, mode='nearest')
            
            # Calculate derivatives
            velocity = np.gradient(smoothed)
            acceleration = np.gradient(velocity)
            
            if np.all(np.isnan(acceleration)) or np.all(np.abs(acceleration) < 1e-10):
                logger.warning(f"Invalid derivatives for cascade detection at {reference_date}")
                return reference_date, reference_date
            
            # Find inflection point
            valid_accel = np.nan_to_num(acceleration, nan=0.0)
            inflection_idx = np.argmax(np.abs(valid_accel))
            
            if inflection_idx >= len(window_data):
                inflection_idx = len(window_data) - 1
            
            inflection_date = window_data.index[inflection_idx]
            
            # Find onset
            if inflection_idx > 0:
                velocity_before = velocity[:inflection_idx]
                zero_crossings = np.where(np.diff(np.sign(velocity_before)))[0]
                
                if len(zero_crossings) > 0:
                    onset_idx = zero_crossings[-1]
                    if onset_idx >= len(window_data):
                        onset_idx = 0
                    onset_date = window_data.index[onset_idx]
                else:
                    if len(velocity_before) > 0:
                        onset_idx = np.argmin(np.abs(velocity_before))
                        if onset_idx >= len(window_data):
                            onset_idx = 0
                        onset_date = window_data.index[onset_idx]
                    else:
                        onset_date = inflection_date
            else:
                onset_date = inflection_date
            
            if onset_date > inflection_date:
                onset_date = inflection_date
            
            return onset_date, inflection_date
            
        except Exception as e:
            logger.error(f"Error in cascade onset detection at {reference_date}: {str(e)}")
            return reference_date, reference_date
    
    def calculate_velocity_metrics(self, series: pd.Series, onset: pd.Timestamp,
                                  inflection: pd.Timestamp, peak: pd.Timestamp) -> Dict[str, float]:
        """Calculate velocity and acceleration metrics with outlier handling."""
        onset_idx = series.index.get_loc(onset)
        inflection_idx = series.index.get_loc(inflection)
        peak_idx = series.index.get_loc(peak)
        
        smoothed = gaussian_filter1d(series.values, sigma=2, mode='nearest')
        velocity = np.gradient(smoothed)
        acceleration = np.gradient(velocity)
        
        initial_velocity_raw = float(velocity[onset_idx])
        initial_velocity_robust = handle_outliers_iqr(np.array([initial_velocity_raw]), factor=3.0)[0]
        
        metrics = {
            'initial_velocity': initial_velocity_raw,
            'initial_velocity_robust': float(initial_velocity_robust),
            'inflection_velocity': float(velocity[inflection_idx]),
            'peak_velocity': float(np.max(velocity[onset_idx:peak_idx+1])),
            'acceleration_at_inflection': float(acceleration[inflection_idx]),
            'max_acceleration': float(np.max(acceleration[onset_idx:peak_idx+1])),
            'momentum': float(np.sum(velocity[onset_idx:peak_idx+1]))
        }
        
        return metrics
    
    def calculate_homogenization_shannon(self, df: pd.DataFrame, frame: str,
                                        onset: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
        """Calculate homogenization using Shannon entropy of frame proportions."""
        frame_cols = [f"{f}_Detection" for f in ['Cult', 'Eco', 'Envt', 'Pbh', 
                                                  'Just', 'Pol', 'Sci', 'Secu']]
        available_frames = [col for col in frame_cols if col in df.columns]
        
        cascade_data = df[(df['date'] >= onset) & (df['date'] <= end)].copy()
        
        for col in available_frames:
            cascade_data[col] = pd.to_numeric(cascade_data[col], errors='coerce').fillna(0)
        
        weekly_entropy = []
        weekly_std = []
        
        for week_start in pd.date_range(onset, end, freq='W'):
            week_end = week_start + pd.Timedelta(days=7)
            week_data = cascade_data[
                (cascade_data['date'] >= week_start) & 
                (cascade_data['date'] < week_end)
            ]
            
            if len(week_data) == 0:
                continue
            
            article_entropies = []
            
            for doc_id in week_data['doc_id'].unique():
                article_data = week_data[week_data['doc_id'] == doc_id]
                
                frame_counts = []
                for col in available_frames:
                    count = article_data[col].sum()
                    frame_counts.append(count)
                
                frame_counts = np.array(frame_counts)
                
                if frame_counts.sum() > 0:
                    article_entropy = shannon_entropy(frame_counts)
                    article_entropies.append(article_entropy)
            
            if article_entropies:
                weekly_entropy.append(np.mean(article_entropies))
                weekly_std.append(np.std(article_entropies))
        
        if not weekly_entropy:
            return {
                'shannon_entropy': 0,
                'entropy_trend': 0,
                'entropy_trend_robust': 0,
                'homogenization_score': 0,
                'proportion_convergence': 0
            }
        
        entropy_trend = 0
        entropy_trend_robust = 0
        
        if len(weekly_entropy) > 1:
            x = np.arange(len(weekly_entropy))
            entropy_slope, _, _, _, _ = stats.linregress(x, weekly_entropy)
            entropy_trend = entropy_slope
            entropy_trend_robust = handle_outliers_iqr(np.array([entropy_slope]), factor=3.0)[0]
        
        initial_std = weekly_std[0] if weekly_std else 1
        final_std = weekly_std[-1] if weekly_std else 1
        convergence = 1 - (final_std / (initial_std + 1e-10))
        
        mean_entropy = np.mean(weekly_entropy)
        max_entropy = np.log(len(available_frames))
        normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else 1
        
        homogenization_score = (1 - normalized_entropy) * 0.5 + max(0, -entropy_trend_robust) * 0.5
        homogenization_score = min(max(homogenization_score, 0), 1)
        
        return {
            'shannon_entropy': float(mean_entropy),
            'entropy_trend': float(entropy_trend),
            'entropy_trend_robust': float(entropy_trend_robust),
            'homogenization_score': float(homogenization_score),
            'proportion_convergence': float(convergence),
            'weekly_entropy': weekly_entropy,
            'weekly_std': weekly_std
        }
    
    def detect_cascade(self, df: pd.DataFrame, frame: str,
                      reference_date: pd.Timestamp,
                      cascade_type: str = 'critical_point',
                      window_days: int = 180) -> Optional[EnhancedMediaCascade]:
        """Detect and analyze a media cascade with enhanced metrics."""
        frame_col = f"{frame}_Detection"
        
        if frame_col not in df.columns:
            return None
        
        df_copy = df.copy()
        df_copy[frame_col] = pd.to_numeric(df_copy[frame_col], errors='coerce').fillna(0)
        
        daily_props = df_copy.groupby('date')[frame_col].mean()
        
        # Detect onset and inflection
        onset, inflection = self.detect_cascade_onset(daily_props, reference_date, window_days)
        
        # Find peak and end
        window_end = reference_date + pd.Timedelta(days=window_days//2)
        window_data = daily_props[onset:window_end]
        
        if len(window_data) < 5:
            return None
        
        peak_idx = window_data.argmax()
        peak_date = window_data.index[peak_idx]
        
        # Find end
        peak_value = window_data.iloc[peak_idx]
        post_peak = window_data[peak_date:]
        
        if len(post_peak) > 1:
            threshold = peak_value * 0.5
            below_threshold = post_peak[post_peak <= threshold]
            if len(below_threshold) > 0:
                end_date = below_threshold.index[0]
            else:
                end_date = post_peak.index[-1]
        else:
            end_date = peak_date + pd.Timedelta(days=30)
        
        # Calculate all enhanced metrics
        duration_days = (end_date - onset).days
        duration_days_transformed = apply_log_transformation(np.array([duration_days]))[0]
        
        # Velocity metrics
        velocity_metrics = self.calculate_velocity_metrics(
            daily_props, onset, inflection, peak_date
        )
        
        # Enhanced source analysis
        source_analysis = self.analyze_source_proportions_enhanced(df, frame, onset, end_date)
        
        # EMD-based adoption analysis
        emd_adoption = self.detect_adoption_with_emd(df, frame, onset, end_date)
        
        # Enhanced correlation network
        correlation_analysis = self.build_enhanced_correlation_network(df, frame, onset, end_date)
        
        # Entity network analysis
        entity_analysis = self.analyze_entity_networks(df, frame, onset, end_date)
        
        # Homogenization analysis
        homogenization = self.calculate_homogenization_shannon(df, frame, onset, end_date)
        
        # Build basic adoption network
        adoption_network = self.build_adoption_network(df, frame, onset, end_date)
        network_metrics = self.calculate_network_metrics(adoption_network)
        
        # Identify key actors
        key_actors = self.identify_key_actors(adoption_network)
        
        # Calculate virality
        virality = self.calculate_virality_score(df, frame, onset, end_date, velocity_metrics)
        
        # Create enhanced metrics object
        metrics = EnhancedCascadeMetrics(
            onset_date=onset,
            inflection_date=inflection,
            peak_date=peak_date,
            end_date=end_date,
            duration_days=duration_days,
            duration_days_transformed=float(duration_days_transformed),
            initial_velocity=velocity_metrics['initial_velocity'],
            initial_velocity_robust=velocity_metrics['initial_velocity_robust'],
            peak_velocity=velocity_metrics['peak_velocity'],
            acceleration=velocity_metrics['acceleration_at_inflection'],
            momentum=velocity_metrics['momentum'],
            network_density=network_metrics['density'],
            network_density_transformed=network_metrics.get('density_transformed', 0),
            clustering_coefficient=network_metrics['clustering'],
            path_length=network_metrics['avg_path_length'],
            correlation_density=correlation_analysis.get('co_occurrence_density', 0),
            correlation_density_transformed=apply_log_transformation(
                np.array([correlation_analysis.get('co_occurrence_density', 0)]))[0],
            mean_correlation=source_analysis.get('mean_correlation', 0),
            co_occurrence_density=correlation_analysis.get('co_occurrence_density', 0),
            temporal_proximity_score=correlation_analysis.get('temporal_proximity_score', 0),
            adaptive_threshold=correlation_analysis.get('adaptive_threshold', 0.3),
            top_journalists=key_actors['journalists'],
            top_media=key_actors['media'],
            early_adopters=self._get_early_adopters(adoption_network, top_n=10),
            source_diversity=shannon_entropy(
                np.array(list(source_analysis.get('source_proportions_mean', {}).values()))
            ) if source_analysis.get('source_proportions_mean') else 0,
            dominant_sources={},
            source_concentration=0,
            source_convergence=source_analysis.get('mean_correlation', 0),
            source_proportions_by_article=source_analysis.get('source_proportions_mean', {}),
            source_network_density=source_analysis.get('source_network_density', 0),
            epistemic_authorities=source_analysis.get('epistemic_authorities', []),
            entity_convergence=entity_analysis.get('entity_convergence', 0),
            shared_entities=entity_analysis.get('shared_entities', {}),
            entity_network=entity_analysis.get('entity_network', nx.Graph()),
            homogenization_score=virality.get('homogenization', 0),
            homogenization_shannon=homogenization['homogenization_score'],
            entropy_trend=homogenization['entropy_trend'],
            entropy_trend_robust=homogenization['entropy_trend_robust'],
            emotional_intensity=virality.get('emotional_intensity', 0),
            viral_coefficient=virality.get('viral_coefficient', 0),
            adoption_emd_score=emd_adoption.get('adoption_rate', 0),
            proportion_change_rate=emd_adoption.get('mean_emd', 0),
            statistical_significance={},
            consensus_significant=False
        )
        
        # Calculate component scores with sub-indices
        sub_indices = self._calculate_sub_indices(metrics, len(cascade_articles) if 'cascade_articles' in locals() else 0)
        
        # Component scores
        journalist_score = np.mean(list(sub_indices['journalist'].values()))
        media_score = np.mean(list(sub_indices['media'].values()))
        intensity_score = np.mean(list(sub_indices['intensity'].values()))
        network_score = np.mean(list(sub_indices['network'].values()))
        virality_score = np.mean(list(sub_indices['virality'].values()))
        
        # Total score
        total_score = 0.2 * (journalist_score + media_score + intensity_score + 
                            network_score + virality_score)
        
        # Classify strength
        if total_score >= 0.7:
            strength = 'strong'
        elif total_score >= 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        # Create enhanced cascade object
        cascade = EnhancedMediaCascade(
            frame=frame,
            reference_date=reference_date,
            cascade_type=cascade_type,
            metrics=metrics,
            journalist_score=journalist_score,
            media_score=media_score,
            intensity_score=intensity_score,
            network_score=network_score,
            virality_score=virality_score,
            sub_indices=sub_indices,
            total_score=total_score,
            cascade_strength=strength,
            temporal_data=window_data,
            network_graph=adoption_network,
            correlation_network=correlation_analysis.get('network', nx.DiGraph()),
            adoption_sequence=emd_adoption.get('network', nx.DiGraph()),
            source_citation_network=source_analysis.get('source_network', nx.Graph()),
            entity_network=entity_analysis.get('entity_network', nx.Graph()),
            detection_method='enhanced_comprehensive',
            confidence=total_score,
            notes="Enhanced detection with source proportions, EMD, adaptive thresholds, and NER entities"
        )
        
        return cascade
    
    def _calculate_sub_indices(self, metrics: EnhancedCascadeMetrics, 
                              n_articles: int) -> Dict[str, Dict[str, float]]:
        """Calculate detailed sub-indices for each component."""
        sub_indices = {
            'journalist': {
                'adoption_count': min(len(metrics.top_journalists) / 50, 1.0),
                'centrality': metrics.top_journalists[0][1] * 10 if metrics.top_journalists else 0,
                'early_adoption': min(len(metrics.early_adopters) / 10, 1.0),
                'emd_adoption': metrics.adoption_emd_score
            },
            'media': {
                'diffusion_breadth': min(len(metrics.top_media) / 10, 1.0),
                'source_diversity': metrics.source_diversity,
                'epistemic_authorities': min(len(metrics.epistemic_authorities) / 5, 1.0),
                'source_network_density': metrics.source_network_density
            },
            'intensity': {
                'velocity': 1 / (1 + np.exp(-5 * metrics.peak_velocity)),
                'acceleration': 1 / (1 + np.exp(-10 * metrics.acceleration)),
                'volume': min(n_articles / 1000, 1.0),
                'proportion_change': metrics.proportion_change_rate
            },
            'network': {
                'density': min(metrics.network_density_transformed * 2, 1.0),
                'correlation': min(abs(metrics.mean_correlation) * 2, 1.0),
                'co_occurrence': metrics.co_occurrence_density,
                'temporal_proximity': metrics.temporal_proximity_score,
                'entity_convergence': metrics.entity_convergence
            },
            'virality': {
                'homogenization_lexical': metrics.homogenization_score,
                'homogenization_shannon': metrics.homogenization_shannon,
                'entropy_trend': max(0, -metrics.entropy_trend_robust),
                'emotional_intensity': metrics.emotional_intensity,
                'viral_coefficient': metrics.viral_coefficient
            }
        }
        
        # Normalize all sub-indices to [0, 1]
        for component in sub_indices:
            for sub_index in sub_indices[component]:
                value = sub_indices[component][sub_index]
                sub_indices[component][sub_index] = min(max(value, 0), 1)
        
        return sub_indices
    
    def build_adoption_network(self, df: pd.DataFrame, frame: str,
                              onset: pd.Timestamp, end: pd.Timestamp) -> nx.DiGraph:
        """Build directed adoption sequence network."""
        G = nx.DiGraph()
        
        cascade_data = df[(df['date'] >= onset) & (df['date'] <= end)].copy()
        
        frame_col = f"{frame}_Detection"
        
        if frame_col in cascade_data.columns:
            cascade_data[frame_col] = pd.to_numeric(cascade_data[frame_col], errors='coerce').fillna(0)
        
        journalist_adoptions = {}
        media_adoptions = {}
        
        for _, row in cascade_data.iterrows():
            try:
                frame_value = float(row.get(frame_col, 0))
                if frame_value == 1:
                    if pd.notna(row.get('author')):
                        author = self._clean_name(row['author'])
                        if author and author not in journalist_adoptions:
                            journalist_adoptions[author] = row['date']
                    
                    if pd.notna(row.get('media')):
                        media = str(row['media'])
                        if media and media not in media_adoptions:
                            media_adoptions[media] = row['date']
            except (ValueError, TypeError):
                continue
        
        # Add nodes
        for journalist, date in journalist_adoptions.items():
            G.add_node(f"J:{journalist}", type='journalist', 
                      adoption_date=date, name=journalist)
        
        for media, date in media_adoptions.items():
            G.add_node(f"M:{media}", type='media',
                      adoption_date=date, name=media)
        
        # Add edges
        for _, row in cascade_data.iterrows():
            try:
                frame_value = float(row.get(frame_col, 0))
                if frame_value == 1 and pd.notna(row.get('author')) and pd.notna(row.get('media')):
                    author = f"J:{self._clean_name(row['author'])}"
                    media = f"M:{row['media']}"
                    if author in G.nodes and media in G.nodes:
                        G.add_edge(author, media, weight=1.0, date=row['date'])
            except (ValueError, TypeError):
                continue
        
        # Media to media based on temporal sequence
        media_list = sorted(media_adoptions.items(), key=lambda x: x[1])
        for i in range(len(media_list)-1):
            for j in range(i+1, min(i+5, len(media_list))):
                time_diff = (media_list[j][1] - media_list[i][1]).days
                if time_diff <= 30:
                    weight = 1.0 / (1 + time_diff/7)
                    G.add_edge(f"M:{media_list[i][0]}", f"M:{media_list[j][0]}",
                             weight=weight, time_diff=time_diff)
        
        return G
    
    def calculate_network_metrics(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Calculate comprehensive network metrics with transformations."""
        if len(G) == 0:
            return {
                'density': 0,
                'density_transformed': 0,
                'clustering': 0,
                'avg_path_length': 0,
                'centrality': {}
            }
        
        G_undirected = G.to_undirected()
        
        density_raw = nx.density(G)
        density_transformed = apply_log_transformation(np.array([density_raw]))[0]
        
        metrics = {
            'density': density_raw,
            'density_transformed': float(density_transformed),
            'clustering': nx.average_clustering(G_undirected) if len(G) > 2 else 0,
            'avg_path_length': 0
        }
        
        if nx.is_weakly_connected(G):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G_undirected)
        else:
            components = list(nx.weakly_connected_components(G))
            if components:
                path_lengths = []
                for comp in components:
                    if len(comp) > 1:
                        subgraph = G.subgraph(comp).to_undirected()
                        path_lengths.append(nx.average_shortest_path_length(subgraph))
                metrics['avg_path_length'] = np.mean(path_lengths) if path_lengths else 0
        
        centrality = {}
        
        try:
            pagerank = nx.pagerank(G, max_iter=100)
            centrality['pagerank'] = pagerank
        except:
            centrality['pagerank'] = {n: 1/len(G) for n in G.nodes()}
        
        try:
            betweenness = nx.betweenness_centrality(G)
            centrality['betweenness'] = betweenness
        except:
            centrality['betweenness'] = {n: 0 for n in G.nodes()}
        
        in_degree = nx.in_degree_centrality(G)
        centrality['in_degree'] = in_degree
        
        out_degree = nx.out_degree_centrality(G)
        centrality['out_degree'] = out_degree
        
        metrics['centrality'] = centrality
        
        return metrics
    
    def identify_key_actors(self, G: nx.DiGraph, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Identify key journalists and media based on centrality."""
        if len(G) == 0:
            return {'journalists': [], 'media': []}
        
        metrics = self.calculate_network_metrics(G)
        pagerank = metrics['centrality']['pagerank']
        
        journalists = []
        media = []
        
        for node, score in pagerank.items():
            node_data = G.nodes.get(node, {})
            node_type = node_data.get('type', None)
            
            if 'name' in node_data:
                name = node_data['name']
            else:
                if node.startswith('J:'):
                    name = node[2:]
                    node_type = 'journalist' if not node_type else node_type
                elif node.startswith('M:'):
                    name = node[2:]
                    node_type = 'media' if not node_type else node_type
                else:
                    name = node
            
            if node_type == 'journalist':
                journalists.append((name, score))
            elif node_type == 'media':
                media.append((name, score))
        
        journalists.sort(key=lambda x: x[1], reverse=True)
        media.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'journalists': journalists[:top_n],
            'media': media[:top_n]
        }
    
    def calculate_virality_score(self, df: pd.DataFrame, frame: str,
                                onset: pd.Timestamp, end: pd.Timestamp,
                                velocity_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive virality score."""
        frame_col = f"{frame}_Detection"
        cascade_data = df[(df['date'] >= onset) & (df['date'] <= end)].copy()
        
        if frame_col in cascade_data.columns:
            cascade_data[frame_col] = pd.to_numeric(cascade_data[frame_col], errors='coerce').fillna(0)
            cascade_data = cascade_data[cascade_data[frame_col] == 1]
        
        homogenization = self._calculate_homogenization(cascade_data)
        
        peak_velocity = velocity_metrics.get('peak_velocity', 0)
        acceleration = velocity_metrics.get('max_acceleration', 0)
        
        if 'Emotion:_Negative' in cascade_data.columns:
            cascade_data['Emotion:_Negative'] = pd.to_numeric(
                cascade_data['Emotion:_Negative'], errors='coerce'
            ).fillna(0)
            negative_ratio = cascade_data['Emotion:_Negative'].mean()
        else:
            negative_ratio = 0
        
        viral_coef = self._calculate_viral_coefficient(cascade_data)
        
        duration_days = (end - onset).days
        if duration_days > 0:
            daily_intensity = len(cascade_data) / duration_days
            intensity_score = min(daily_intensity / 50, 1.0)
        else:
            intensity_score = 0
        
        virality_components = {
            'homogenization': homogenization,
            'peak_velocity': peak_velocity,
            'acceleration': 1 / (1 + np.exp(-10 * acceleration)),
            'emotional_intensity': negative_ratio,
            'viral_coefficient': viral_coef,
            'intensity': intensity_score
        }
        
        weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
        total_virality = np.average(list(virality_components.values()), weights=weights)
        
        virality_components['total'] = float(total_virality)
        
        return virality_components
    
    def _calculate_homogenization(self, cascade_data: pd.DataFrame) -> float:
        """Calculate content homogenization score."""
        if len(cascade_data) < 2:
            return 0
        
        if 'sentences' not in cascade_data.columns:
            return 0
        
        valid_sentences = cascade_data['sentences'].dropna()
        valid_sentences = valid_sentences[valid_sentences.str.strip() != '']
        
        if len(valid_sentences) == 0:
            return 0
        
        try:
            all_words = ' '.join(valid_sentences.astype(str)).lower().split()
            if len(all_words) > 0:
                unique_words = len(set(all_words))
                total_words = len(all_words)
                lexical_diversity = unique_words / total_words
                homogenization = 1 - lexical_diversity
            else:
                homogenization = 0
        except:
            homogenization = 0
        
        return min(max(homogenization, 0), 1)
    
    def _calculate_viral_coefficient(self, cascade_data: pd.DataFrame) -> float:
        """Calculate viral coefficient (reproduction rate)."""
        cascade_data['week'] = cascade_data['date'].dt.to_period('W')
        weekly_counts = cascade_data.groupby('week').size()
        
        if len(weekly_counts) < 2:
            return 0
        
        growth_rates = []
        for i in range(1, len(weekly_counts)):
            if weekly_counts.iloc[i-1] > 0:
                rate = (weekly_counts.iloc[i] - weekly_counts.iloc[i-1]) / weekly_counts.iloc[i-1]
                growth_rates.append(rate)
        
        if growth_rates:
            avg_growth = np.mean(growth_rates)
            viral_coef = min(max(avg_growth, 0), 2.0) / 2.0
        else:
            viral_coef = 0
        
        return viral_coef
    
    def _clean_name(self, name: str) -> str:
        """Clean author name for matching."""
        if pd.isna(name):
            return ""
        name = str(name).strip()
        for suffix in [', Staff', ', Reuters', ', AP', ', Canadian Press']:
            name = name.replace(suffix, '')
        return name.strip()
    
    def _get_early_adopters(self, G: nx.DiGraph, top_n: int = 10) -> List[Tuple[str, pd.Timestamp]]:
        """Get earliest adopters from network."""
        adopters = []
        
        for node, data in G.nodes(data=True):
            if data.get('type') == 'journalist':
                if 'name' in data:
                    name = data['name']
                else:
                    if node.startswith('J:'):
                        name = node[2:]
                    else:
                        name = node
                
                adoption_date = data.get('adoption_date')
                if adoption_date:
                    adopters.append((name, adoption_date))
        
        adopters.sort(key=lambda x: x[1])
        
        return adopters[:top_n]
    
    def detect_spontaneous_cascades(self, df: pd.DataFrame, frame: str,
                                   exclude_periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
                                   min_threshold: float = 0.1) -> List[EnhancedMediaCascade]:
        """Detect cascades outside of critical points."""
        cascades = []
        
        frame_col = f"{frame}_Detection"
        
        if frame_col not in df.columns:
            return cascades
        
        df_copy = df.copy()
        df_copy[frame_col] = pd.to_numeric(df_copy[frame_col], errors='coerce').fillna(0)
        
        daily_props = df_copy.groupby('date')[frame_col].mean()
        
        smoothed = gaussian_filter1d(daily_props.values, sigma=7, mode='nearest')
        
        peaks, properties = signal.find_peaks(smoothed, 
                                             height=min_threshold,
                                             distance=30,
                                             prominence=0.05)
        
        for peak_idx in peaks:
            if peak_idx < len(daily_props):
                peak_date = daily_props.index[peak_idx]
                
                in_excluded = False
                for start, end in exclude_periods:
                    if start <= peak_date <= end:
                        in_excluded = True
                        break
                
                if not in_excluded:
                    cascade = self.detect_cascade(df, frame, peak_date, 
                                                cascade_type='spontaneous',
                                                window_days=120)
                    
                    if cascade and cascade.total_score >= 0.3:
                        cascades.append(cascade)
        
        return cascades


# Alias for backward compatibility
AdvancedCascadeDetector = EnhancedCascadeDetector
MediaCascade = EnhancedMediaCascade
CascadeMetrics = EnhancedCascadeMetrics