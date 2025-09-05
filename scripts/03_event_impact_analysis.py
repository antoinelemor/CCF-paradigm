"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
03_event_impact_analysis.py

MAIN OBJECTIVE:
---------------
This script analyzes the relationship between detected events and their impact on
media cascades and frame trend changes. It identifies which types of events act as
"focusing events" that trigger paradigm shifts through cascade effects.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm
- All custom modules

MAIN FEATURES:
--------------
1) Rigorous event detection from mentions
2) Event impact analysis on cascades
3) Event impact analysis on trends
4) Event-frame relationship mapping
5) Focusing event identification
6) Statistical validation
7) Comprehensive visualization

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import custom modules
from src.database_access.archives.db_connector import DatabaseConnector
from src.database_access.archives.data_processor import FrameDataProcessor
from src.event_detection_utils import EventDetector, EventImpactAnalyzer, EventFramePatternAnalyzer

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

EVENT_COLORS = {
    'Event_1_SUB': '#FF6B6B',  # Red - extreme weather
    'Event_2_SUB': '#4ECDC4',  # Teal - conferences
    'Event_3_SUB': '#45B7D1',  # Blue - publications
    'Event_4_SUB': '#F7DC6F',  # Yellow - elections
    'Event_5_SUB': '#BB8FCE',  # Purple - policy
    'Event_6_SUB': '#85C1E2',  # Light blue - court
    'Event_7_SUB': '#F8C471',  # Orange - cultural
    'Event_8_SUB': '#82E0AA'   # Green - protest
}

# Paths
DATA_DIR = project_root / "data" / "03_events_effects"
RESULTS_DIR = project_root / "results" / "03_events_effects"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EventImpactAnalysis:
    """Main class for event impact analysis."""
    
    def __init__(self, n_workers: int = 8):
        """Initialize analysis."""
        self.n_workers = n_workers
        self.db_connector = DatabaseConnector()
        self.processor = FrameDataProcessor(n_workers=n_workers)
        self.event_detector = EventDetector(n_workers=n_workers)
        self.impact_analyzer = EventImpactAnalyzer()
        
        self.detected_events = None
        self.impact_results = None
        self.relationships = None
        self.focusing_events = None
    
    @staticmethod
    def convert_all_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Convert all date-like columns to pandas Timestamp format."""
        df_copy = df.copy()
        
        for col in df_copy.columns:
            # Check if column contains date-like data
            if df_copy[col].dtype == 'datetime64[ns]' or \
               str(df_copy[col].dtype).startswith('datetime') or \
               'date' in col.lower() or 'time' in col.lower():
                try:
                    # Convert to pandas datetime first
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                    
                    # Then ensure all values are pandas Timestamps
                    df_copy[col] = df_copy[col].apply(
                        lambda x: pd.Timestamp(x) if pd.notna(x) else pd.NaT
                    )
                except Exception as e:
                    print(f"Warning: Could not convert column {col}: {e}")
        
        return df_copy
    
    @staticmethod
    def ensure_timestamp(date_value):
        """Ensure a date value is a pandas Timestamp."""
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
        
    def load_event_data(self) -> pd.DataFrame:
        """Load data with event detection columns."""
        print("Loading event data from database...")
        
        # Query for all necessary columns
        query = """
        SELECT doc_id, sentence_id, sentences, date, author, media,
               "Cult_Detection", "Eco_Detection", "Envt_Detection", "Pbh_Detection",
               "Just_Detection", "Pol_Detection", "Sci_Detection", "Secu_Detection",
               "Event_Detection", "Event_1_SUB", "Event_2_SUB", "Event_3_SUB",
               "Event_4_SUB", "Event_5_SUB", "Event_6_SUB", "Event_7_SUB", "Event_8_SUB"
        FROM "CCF_processed_data"
        WHERE date IS NOT NULL
        AND date < '2025-01-01'
        ORDER BY date, doc_id, sentence_id
        """
        
        df = self.db_connector.read_data(query, show_progress=True)
        
        # Apply comprehensive date conversion
        print("Converting all date columns...")
        df = self.convert_all_dates(df)
        
        # Double-check the main date column
        if 'date' in df.columns:
            df['date'] = df['date'].apply(self.ensure_timestamp)
        
        # Filter to ensure no 2025 data
        df = df[df['date'] < pd.Timestamp('2025-01-01')]
        
        print(f"Loaded {len(df):,} sentences")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Date column type: {df['date'].dtype}")
        
        # Check event detection coverage
        event_coverage = (df['Event_Detection'] == 1).sum()
        print(f"Sentences with events: {event_coverage:,} ({event_coverage/len(df)*100:.1f}%)")
        
        return df
    
    def load_previous_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load results from previous analyses."""
        print("\nLoading previous analysis results...")
        
        # Load cascade results
        cascade_path = DATA_DIR / "media_cascade_summary.csv"
        if not cascade_path.exists():
            raise FileNotFoundError("Run media cascade analysis first!")
        
        cascade_results = pd.read_csv(cascade_path)
        cascade_results['critical_date'] = pd.to_datetime(cascade_results['critical_date'])
        
        # Convert numeric columns to appropriate types
        numeric_columns = ['cascade_strength', 'media_count', 'start_week', 'peak_week', 
                          'end_week', 'duration_weeks', 'magnitude_increase']
        for col in numeric_columns:
            if col in cascade_results.columns:
                cascade_results[col] = pd.to_numeric(cascade_results[col], errors='coerce')
        
        # Validate cascade_strength column
        if 'cascade_strength' in cascade_results.columns:
            non_numeric = cascade_results['cascade_strength'].isna().sum()
            if non_numeric > 0:
                print(f"Warning: {non_numeric} non-numeric values found in cascade_strength column")
                # Fill NaN values with 0
                cascade_results['cascade_strength'] = cascade_results['cascade_strength'].fillna(0)
        
        print(f"Loaded {len(cascade_results)} cascade results")
        
        # Debug info
        if 'cascade_strength' in cascade_results.columns:
            print(f"Cascade strength data type: {cascade_results['cascade_strength'].dtype}")
            print(f"Cascade strength range: {cascade_results['cascade_strength'].min():.3f} to {cascade_results['cascade_strength'].max():.3f}")
        
        # Load trend results
        trend_path = DATA_DIR / "critical_trend_dates.csv"
        if not trend_path.exists():
            raise FileNotFoundError("Run trend analysis first!")
        
        trend_results = pd.read_csv(trend_path)
        trend_results['date'] = pd.to_datetime(trend_results['date'])
        
        # Convert numeric columns
        numeric_trend_columns = ['value', 'magnitude', 'rate', 'significance']
        for col in numeric_trend_columns:
            if col in trend_results.columns:
                trend_results[col] = pd.to_numeric(trend_results[col], errors='coerce')
        
        print(f"Loaded {len(trend_results)} trend critical dates")
        
        return cascade_results, trend_results
    
    def detect_events(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Detect actual event occurrences."""
        print("\nDetecting event occurrences...")
        
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Comprehensive date conversion - handle all possible date formats
        print("Converting date formats...")
        df_copy = self.convert_all_dates(df_copy)
        
        # Verify conversion
        if 'date' in df_copy.columns:
            print(f"Date column type after conversion: {df_copy['date'].dtype}")
        
        # Adjust detector parameters based on data volume
        total_days = (df_copy['date'].max() - df_copy['date'].min()).days
        avg_daily_articles = df_copy.groupby('date')['doc_id'].nunique().mean()
        
        # Dynamic thresholds
        if avg_daily_articles < 10:
            min_mentions = 5
            min_media = 2
        elif avg_daily_articles < 50:
            min_mentions = 10
            min_media = 3
        else:
            min_mentions = 20
            min_media = 5
        
        print(f"Using detection thresholds: min_mentions={min_mentions}, min_media={min_media}")
        
        # Update detector parameters
        self.event_detector.min_mentions_threshold = min_mentions
        self.event_detector.min_media_diversity = min_media
        
        # Wrap the detector to handle potential numpy.datetime64 issues
        try:
            # Call the original detector
            detected_events = self.event_detector.detect_event_occurrences(df_copy, show_progress=True)
        except Exception as e:
            print(f"Error during event detection: {e}")
            print("Attempting alternative detection approach...")
            detected_events = self._safe_detect_events(df_copy)
        
        # Convert any dates in the results
        for event_type, events_df in detected_events.items():
            if len(events_df) > 0:
                detected_events[event_type] = self.convert_all_dates(events_df)
        
        # Summary statistics
        print("\nDetected events summary:")
        total_events = 0
        for event_type, events_df in detected_events.items():
            event_name = EventDetector.EVENT_TYPES.get(event_type, event_type)
            print(f"  {event_name}: {len(events_df)} events")
            total_events += len(events_df)
        
        print(f"Total detected events: {total_events}")
        
        self.detected_events = detected_events
        return detected_events
    
    def _safe_detect_events(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Alternative event detection with explicit date handling."""
        detected_events = {}
        
        # Event types to detect
        event_types = [f'Event_{i}_SUB' for i in range(1, 9)]
        
        for event_type in tqdm(event_types, desc="Detecting event occurrences"):
            try:
                if event_type not in df.columns:
                    continue
                
                # Filter data where event is detected
                event_data = df[df[event_type] == 1].copy()
                
                if len(event_data) == 0:
                    detected_events[event_type] = pd.DataFrame()
                    continue
                
                # Ensure all dates are properly formatted
                event_data = self.convert_all_dates(event_data)
                
                # Group by date to find event occurrences
                daily_counts = event_data.groupby('date').agg({
                    'doc_id': 'nunique',
                    'media': lambda x: x.nunique(),
                    event_type: 'sum'
                }).reset_index()
                
                daily_counts.columns = ['date', 'unique_docs', 'unique_media', 'mentions']
                
                # Filter based on thresholds
                significant_dates = daily_counts[
                    (daily_counts['mentions'] >= self.event_detector.min_mentions_threshold) & 
                    (daily_counts['unique_media'] >= self.event_detector.min_media_diversity)
                ]
                
                if len(significant_dates) == 0:
                    detected_events[event_type] = pd.DataFrame()
                    continue
                
                # Create event records
                events = []
                for _, row in significant_dates.iterrows():
                    event_record = {
                        'event_type': event_type,
                        'event_name': EventDetector.EVENT_TYPES.get(event_type, event_type),
                        'start_date': self.ensure_timestamp(row['date']),
                        'end_date': self.ensure_timestamp(row['date']),
                        'peak_date': self.ensure_timestamp(row['date']),
                        'duration_days': 1,
                        'total_mentions': row['mentions'],
                        'unique_media': row['unique_media'],
                        'intensity': row['mentions']
                    }
                    events.append(event_record)
                
                detected_events[event_type] = pd.DataFrame(events)
                
            except Exception as e:
                print(f"Error detecting {event_type}: {e}")
                detected_events[event_type] = pd.DataFrame()
        
        return detected_events
    
    def analyze_impacts(self, df: pd.DataFrame, cascade_results: pd.DataFrame,
                       trend_results: pd.DataFrame) -> pd.DataFrame:
        """Analyze event impacts on cascades and trends."""
        print("\nAnalyzing event impacts...")
        
        if not self.detected_events:
            print("No events detected!")
            return pd.DataFrame()
        
        # Prepare frame data (weekly aggregation)
        print("Preparing frame data...")
        weekly_frames = self._prepare_weekly_frame_data(df)
        
        # Analyze impacts
        impact_results = self.impact_analyzer.analyze_event_impacts(
            self.detected_events,
            cascade_results,
            trend_results,
            weekly_frames,
            show_progress=True
        )
        
        print(f"\nAnalyzed impacts for {len(impact_results)} events")
        
        # Summary statistics
        if len(impact_results) > 0:
            avg_impact = impact_results['overall_impact_score'].mean()
            high_impact = (impact_results['overall_impact_score'] > 0.5).sum()
            print(f"Average impact score: {avg_impact:.3f}")
            print(f"High-impact events (>0.5): {high_impact}")
        
        self.impact_results = impact_results
        return impact_results
    
    def _prepare_weekly_frame_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare weekly frame proportions."""
        # Process frame data
        processed = self.processor.process_frame_data(df, show_progress=False)
        
        # Apply comprehensive date conversion
        processed = self.convert_all_dates(processed)
        
        # Ensure date is properly formatted
        processed['date'] = processed['date'].apply(self.ensure_timestamp)
        
        # Weekly aggregation
        processed['week'] = processed['date'].dt.to_period('W').dt.to_timestamp()
        
        weekly_data = []
        for week, group in processed.groupby('week'):
            week_dict = {'date': self.ensure_timestamp(week)}
            
            # Calculate frame proportions
            total = 0
            for frame in FRAME_NAMES:
                col = f"{frame}_Detection"
                if col in group.columns:
                    count = group[col].sum()
                    week_dict[frame] = count
                    total += count
            
            # Normalize to proportions
            if total > 0:
                for frame in FRAME_NAMES:
                    week_dict[frame] = week_dict.get(frame, 0) / total
            
            weekly_data.append(week_dict)
        
        return pd.DataFrame(weekly_data)
    
    def analyze_relationships(self) -> Dict:
        """Analyze event-frame relationships."""
        print("\nAnalyzing event-frame relationships...")
        
        if self.impact_results is None or len(self.impact_results) == 0:
            print("No impact results to analyze!")
            return {}
        
        relationships = EventFramePatternAnalyzer.analyze_event_frame_relationships(
            self.impact_results
        )
        
        # Print summary
        print("\nEvent effectiveness ranking:")
        effectiveness = relationships['event_effectiveness']
        sorted_events = sorted(effectiveness.items(), 
                             key=lambda x: x[1]['avg_impact_score'], 
                             reverse=True)
        
        for event_type, metrics in sorted_events[:5]:
            event_name = EventDetector.EVENT_TYPES.get(event_type, event_type)
            print(f"  {event_name}: {metrics['avg_impact_score']:.3f} "
                  f"(n={metrics['n_events']})")
        
        print("\nFrame responsiveness ranking:")
        responsiveness = relationships['frame_responsiveness']
        sorted_frames = sorted(responsiveness.items(),
                             key=lambda x: x[1]['responsiveness_score'],
                             reverse=True)
        
        for frame, metrics in sorted_frames:
            print(f"  {frame}: {metrics['responsiveness_score']:.3f} "
                  f"({metrics['total_responses']} responses)")
        
        self.relationships = relationships
        return relationships
    
    def identify_focusing_events(self) -> pd.DataFrame:
        """Identify focusing events."""
        print("\nIdentifying focusing events...")
        
        if self.impact_results is None or len(self.impact_results) == 0:
            print("No impact results!")
            return pd.DataFrame()
        
        # Identify focusing events
        focusing_events = EventFramePatternAnalyzer.identify_focusing_events(
            self.impact_results,
            impact_threshold=0.6  # Slightly lower threshold for more events
        )
        
        print(f"Identified {len(focusing_events)} focusing events")
        
        if len(focusing_events) > 0:
            print("\nTop focusing events:")
            for idx, event in focusing_events.head(10).iterrows():
                event_name = EventDetector.EVENT_TYPES.get(event['event_type'], event['event_type'])
                print(f"  {event['peak_date'].strftime('%Y-%m-%d')} - {event_name}: "
                      f"strength={event['focusing_strength']:.3f}")
        
        self.focusing_events = focusing_events
        return focusing_events
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\nCreating visualizations...")
        
        # 1. Event timeline
        self._create_event_timeline()
        
        # 2. Impact analysis dashboard
        self._create_impact_dashboard()
        
        # 3. Event-frame relationships
        self._create_relationship_heatmap()
        
        # 4. Focusing events analysis
        self._create_focusing_events_plot()
        
        # 5. Temporal patterns
        self._create_temporal_patterns()
    
    def _create_event_timeline(self):
        """Create timeline of detected events."""
        if not self.detected_events:
            return
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data
        all_events = []
        for event_type, events_df in self.detected_events.items():
            for _, event in events_df.iterrows():
                all_events.append({
                    'date': pd.Timestamp(event['peak_date']),  # Ensure Timestamp format
                    'type': event_type,
                    'intensity': event['intensity'],
                    'name': event['event_name']
                })
        
        events_df = pd.DataFrame(all_events)
        
        # Plot by event type
        y_positions = {event_type: i for i, event_type in enumerate(EVENT_COLORS.keys())}
        
        for event_type in EVENT_COLORS.keys():
            type_events = events_df[events_df['type'] == event_type]
            if len(type_events) > 0:
                ax.scatter(type_events['date'], 
                         [y_positions[event_type]] * len(type_events),
                         s=type_events['intensity'] * 50,
                         c=EVENT_COLORS[event_type],
                         alpha=0.7,
                         edgecolors='black',
                         linewidth=0.5,
                         label=EventDetector.EVENT_TYPES.get(event_type, event_type))
        
        # Formatting
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([EventDetector.EVENT_TYPES.get(et, et) 
                           for et in y_positions.keys()])
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Timeline of Detected Events (size = intensity)', fontsize=14)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "event_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_impact_dashboard(self):
        """Create comprehensive impact analysis dashboard."""
        if self.impact_results is None or len(self.impact_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Impact score distribution
        ax1 = axes[0, 0]
        ax1.hist(self.impact_results['overall_impact_score'], 
                bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(self.impact_results['overall_impact_score'].mean(), 
                   color='red', linestyle='--', label='Mean')
        ax1.set_xlabel('Overall Impact Score')
        ax1.set_ylabel('Number of Events')
        ax1.set_title('Distribution of Event Impact Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cascade vs Trend Impact
        ax2 = axes[0, 1]
        cascade_scores = self.impact_results['cascade_impact'].apply(
            lambda x: x.get('triggered_cascades', 0) * x.get('avg_cascade_strength', 0)
        )
        trend_scores = self.impact_results['trend_impact'].apply(
            lambda x: x.get('triggered_trends', 0) / 10
        )
        
        ax2.scatter(cascade_scores, trend_scores, 
                   c=self.impact_results['overall_impact_score'],
                   cmap='viridis', alpha=0.6, s=50)
        
        # Add diagonal line
        max_val = max(cascade_scores.max(), trend_scores.max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        ax2.set_xlabel('Cascade Impact')
        ax2.set_ylabel('Trend Impact')
        ax2.set_title('Cascade vs Trend Impact')
        
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Overall Impact')
        
        # 3. Impact by Event Type
        ax3 = axes[1, 0]
        impact_by_type = self.impact_results.groupby('event_type')['overall_impact_score'].agg(['mean', 'std', 'count'])
        
        event_types = impact_by_type.index
        x_pos = np.arange(len(event_types))
        
        bars = ax3.bar(x_pos, impact_by_type['mean'], 
                       yerr=impact_by_type['std'],
                       capsize=5,
                       color=[EVENT_COLORS[et] for et in event_types],
                       edgecolor='black',
                       linewidth=0.5)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, impact_by_type['count'])):
            ax3.text(i, bar.get_height() + impact_by_type['std'].iloc[i] + 0.01,
                    f'n={count}', ha='center', fontsize=8)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([EventDetector.EVENT_TYPES.get(et, et)[:10] 
                            for et in event_types], rotation=45, ha='right')
        ax3.set_ylabel('Average Impact Score')
        ax3.set_title('Impact by Event Type')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 4. Temporal Lag Analysis
        ax4 = axes[1, 1]
        
        cascade_lags = self.impact_results['cascade_impact'].apply(
            lambda x: x.get('cascade_lag_days', np.nan)
        ).dropna()
        
        trend_lags = self.impact_results['trend_impact'].apply(
            lambda x: x.get('trend_lag_days', np.nan)
        ).dropna()
        
        data_to_plot = []
        labels = []
        
        if len(cascade_lags) > 0:
            data_to_plot.append(cascade_lags)
            labels.append(f'Cascade\n(n={len(cascade_lags)})')
        
        if len(trend_lags) > 0:
            data_to_plot.append(trend_lags)
            labels.append(f'Trend\n(n={len(trend_lags)})')
        
        if data_to_plot:
            bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Days from Event')
        ax4.set_title('Temporal Lag Distribution')
        ax4.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('Event Impact Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "impact_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_relationship_heatmap(self):
        """Create event-frame relationship heatmap."""
        if not self.relationships:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Event-Frame Affinity Matrix
        affinity_data = self.relationships['event_frame_affinity']
        
        # Convert to DataFrame
        event_types = list(EVENT_COLORS.keys())
        frames = FRAME_NAMES
        
        affinity_matrix = pd.DataFrame(index=event_types, columns=frames)
        
        for event_type in event_types:
            if event_type in affinity_data:
                for frame in frames:
                    affinity_matrix.loc[event_type, frame] = affinity_data[event_type].get(frame, 0)
        
        affinity_matrix = affinity_matrix.fillna(0).astype(float)
        
        # Create heatmap
        sns.heatmap(affinity_matrix, 
                   annot=True, fmt='.2f',
                   cmap='YlOrRd',
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Affinity Score'},
                   ax=ax1)
        
        ax1.set_yticklabels([EventDetector.EVENT_TYPES.get(et, et)[:15] 
                            for et in event_types], rotation=0)
        ax1.set_xticklabels(frames, rotation=45)
        ax1.set_title('Event-Frame Affinity Matrix')
        
        # 2. Summary metrics
        ax2.axis('off')
        
        # Event effectiveness
        effectiveness = self.relationships['event_effectiveness']
        sorted_events = sorted(effectiveness.items(),
                             key=lambda x: x[1]['avg_impact_score'],
                             reverse=True)
        
        y_pos = 0.9
        ax2.text(0.1, y_pos, 'Event Effectiveness:', fontsize=12, fontweight='bold')
        y_pos -= 0.05
        
        for event_type, metrics in sorted_events[:5]:
            event_name = EventDetector.EVENT_TYPES.get(event_type, event_type)
            text = f"{event_name}: {metrics['avg_impact_score']:.3f} (n={metrics['n_events']})"
            ax2.text(0.15, y_pos, text, fontsize=10)
            y_pos -= 0.04
        
        # Frame responsiveness
        y_pos -= 0.05
        ax2.text(0.1, y_pos, 'Frame Responsiveness:', fontsize=12, fontweight='bold')
        y_pos -= 0.05
        
        responsiveness = self.relationships['frame_responsiveness']
        sorted_frames = sorted(responsiveness.items(),
                             key=lambda x: x[1]['responsiveness_score'],
                             reverse=True)
        
        for frame, metrics in sorted_frames:
            text = f"{frame}: {metrics['responsiveness_score']:.3f} ({metrics['total_responses']} responses)"
            ax2.text(0.15, y_pos, text, fontsize=10)
            y_pos -= 0.04
        
        # Temporal patterns
        y_pos -= 0.05
        ax2.text(0.1, y_pos, 'Temporal Patterns:', fontsize=12, fontweight='bold')
        y_pos -= 0.05
        
        temporal = self.relationships['temporal_patterns']
        ax2.text(0.15, y_pos, f"Avg cascade lag: {temporal['avg_cascade_lag']:.1f} days", fontsize=10)
        y_pos -= 0.04
        ax2.text(0.15, y_pos, f"Avg trend lag: {temporal['avg_trend_lag']:.1f} days", fontsize=10)
        
        plt.suptitle('Event-Frame Relationships Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "relationship_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_focusing_events_plot(self):
        """Create focusing events visualization."""
        if self.focusing_events is None or len(self.focusing_events) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # 1. Focusing events timeline with impact
        for _, event in self.focusing_events.iterrows():
            # Get color based on event type
            color = EVENT_COLORS.get(event['event_type'], 'gray')
            
            # Ensure date is Timestamp
            peak_date = pd.Timestamp(event['peak_date'])
            
            # Plot event
            ax1.scatter(peak_date, event['focusing_strength'],
                       s=event['total_mentions'] * 2,
                       c=color, alpha=0.7,
                       edgecolors='black', linewidth=1.5)
            
            # Add label for top events
            if event['focusing_strength'] > self.focusing_events['focusing_strength'].quantile(0.9):
                ax1.annotate(
                    EventDetector.EVENT_TYPES.get(event['event_type'], '')[:10],
                    (peak_date, event['focusing_strength']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8
                )
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Focusing Strength', fontsize=12)
        ax1.set_title('Focusing Events Over Time (size = total mentions)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_minor_locator(mdates.YearLocator())
        
        # 2. Focusing event characteristics
        event_counts = self.focusing_events['event_type'].value_counts()
        
        ax2.bar(range(len(event_counts)), event_counts.values,
               color=[EVENT_COLORS.get(et, 'gray') for et in event_counts.index],
               edgecolor='black', linewidth=0.5)
        
        ax2.set_xticks(range(len(event_counts)))
        ax2.set_xticklabels([EventDetector.EVENT_TYPES.get(et, et)[:15] 
                            for et in event_counts.index],
                           rotation=45, ha='right')
        ax2.set_ylabel('Number of Focusing Events')
        ax2.set_title('Focusing Events by Type')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "focusing_events_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_patterns(self):
        """Create temporal pattern analysis."""
        if self.impact_results is None or len(self.impact_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Ensure peak_date is properly formatted
        self.impact_results['peak_date'] = pd.to_datetime(self.impact_results['peak_date'])
        
        # 1. Monthly event frequency
        ax1 = axes[0, 0]
        
        # Count events by month
        self.impact_results['month'] = self.impact_results['peak_date'].dt.to_period('M')
        monthly_counts = self.impact_results.groupby('month').size()
        
        ax1.plot(monthly_counts.index.to_timestamp(), monthly_counts.values,
                'b-', linewidth=2)
        ax1.fill_between(monthly_counts.index.to_timestamp(), 
                        monthly_counts.values, alpha=0.3)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Events')
        ax1.set_title('Monthly Event Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Impact evolution
        ax2 = axes[0, 1]
        
        monthly_impact = self.impact_results.groupby('month')['overall_impact_score'].mean()
        
        ax2.plot(monthly_impact.index.to_timestamp(), monthly_impact.values,
                'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Average Impact Score')
        ax2.set_title('Evolution of Event Impact')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cascade trigger rate
        ax3 = axes[1, 0]
        
        cascade_trigger_rate = self.impact_results.groupby('month').apply(
            lambda x: (x['cascade_impact'].apply(
                lambda y: y.get('triggered_cascades', 0) > 0).mean())
        )
        
        ax3.plot(cascade_trigger_rate.index.to_timestamp(), 
                cascade_trigger_rate.values * 100,
                'r-', linewidth=2, marker='s', markersize=4)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cascade Trigger Rate (%)')
        ax3.set_title('Percentage of Events Triggering Cascades')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. Event type evolution
        ax4 = axes[1, 1]
        
        # Stacked area chart of event types over time
        event_type_counts = self.impact_results.groupby(['month', 'event_type']).size().unstack(fill_value=0)
        
        # Convert to percentages
        event_type_pct = event_type_counts.div(event_type_counts.sum(axis=1), axis=0) * 100
        
        # Plot stacked area
        ax4.stackplot(event_type_pct.index.to_timestamp(),
                     [event_type_pct[col] for col in event_type_pct.columns],
                     labels=[EventDetector.EVENT_TYPES.get(col, col)[:10] 
                            for col in event_type_pct.columns],
                     colors=[EVENT_COLORS.get(col, 'gray') 
                            for col in event_type_pct.columns],
                     alpha=0.8)
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Percentage of Events')
        ax4.set_title('Event Type Distribution Over Time')
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax4.set_ylim(0, 100)
        
        # Format x-axes
        for ax in axes.flat:
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        plt.suptitle('Temporal Patterns in Event Impacts', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "temporal_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all analysis results."""
        print("\nSaving results...")
        
        # Save detected events
        if self.detected_events:
            all_events = []
            for event_type, events_df in self.detected_events.items():
                events_df['event_type'] = event_type
                all_events.append(events_df)
            
            if all_events:
                combined_events = pd.concat(all_events, ignore_index=True)
                # Ensure dates are saved in string format to avoid issues
                date_columns = ['start_date', 'end_date', 'peak_date']
                for col in date_columns:
                    if col in combined_events.columns:
                        combined_events[col] = combined_events[col].astype(str)
                combined_events.to_csv(DATA_DIR / "detected_events.csv", index=False)
                print(f"Saved {len(combined_events)} detected events")
        
        # Save impact results
        if self.impact_results is not None and len(self.impact_results) > 0:
            # Convert nested dictionaries to JSON strings for CSV
            impact_export = self.impact_results.copy()
            
            # Convert dates to string
            if 'peak_date' in impact_export.columns:
                impact_export['peak_date'] = impact_export['peak_date'].astype(str)
            
            for col in ['cascade_impact', 'trend_impact', 'frame_shift', 'frame_cooccurrence']:
                if col in impact_export.columns:
                    impact_export[col] = impact_export[col].apply(json.dumps)
            
            impact_export.to_csv(DATA_DIR / "event_impact_results.csv", index=False)
            print(f"Saved impact results for {len(impact_export)} events")
        
        # Save relationships
        if self.relationships:
            with open(DATA_DIR / "event_frame_relationships.json", 'w') as f:
                json.dump(self.relationships, f, indent=2, default=str)
            print("Saved event-frame relationships")
        
        # Save focusing events
        if self.focusing_events is not None and len(self.focusing_events) > 0:
            focusing_export = self.focusing_events.copy()
            
            # Convert dates to string
            if 'peak_date' in focusing_export.columns:
                focusing_export['peak_date'] = focusing_export['peak_date'].astype(str)
            
            for col in ['cascade_impact', 'trend_impact', 'frame_shift', 'frame_cooccurrence']:
                if col in focusing_export.columns:
                    focusing_export[col] = focusing_export[col].apply(json.dumps)
            
            focusing_export.to_csv(DATA_DIR / "focusing_events.csv", index=False)
            print(f"Saved {len(focusing_export)} focusing events")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        report_path = RESULTS_DIR / "event_impact_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVENT IMPACT ANALYSIS REPORT\n")
            f.write("Testing the 'Focusing Event' Hypothesis\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("HYPOTHESIS\n")
            f.write("-"*40 + "\n")
            f.write("Testing whether specific events act as 'meteorites' that create\n")
            f.write("shock waves in media coverage, triggering cascades and paradigm shifts.\n\n")
            
            # Event detection summary
            if self.detected_events:
                f.write("EVENT DETECTION RESULTS\n")
                f.write("-"*40 + "\n")
                
                total_events = sum(len(df) for df in self.detected_events.values())
                f.write(f"Total detected events: {total_events}\n\n")
                
                for event_type, events_df in self.detected_events.items():
                    event_name = EventDetector.EVENT_TYPES.get(event_type, event_type)
                    f.write(f"{event_name}: {len(events_df)} events\n")
                    
                    if len(events_df) > 0:
                        f.write(f"  Date range: {events_df['start_date'].min().strftime('%Y-%m-%d')} "
                               f"to {events_df['end_date'].max().strftime('%Y-%m-%d')}\n")
                        f.write(f"  Avg intensity: {events_df['intensity'].mean():.2f} mentions/day\n")
                        f.write(f"  Avg media coverage: {events_df['unique_media'].mean():.1f} outlets\n\n")
            
            # Impact analysis summary
            if self.impact_results is not None and len(self.impact_results) > 0:
                f.write("\nIMPACT ANALYSIS RESULTS\n")
                f.write("-"*40 + "\n")
                
                f.write(f"Events analyzed: {len(self.impact_results)}\n")
                f.write(f"Average impact score: {self.impact_results['overall_impact_score'].mean():.3f}\n")
                
                # High impact events
                high_impact = self.impact_results[self.impact_results['overall_impact_score'] > 0.5]
                f.write(f"High-impact events (>0.5): {len(high_impact)} "
                       f"({len(high_impact)/len(self.impact_results)*100:.1f}%)\n\n")
                
                # Cascade triggering
                cascade_triggers = self.impact_results['cascade_impact'].apply(
                    lambda x: x.get('triggered_cascades', 0) > 0
                ).sum()
                f.write(f"Events triggering cascades: {cascade_triggers} "
                       f"({cascade_triggers/len(self.impact_results)*100:.1f}%)\n")
                
                # Trend triggering
                trend_triggers = self.impact_results['trend_impact'].apply(
                    lambda x: x.get('triggered_trends', 0) > 0
                ).sum()
                f.write(f"Events triggering trends: {trend_triggers} "
                       f"({trend_triggers/len(self.impact_results)*100:.1f}%)\n\n")
            
            # Event-frame relationships
            if self.relationships:
                f.write("EVENT-FRAME RELATIONSHIPS\n")
                f.write("-"*40 + "\n")
                
                # Most effective event types
                effectiveness = self.relationships['event_effectiveness']
                sorted_events = sorted(effectiveness.items(),
                                     key=lambda x: x[1]['avg_impact_score'],
                                     reverse=True)
                
                f.write("Most effective event types:\n")
                for event_type, metrics in sorted_events[:5]:
                    event_name = EventDetector.EVENT_TYPES.get(event_type, event_type)
                    f.write(f"  {event_name}: impact={metrics['avg_impact_score']:.3f}, "
                           f"cascade_rate={metrics['cascade_trigger_rate']:.2f}, "
                           f"n={metrics['n_events']}\n")
                
                # Most responsive frames
                f.write("\nMost responsive frames:\n")
                responsiveness = self.relationships['frame_responsiveness']
                sorted_frames = sorted(responsiveness.items(),
                                     key=lambda x: x[1]['responsiveness_score'],
                                     reverse=True)
                
                for frame, metrics in sorted_frames:
                    f.write(f"  {frame}: score={metrics['responsiveness_score']:.3f}, "
                           f"responses={metrics['total_responses']}\n")
                
                # Temporal patterns
                f.write("\nTemporal patterns:\n")
                temporal = self.relationships['temporal_patterns']
                f.write(f"  Average cascade lag: {temporal['avg_cascade_lag']:.1f} days\n")
                f.write(f"  Average trend lag: {temporal['avg_trend_lag']:.1f} days\n")
                f.write(f"  Lag correlation: {temporal['lag_correlation']:.3f}\n\n")
            
            # Focusing events
            if self.focusing_events is not None and len(self.focusing_events) > 0:
                f.write("FOCUSING EVENTS\n")
                f.write("-"*40 + "\n")
                f.write(f"Identified {len(self.focusing_events)} focusing events\n\n")
                
                f.write("Top 10 focusing events:\n")
                for idx, event in self.focusing_events.head(10).iterrows():
                    event_name = EventDetector.EVENT_TYPES.get(event['event_type'], event['event_type'])
                    f.write(f"\n{idx+1}. {event['peak_date'].strftime('%Y-%m-%d')} - {event_name}\n")
                    f.write(f"   Focusing strength: {event['focusing_strength']:.3f}\n")
                    f.write(f"   Overall impact: {event['overall_impact_score']:.3f}\n")
                    f.write(f"   Triggered cascades: {event['cascade_impact']['triggered_cascades']}\n")
                    f.write(f"   Triggered trends: {event['trend_impact']['triggered_trends']}\n")
                    f.write(f"   Affected frames: {', '.join(event['cascade_impact']['affected_frames'])}\n")
            
            # Conclusions
            f.write("\n\nCONCLUSIONS\n")
            f.write("-"*40 + "\n")
            
            if self.impact_results is not None and len(self.impact_results) > 0:
                impact_rate = (self.impact_results['overall_impact_score'] > 0.5).mean()
                
                if impact_rate > 0.3:
                    f.write("- Strong evidence for the 'focusing event' hypothesis: ")
                    f.write(f"{impact_rate*100:.1f}% of detected events show significant impact.\n")
                elif impact_rate > 0.15:
                    f.write("- Moderate evidence for the 'focusing event' hypothesis: ")
                    f.write(f"{impact_rate*100:.1f}% of detected events show significant impact.\n")
                else:
                    f.write("- Limited evidence for the 'focusing event' hypothesis: ")
                    f.write(f"Only {impact_rate*100:.1f}% of detected events show significant impact.\n")
                
                if self.focusing_events is not None and len(self.focusing_events) > 0:
                    f.write(f"\n- {len(self.focusing_events)} events qualify as true 'focusing events' ")
                    f.write("that trigger cascading changes in media coverage.\n")
                    
                    # Most common focusing event types
                    event_type_counts = self.focusing_events['event_type'].value_counts()
                    if len(event_type_counts) > 0:
                        top_type = event_type_counts.index[0]
                        top_name = EventDetector.EVENT_TYPES.get(top_type, top_type)
                        f.write(f"\n- {top_name} events are the most common focusing events ")
                        f.write(f"({event_type_counts.iloc[0]} occurrences).\n")
                
                # Temporal insights
                if 'avg_cascade_lag' in self.relationships.get('temporal_patterns', {}):
                    avg_lag = self.relationships['temporal_patterns']['avg_cascade_lag']
                    f.write(f"\n- Media cascades typically begin {avg_lag:.1f} days after ")
                    f.write("the triggering event.\n")
        
        print(f"Report saved to {report_path}")
    
    def run_analysis(self):
        """Run complete event impact analysis."""
        print("="*80)
        print("EVENT IMPACT ANALYSIS")
        print("="*80)
        
        # Load data
        df = self.load_event_data()
        cascade_results, trend_results = self.load_previous_results()
        
        # Detect events
        self.detect_events(df)
        
        # Analyze impacts
        self.analyze_impacts(df, cascade_results, trend_results)
        
        # Analyze relationships
        self.analyze_relationships()
        
        # Identify focusing events
        self.identify_focusing_events()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        # Generate report
        self.generate_report()
        
        # Close database connection
        self.db_connector.close()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        if self.focusing_events is not None and len(self.focusing_events) > 0:
            print(f"\nKey finding: Identified {len(self.focusing_events)} focusing events")
            print("that act as 'meteorites' triggering paradigm shifts.")


def main():
    """Main execution function."""
    analysis = EventImpactAnalysis()
    analysis.run_analysis()


if __name__ == "__main__":
    main()