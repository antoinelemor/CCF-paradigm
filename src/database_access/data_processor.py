"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
data_processor.py

MAIN OBJECTIVE:
---------------
This script provides parallelized data processing functions for frame detection data,
optimized for Apple Silicon M4 Max with efficient memory management and multiprocessing.
Automatically excludes data from 2025 onwards.

Dependencies:
-------------
- pandas
- numpy
- multiprocessing
- concurrent.futures
- functools
- tqdm

MAIN FEATURES:
--------------
1) Parallel processing of frame data using multiprocessing
2) Efficient memory management for large datasets
3) Batch processing with configurable chunk sizes
4) Frame proportion calculations at different temporal resolutions
5) Automatic exclusion of 2025 data
6) Optimized for Apple Silicon architecture
7) Progress tracking with tqdm

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import logging
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FrameDataProcessor:
    """Processes frame detection data with parallel computing capabilities."""
    
    # Frame column names in CCF_Database_texts
    FRAME_COLUMNS = [
        "cultural_frame", "economic_frame", "environmental_frame",
        "health_frame", "justice_frame", "political_frame",
        "scientific_frame", "security_frame"
    ]

    # Short codes used for output DataFrames
    FRAME_CODES = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]

    # DB column -> short code mapping
    FRAME_COL_TO_CODE = {
        "cultural_frame": "Cult",
        "economic_frame": "Eco",
        "environmental_frame": "Envt",
        "health_frame": "Pbh",
        "justice_frame": "Just",
        "political_frame": "Pol",
        "scientific_frame": "Sci",
        "security_frame": "Secu",
    }

    # Frame full names mapping
    FRAME_NAMES = {
        "Cult": "Culture",
        "Eco": "Economy",
        "Envt": "Environment",
        "Pbh": "Health",
        "Just": "Justice",
        "Pol": "Politics",
        "Sci": "Science",
        "Secu": "Security",
    }
    
    def __init__(self, n_workers: Optional[int] = None, exclude_2025: bool = True):
        """
        Initialize processor with specified number of workers.
        
        Args:
            n_workers: Number of parallel workers (None for auto-detect)
            exclude_2025: Whether to automatically exclude data from 2025 onwards
        """
        self.n_workers = n_workers or min(cpu_count(), 8)  # Cap at 8 for efficiency
        self.exclude_2025 = exclude_2025
        logger.info(f"Initialized processor with {self.n_workers} workers")
        if self.exclude_2025:
            logger.info("Data from 2025 onwards will be automatically excluded")
    
    def process_frame_data(self, df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """
        Process raw frame data to ensure correct types and clean values.
        
        Args:
            df: Raw DataFrame from database
            show_progress: Whether to show progress bar
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        if show_progress:
            pbar = tqdm(total=4, desc="Processing frame data")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if show_progress:
            pbar.update(1)
        
        # Exclude 2025 data if requested
        if self.exclude_2025:
            initial_rows = len(df)
            df = df[df['date'].dt.year < 2025]
            excluded_rows = initial_rows - len(df)
            if excluded_rows > 0:
                logger.info(f"Excluded {excluded_rows:,} rows from 2025 onwards")
        if show_progress:
            pbar.update(1)
        
        # Get frame columns (new names from CCF_Database_texts)
        frame_cols = [col for col in self.FRAME_COLUMNS if col in df.columns]

        # Convert frame columns to numeric
        for col in frame_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if show_progress:
            pbar.update(1)
        
        # Filter valid sentences
        df = df[df['sentences'].notna() & (df['sentences'].str.strip() != '')]
        if show_progress:
            pbar.update(1)
            pbar.close()
        
        return df
    
    def calculate_article_proportions(self, df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
        """
        Calculate frame proportions at article level.
        
        Args:
            df: Processed DataFrame
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with article-level proportions
        """
        frame_cols = [col for col in self.FRAME_COLUMNS if col in df.columns]

        if show_progress:
            tqdm.pandas(desc="Calculating article proportions")
            article_groups = df.groupby('doc_id', progress=True)
        else:
            article_groups = df.groupby('doc_id')
        
        # Calculate number of sentences per article
        n_sentences = article_groups['sentence_id'].nunique()
        
        # Sum detections per article
        frame_sums = article_groups[frame_cols].sum()
        
        # Calculate proportions
        proportions = frame_sums.div(n_sentences, axis=0)
        
        # Add metadata
        metadata = article_groups[['date', 'media']].first()
        
        # Ensure no 2025 data in results
        if self.exclude_2025:
            metadata = metadata[metadata['date'].dt.year < 2025]
            proportions = proportions.loc[metadata.index]
        
        return pd.concat([metadata, proportions], axis=1).reset_index()
    
    def _process_chunk(self, chunk: pd.DataFrame, 
                      temporal_resolution: str) -> pd.DataFrame:
        """
        Process a chunk of data for parallel execution.
        
        Args:
            chunk: DataFrame chunk
            temporal_resolution: 'week', 'month', or 'year'
            
        Returns:
            Aggregated DataFrame
        """
        # Calculate article proportions
        article_props = self.calculate_article_proportions(chunk)
        
        # Create temporal grouping
        if temporal_resolution == 'week':
            article_props['period'] = article_props['date'].dt.to_period('W').dt.to_timestamp()
        elif temporal_resolution == 'month':
            article_props['period'] = article_props['date'].dt.to_period('M').dt.to_timestamp()
        else:  # year
            article_props['period'] = article_props['date'].dt.to_period('Y').dt.to_timestamp()
        
        # Get frame columns
        frame_cols = [col for col in article_props.columns if col.endswith('_frame')]

        # Aggregate by period
        return article_props.groupby('period')[frame_cols].mean().reset_index()
    
    def aggregate_frames_parallel(self, df: pd.DataFrame, 
                                temporal_resolution: str = 'week',
                                chunk_size: int = 50000,
                                show_progress: bool = True) -> pd.DataFrame:
        """
        Aggregate frame proportions at specified temporal resolution using parallel processing.
        
        Args:
            df: Processed DataFrame
            temporal_resolution: 'week', 'month', or 'year'
            chunk_size: Size of chunks for parallel processing
            show_progress: Whether to show progress bar
            
        Returns:
            Aggregated DataFrame
        """
        # Split data into chunks by doc_id to maintain article integrity
        unique_docs = df['doc_id'].unique()
        n_chunks = max(1, len(unique_docs) // chunk_size)
        doc_chunks = np.array_split(unique_docs, n_chunks)
        
        # Create data chunks
        data_chunks = [df[df['doc_id'].isin(docs)] for docs in doc_chunks]
        
        # Process chunks in parallel
        process_func = partial(self._process_chunk, temporal_resolution=temporal_resolution)
        
        results = []
        
        if show_progress:
            pbar = tqdm(total=len(data_chunks), desc=f"Aggregating by {temporal_resolution}")
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_chunk = {executor.submit(process_func, chunk): i 
                             for i, chunk in enumerate(data_chunks)}
            
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    results.append(result)
                    if show_progress:
                        pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    if show_progress:
                        pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # Combine results
        combined = pd.concat(results, ignore_index=True)
        
        # Final aggregation (in case chunks had overlapping periods)
        frame_cols = [col for col in combined.columns if col.endswith('_frame')]
        final_result = combined.groupby('period')[frame_cols].mean().reset_index()
        
        return final_result.sort_values('period')
    
    def calculate_frame_counts(self, df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate raw frame counts for diversity analysis.
        
        Args:
            df: Processed DataFrame
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with frame counts per period
        """
        frame_cols = [col for col in self.FRAME_COLUMNS if col in df.columns]

        if show_progress:
            pbar = tqdm(total=3, desc="Calculating frame counts")

        # Create weekly periods
        df_copy = df.copy()
        df_copy['week'] = df_copy['date'].dt.to_period('W').dt.to_timestamp()

        # Exclude 2025 data
        if self.exclude_2025:
            df_copy = df_copy[df_copy['date'].dt.year < 2025]
        if show_progress:
            pbar.update(1)

        # Ensure frame columns are numeric
        for col in frame_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(float)
        if show_progress:
            pbar.update(1)

        # Sum detections by week
        weekly_counts = df_copy.groupby('week')[frame_cols].sum().reset_index()

        # Rename columns from DB names to short codes
        weekly_counts = weekly_counts.rename(columns=self.FRAME_COL_TO_CODE)

        # Ensure all values are float
        for code in self.FRAME_CODES:
            if code in weekly_counts.columns:
                weekly_counts[code] = weekly_counts[code].astype(float)
        
        if show_progress:
            pbar.update(1)
            pbar.close()
        
        return weekly_counts
    
    def prepare_diversity_data(self, df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """
        Prepare data specifically for diversity analysis.
        
        Args:
            df: Raw DataFrame from database
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame ready for diversity calculations
        """
        if show_progress:
            print("Preparing data for diversity analysis...")
        
        # Process data
        processed = self.process_frame_data(df, show_progress=show_progress)
        
        # Get weekly counts
        weekly_counts = self.calculate_frame_counts(processed, show_progress=show_progress)
        
        return weekly_counts


def parallel_frame_aggregation(df: pd.DataFrame, 
                             temporal_resolution: str = 'week',
                             n_workers: Optional[int] = None,
                             exclude_2025: bool = True,
                             show_progress: bool = True) -> pd.DataFrame:
    """
    Convenience function for parallel frame aggregation.
    
    Args:
        df: Input DataFrame
        temporal_resolution: Time resolution for aggregation
        n_workers: Number of parallel workers
        exclude_2025: Whether to exclude 2025 data
        show_progress: Whether to show progress bar
        
    Returns:
        Aggregated DataFrame
    """
    processor = FrameDataProcessor(n_workers, exclude_2025=exclude_2025)
    processed = processor.process_frame_data(df, show_progress=show_progress)
    return processor.aggregate_frames_parallel(processed, temporal_resolution, show_progress=show_progress)