"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
explore_database_structure.py

MAIN OBJECTIVE:
---------------
This script explores and documents the database structure to understand table schemas,
column types, data distribution, and sample data for debugging purposes.

Dependencies:
-------------
- pandas
- psycopg2-binary
- sqlalchemy
- tabulate

MAIN FEATURES:
--------------
1) List all tables in the database
2) Show column information for CCF_processed_data
3) Display data types and constraints
4) Show sample data
5) Analyze frame detection columns
6) Export structure report

Author:
-------
Antoine Lemor
"""

import os
import sys
from pathlib import Path
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, inspect
from tabulate import tabulate
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


class DatabaseExplorer:
    """Explore and document database structure."""
    
    def __init__(self):
        """Initialize database connection."""
        self.db_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "dbname": os.getenv("DB_NAME", "CCF"),
            "user": os.getenv("DB_USER", "antoine"),
            "password": os.getenv("PGPASSWORD", ""),
        }
        
        # Create connection string
        self.connection_string = (
            f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
            f"@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
        )
        
        self.engine = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "database": self.db_params['dbname'],
            "tables": {},
            "analysis": {}
        }
    
    def connect(self):
        """Establish database connection."""
        try:
            self.engine = create_engine(self.connection_string)
            print(f"✓ Connected to database: {self.db_params['dbname']}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def list_tables(self):
        """List all tables in the database."""
        print("\n" + "="*60)
        print("DATABASE TABLES")
        print("="*60)
        
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
        """
        
        df = pd.read_sql(query, self.engine)
        tables = df['table_name'].tolist()
        
        print(f"\nFound {len(tables)} tables:")
        for i, table in enumerate(tables, 1):
            print(f"  {i}. {table}")
        
        self.results['tables']['list'] = tables
        return tables
    
    def explore_table_structure(self, table_name="CCF_processed_data"):
        """Explore structure of a specific table."""
        print("\n" + "="*60)
        print(f"TABLE STRUCTURE: {table_name}")
        print("="*60)
        
        # Get column information
        query = f"""
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position;
        """
        
        columns_df = pd.read_sql(query, self.engine)
        
        print("\nColumns:")
        print(tabulate(columns_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Store in results
        self.results['tables'][table_name] = {
            'columns': columns_df.to_dict('records'),
            'total_columns': len(columns_df)
        }
        
        return columns_df
    
    def analyze_frame_columns(self, table_name="CCF_processed_data"):
        """Analyze frame detection columns specifically."""
        print("\n" + "="*60)
        print("FRAME DETECTION COLUMNS")
        print("="*60)
        
        # Get all columns ending with _Detection
        query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %(table_name)s 
        AND column_name LIKE '%%_Detection'
        ORDER BY column_name;
        """
        
        frame_cols_df = pd.read_sql(query, self.engine, params={"table_name": table_name})
        frame_columns = frame_cols_df['column_name'].tolist()
        
        print(f"\nFound {len(frame_columns)} frame detection columns:")
        
        # Separate main frames from others
        main_frames = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]
        main_frame_cols = []
        other_frame_cols = []
        
        for col in frame_columns:
            frame_name = col.replace('_Detection', '')
            if frame_name in main_frames:
                main_frame_cols.append(col)
                print(f"  - {col} (Main Frame: {frame_name})")
            else:
                other_frame_cols.append(col)
        
        if other_frame_cols:
            print(f"\nOther detection columns (not used for paradigm analysis):")
            for col in other_frame_cols:
                print(f"  - {col}")
        
        # Get emotion columns
        query_emotions = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %(table_name)s 
        AND column_name LIKE 'Emotion:%%'
        ORDER BY column_name;
        """
        
        emotion_cols_df = pd.read_sql(query_emotions, self.engine, params={"table_name": table_name})
        emotion_columns = emotion_cols_df['column_name'].tolist()
        
        print(f"\nFound {len(emotion_columns)} emotion columns:")
        for col in emotion_columns:
            print(f"  - {col}")
        
        self.results['analysis']['main_frame_columns'] = main_frame_cols
        self.results['analysis']['other_detection_columns'] = other_frame_cols
        self.results['analysis']['emotion_columns'] = emotion_columns
        
        return main_frame_cols, emotion_columns
    
    def get_sample_data(self, table_name="CCF_processed_data", limit=5):
        """Get sample data from the table."""
        print("\n" + "="*60)
        print(f"SAMPLE DATA FROM {table_name}")
        print("="*60)
        
        # Get first few rows
        query = f'SELECT * FROM "{table_name}" LIMIT {limit};'
        sample_df = pd.read_sql(query, self.engine)
        
        print(f"\nFirst {limit} rows:")
        print(sample_df.to_string())
        
        # Store sample in results
        self.results['tables'][table_name]['sample'] = sample_df.to_dict('records')
        
        return sample_df
    
    def analyze_data_distribution(self, table_name="CCF_processed_data"):
        """Analyze data distribution and statistics."""
        print("\n" + "="*60)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Count total rows
        count_query = f'SELECT COUNT(*) as total_rows FROM "{table_name}";'
        total_rows = pd.read_sql(count_query, self.engine).iloc[0]['total_rows']
        print(f"\nTotal rows: {total_rows:,}")
        
        # Count unique documents
        doc_query = f'SELECT COUNT(DISTINCT doc_id) as unique_docs FROM "{table_name}";'
        unique_docs = pd.read_sql(doc_query, self.engine).iloc[0]['unique_docs']
        print(f"Unique documents: {unique_docs:,}")
        
        # Date range
        date_query = f"""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM "{table_name}"
        WHERE date IS NOT NULL;
        """
        date_range = pd.read_sql(date_query, self.engine).iloc[0]
        print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")
        
        # Media distribution
        media_query = f"""
        SELECT 
            media,
            COUNT(DISTINCT doc_id) as n_documents,
            COUNT(*) as n_sentences
        FROM "{table_name}"
        GROUP BY media
        ORDER BY n_documents DESC
        LIMIT 10;
        """
        media_dist = pd.read_sql(media_query, self.engine)
        
        print("\nTop 10 media outlets:")
        print(tabulate(media_dist, headers='keys', tablefmt='grid', showindex=False))
        
        # Frame detection statistics
        main_frame_cols, _ = self.analyze_frame_columns(table_name)
        
        if main_frame_cols:
            print("\nFrame detection statistics (8 main frames only):")
            frame_stats_query = f"""
            SELECT 
                {', '.join([f'SUM(CASE WHEN "{col}" = 1 THEN 1 ELSE 0 END) as {col.replace("_Detection", "")}' for col in main_frame_cols])}
            FROM "{table_name}";
            """
            frame_stats = pd.read_sql(frame_stats_query, self.engine)
            
            # Transpose for better display
            frame_stats_t = frame_stats.T
            frame_stats_t.columns = ['Detections']
            frame_stats_t['Percentage'] = (frame_stats_t['Detections'] / total_rows * 100).round(2)
            
            print(tabulate(frame_stats_t, headers='keys', tablefmt='grid'))
        
        # Store analysis results
        self.results['analysis']['statistics'] = {
            'total_rows': int(total_rows),
            'unique_documents': int(unique_docs),
            'date_range': {
                'min': str(date_range['min_date']),
                'max': str(date_range['max_date'])
            },
            'media_distribution': media_dist.to_dict('records')
        }
    
    def check_data_types(self, table_name="CCF_processed_data"):
        """Check actual data types of frame columns."""
        print("\n" + "="*60)
        print("CHECKING DATA TYPES")
        print("="*60)
        
        # Get a sample to check data types
        query = f"""
        SELECT *
        FROM "{table_name}"
        WHERE doc_id IS NOT NULL
        LIMIT 1000;
        """
        
        sample_df = pd.read_sql(query, self.engine)
        
        # Check frame columns - only the 8 main frames
        main_frames = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]
        frame_cols = [f"{frame}_Detection" for frame in main_frames if f"{frame}_Detection" in sample_df.columns]
        
        print("\nMain frame column data types and unique values:")
        for col in frame_cols:
            dtype = sample_df[col].dtype
            unique_vals = sample_df[col].dropna().unique()
            null_count = sample_df[col].isna().sum()
            
            print(f"\n{col}:")
            print(f"  - Data type: {dtype}")
            print(f"  - Unique values: {sorted(unique_vals)}")
            print(f"  - Null count: {null_count}/{len(sample_df)}")
            print(f"  - Value counts:")
            value_counts = sample_df[col].value_counts().to_dict()
            for val, count in value_counts.items():
                print(f"    {val}: {count}")
    
    def save_report(self):
        """Save exploration report."""
        report_path = project_root / "results" / "database_structure_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n✓ Report saved to: {report_path}")
        
        # Also save a human-readable summary
        summary_path = project_root / "results" / "database_structure_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("DATABASE STRUCTURE SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Database: {self.results['database']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")
            
            f.write("TABLES:\n")
            for table in self.results['tables'].get('list', []):
                f.write(f"  - {table}\n")
            
            f.write(f"\nFRAME COLUMNS ({len(self.results['analysis'].get('main_frame_columns', []))}):\n")
            for col in self.results['analysis'].get('main_frame_columns', []):
                f.write(f"  - {col}\n")
            
            f.write(f"\nEMOTION COLUMNS ({len(self.results['analysis'].get('emotion_columns', []))}):\n")
            for col in self.results['analysis'].get('emotion_columns', []):
                f.write(f"  - {col}\n")
            
        if 'statistics' in self.results['analysis']:
                stats = self.results['analysis']['statistics']
                f.write(f"\nSTATISTICS:\n")
                f.write(f"  - Total rows: {stats['total_rows']:,}\n")
                f.write(f"  - Unique documents: {stats['unique_documents']:,}\n")
                f.write(f"  - Date range: {stats['date_range']['min']} to {stats['date_range']['max']}\n")
        
        print(f"✓ Summary saved to: {summary_path}")
    
    def run_exploration(self):
        """Run complete database exploration."""
        print("="*60)
        print("DATABASE STRUCTURE EXPLORATION")
        print("="*60)
        
        if not self.connect():
            return
        
        # List all tables
        tables = self.list_tables()
        
        # Focus on CCF_processed_data
        if "CCF_processed_data" in tables:
            # Explore table structure
            self.explore_table_structure("CCF_processed_data")
            
            # Analyze frame columns
            main_frame_cols, emotion_cols = self.analyze_frame_columns("CCF_processed_data")
            
            # Get sample data
            self.get_sample_data("CCF_processed_data", limit=3)
            
            # Analyze distribution
            self.analyze_data_distribution("CCF_processed_data")
            
            # Check data types
            self.check_data_types("CCF_processed_data")
        else:
            print("\n⚠ Warning: CCF_processed_data table not found!")
        
        # Save report
        self.save_report()
        
        print("\n" + "="*60)
        print("EXPLORATION COMPLETE")
        print("="*60)


def main():
    """Run database exploration."""
    explorer = DatabaseExplorer()
    explorer.run_exploration()


if __name__ == "__main__":
    main()