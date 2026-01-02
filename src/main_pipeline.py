"""
Main Pipeline - Complete integration of all matching components
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

import config
from schema_profiler import SchemaProfiler
from embedding_matcher import EmbeddingMatcher
from rule_matcher import RuleMatcher
from confidence_scorer import ConfidenceScorer, MatchResult


class MasterDataPipeline:
    """Main pipeline for schema matching and data integration"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the pipeline
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.profiler = SchemaProfiler()
        self.embedding_matcher = EmbeddingMatcher(
            model_name=config.EMBEDDING_MODEL,
            cache_dir=str(config.CACHE_DIR)
        )
        self.rule_matcher = RuleMatcher()
        self.scorer = ConfidenceScorer()
        
        # Initialize master embeddings
        self._initialize_master_embeddings()
    
    def _initialize_master_embeddings(self):
        """Initialize or load master schema embeddings"""
        if self.verbose:
            print("Initializing master schema embeddings...")
        
        # Try to load from cache
        if not self.embedding_matcher.load_master_embeddings():
            # Build new embeddings
            self.embedding_matcher.build_master_embeddings(config.MASTER_SCHEMA)
        
        if self.verbose:
            print("✓ Master embeddings ready\n")
    
    def process_source_data(
        self,
        data: pd.DataFrame,
        source_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Process source data through complete pipeline
        
        Args:
            data: Source DataFrame
            source_name: Name/identifier for the source
            
        Returns:
            Dictionary containing all analysis results
        """
        if self.verbose:
            print("=" * 80)
            print(f"PROCESSING SOURCE: {source_name}")
            print("=" * 80)
            print(f"Rows: {len(data)}, Columns: {len(data.columns)}")
            print()
        
        # Step 1: Profile schema
        if self.verbose:
            print("Step 1: Profiling schema...")
        profile = self.profiler.profile(data)
        
        # Step 2: Embedding-based matching
        if self.verbose:
            print("Step 2: Semantic matching (embeddings)...")
        embedding_matches = self.embedding_matcher.batch_match(
            profile['columns'],
            threshold=0.0,  # Get all matches for scoring
            top_k=3
        )
        
        # Step 3: Rule-based matching
        if self.verbose:
            print("Step 3: Rule-based matching...")
        rule_matches = self.rule_matcher.batch_match(
            profile['columns'],
            top_k=3
        )
        
        # Step 4: Combined scoring
        if self.verbose:
            print("Step 4: Calculating combined confidence scores...")
        scoring_results = self.scorer.batch_score(
            embedding_matches,
            rule_matches,
            profile['columns']
        )
        
        # Step 5: Get best matches
        best_matches = self.scorer.get_best_matches(
            scoring_results,
            min_score=config.CONFIDENCE_THRESHOLDS['reject']
        )
        
        # Step 6: Identify unmapped columns
        unmapped = self.scorer.get_unmapped_columns(
            list(data.columns),
            best_matches
        )
        
        if self.verbose:
            print("✓ Processing complete\n")
        
        return {
            'source_name': source_name,
            'profile': profile,
            'embedding_matches': embedding_matches,
            'rule_matches': rule_matches,
            'scoring_results': scoring_results,
            'best_matches': best_matches,
            'unmapped_columns': unmapped,
            'timestamp': datetime.now().isoformat()
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results"""
        print("\n" + "=" * 80)
        print(f"RESULTS FOR: {results['source_name']}")
        print("=" * 80)
        
        # Profile summary
        profile = results['profile']
        print(f"\nDataset: {profile['row_count']} rows, {profile['column_count']} columns")
        
        # Mapping report
        print("\n" + self.scorer.generate_mapping_report(results['best_matches']))
        
        # Unmapped columns
        if results['unmapped_columns']:
            print("\n⚠ UNMAPPED COLUMNS:")
            print("-" * 80)
            for col in results['unmapped_columns']:
                print(f"  • {col}")
            print("\nThese columns have no acceptable mapping and may need manual review.")
        
        # Column details
        print("\n" + "=" * 80)
        print("DETAILED COLUMN ANALYSIS")
        print("=" * 80)
        
        for source_col, match in results['best_matches'].items():
            col_profile = profile['columns'][source_col]
            print(f"\n{source_col} → {match.master_column}")
            print(f"  Semantic Type: {col_profile['semantic_type']}")
            print(f"  NULL%: {col_profile['null_percentage']:.1f}%")
            print(f"  Unique: {col_profile['unique_count']}")
            print(f"  Scores: emb={match.embedding_score:.3f}, "
                  f"rule={match.rule_score:.3f}, "
                  f"quality={match.quality_score:.3f}")
            print(f"  Combined: {match.combined_score:.3f}")
            print(f"  Decision: {match.recommendation}")
    
    def export_mapping(
        self,
        results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Dict[str, str]:
        """
        Export column mapping as dictionary
        
        Args:
            results: Results from process_source_data
            output_path: Optional path to save JSON
            
        Returns:
            Dictionary of source_column -> master_column
        """
        mapping = {}
        
        for source_col, match in results['best_matches'].items():
            if "AUTO_ACCEPT" in match.recommendation:
                mapping[source_col] = match.master_column
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(mapping, f, indent=2)
            if self.verbose:
                print(f"\n✓ Mapping exported to {output_path}")
        
        return mapping
    
    def apply_mapping(
        self,
        data: pd.DataFrame,
        mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Apply column mapping to transform data
        
        Args:
            data: Source DataFrame
            mapping: Dict of source_column -> master_column
            
        Returns:
            Transformed DataFrame with master column names
        """
        # Rename columns according to mapping
        renamed_data = data.rename(columns=mapping)
        
        # Keep only mapped columns that are in master schema
        master_columns = list(config.MASTER_SCHEMA.keys())
        available_columns = [col for col in master_columns if col in renamed_data.columns]
        
        return renamed_data[available_columns]
    
    def generate_comparison_report(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any]
    ) -> str:
        """Compare results from two different sources"""
        report = []
        report.append("=" * 80)
        report.append("SOURCE COMPARISON REPORT")
        report.append("=" * 80)
        
        # Basic stats
        report.append(f"\nSource 1: {results1['source_name']}")
        report.append(f"  Rows: {results1['profile']['row_count']}")
        report.append(f"  Mapped: {len(results1['best_matches'])}")
        report.append(f"  Unmapped: {len(results1['unmapped_columns'])}")
        
        report.append(f"\nSource 2: {results2['source_name']}")
        report.append(f"  Rows: {results2['profile']['row_count']}")
        report.append(f"  Mapped: {len(results2['best_matches'])}")
        report.append(f"  Unmapped: {len(results2['unmapped_columns'])}")
        
        # Common mappings
        mapped1 = set(m.master_column for m in results1['best_matches'].values())
        mapped2 = set(m.master_column for m in results2['best_matches'].values())
        
        common = mapped1 & mapped2
        only1 = mapped1 - mapped2
        only2 = mapped2 - mapped1
        
        report.append(f"\nCommon master columns: {len(common)}")
        report.append(f"Only in source 1: {len(only1)}")
        report.append(f"Only in source 2: {len(only2)}")
        
        if common:
            report.append(f"\nCommon columns: {', '.join(sorted(common))}")
        
        return "\n".join(report)


# Example usage and demonstration
if __name__ == "__main__":
    print("MasterData Pipeline - Complete Demo")
    print("=" * 80)
    print()
    
    # Create three different source datasets
    source1 = pd.DataFrame({
        'cust_id': [101, 102, 103, 104, 105],
        'customer_name': ['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown', 'Eve Davis'],
        'email': ['alice@email.com', 'bob@email.com', None, 'david@email.com', 'eve@email.com'],
        'phone': ['+1-555-0101', '+1-555-0102', '+1-555-0103', None, '+1-555-0105'],
        'signup_date': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05', '2024-05-12']),
        'purchase_amount': [1250.50, 890.25, 2100.00, 450.75, 1680.30],
        'status': ['active', 'active', 'inactive', 'active', 'active']
    })
    
    source2 = pd.DataFrame({
        'ID': [201, 202, 203, 204],
        'Name': ['Frank Wilson', 'Grace Lee', None, 'Iris Martinez'],
        'Email_Address': ['frank@test.com', None, 'henry@test.com', 'iris@test.com'],
        'Contact_Phone': [None, '+1-555-0202', '+1-555-0203', '+1-555-0204'],
        'Reg_Date': pd.to_datetime(['2024-06-01', '2024-06-15', '2024-07-20', '2024-08-10']),
        'Total_Spend': [None, 1200.00, 750.50, 920.25],
        'Acct_Status': ['active', 'active', 'suspended', 'active']
    })
    
    # Initialize pipeline
    pipeline = MasterDataPipeline(verbose=True)
    
    # Process both sources
    results1 = pipeline.process_source_data(source1, "E-commerce Platform")
    results2 = pipeline.process_source_data(source2, "CRM System")
    
    # Print results
    pipeline.print_results(results1)
    print("\n\n")
    pipeline.print_results(results2)
    
    # Export mappings
    mapping1 = pipeline.export_mapping(results1)
    print(f"\nGenerated mapping for {results1['source_name']}:")
    print(json.dumps(mapping1, indent=2))
    
    # Apply mapping and show result
    transformed1 = pipeline.apply_mapping(source1, mapping1)
    print(f"\n✓ Transformed {results1['source_name']} to master schema")
    print(f"Columns: {list(transformed1.columns)}")
    print(f"\nFirst 3 rows:")
    print(transformed1.head(3))
    
    # Comparison report
    print("\n\n")
    print(pipeline.generate_comparison_report(results1, results2))
