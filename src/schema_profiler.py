"""
Schema Profiler - Analyzes uploaded datasets and extracts metadata
"""
import pandas as pd
import re
from typing import Dict, Any, List
from collections import Counter


class SchemaProfiler:
    """Profiles dataset schemas and generates metadata"""
    
    # Common regex patterns for data type detection
    PATTERNS = {
        'email': r'^[\w\.-]+@[\w\.-]+\.\w+$',
        'phone': r'^\+?[\d\s\-\(\)]{10,}$',
        'url': r'^https?://[\w\.-]+\.\w+',
        'date': r'^\d{4}-\d{2}-\d{2}',
        'zip_code': r'^\d{5}(-\d{4})?$',
    }
    
    def __init__(self, sample_size: int = 100):
        """
        Args:
            sample_size: Number of rows to sample for pattern detection
        """
        self.sample_size = sample_size
    
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive profile of a dataframe
        
        Args:
            df: Input dataframe to profile
            
        Returns:
            Dictionary containing schema profile
        """
        profile = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': {}
        }
        
        for col in df.columns:
            profile['columns'][col] = self._profile_column(df, col)
        
        return profile
    
    def _profile_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Profile individual column"""
        series = df[col]
        total_rows = len(series)
        non_null_series = series.dropna()
        
        column_profile = {
            'dtype': str(series.dtype),
            'null_count': series.isna().sum(),
            'null_percentage': (series.isna().sum() / total_rows) * 100,
            'unique_count': series.nunique(),
            'cardinality_ratio': series.nunique() / total_rows if total_rows > 0 else 0,
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            column_profile.update({
                'min': non_null_series.min() if len(non_null_series) > 0 else None,
                'max': non_null_series.max() if len(non_null_series) > 0 else None,
                'mean': non_null_series.mean() if len(non_null_series) > 0 else None,
                'median': non_null_series.median() if len(non_null_series) > 0 else None,
            })
        
        # Sample values for pattern detection
        sample_values = non_null_series.sample(
            min(self.sample_size, len(non_null_series))
        ).astype(str).tolist() if len(non_null_series) > 0 else []
        
        column_profile['sample_values'] = sample_values[:10]  # Store first 10
        
        # Detect patterns
        column_profile['detected_patterns'] = self._detect_patterns(sample_values)
        
        # Infer semantic type
        column_profile['semantic_type'] = self._infer_semantic_type(
            col, 
            series.dtype, 
            column_profile['detected_patterns']
        )
        
        return column_profile
    
    def _detect_patterns(self, values: List[str]) -> Dict[str, float]:
        """
        Detect common patterns in sample values
        
        Returns:
            Dictionary of pattern_name -> match_percentage
        """
        if not values:
            return {}
        
        pattern_matches = {}
        total_values = len(values)
        
        for pattern_name, pattern in self.PATTERNS.items():
            matches = sum(1 for v in values if re.match(pattern, str(v)))
            if matches > 0:
                pattern_matches[pattern_name] = (matches / total_values) * 100
        
        return pattern_matches
    
    def _infer_semantic_type(
        self, 
        col_name: str, 
        dtype: Any, 
        patterns: Dict[str, float]
    ) -> str:
        """
        Infer semantic meaning of column based on name, type, and patterns
        
        Returns:
            Semantic type string (e.g., 'identifier', 'email', 'numeric')
        """
        col_lower = col_name.lower()
        
        # Check patterns first (high confidence)
        for pattern_name, match_pct in patterns.items():
            if match_pct > 80:  # 80% of values match pattern
                return pattern_name
        
        # Check column name keywords
        if any(kw in col_lower for kw in ['id', 'key', 'code']):
            return 'identifier'
        elif any(kw in col_lower for kw in ['name', 'title']):
            return 'name'
        elif any(kw in col_lower for kw in ['email', 'mail']):
            return 'email'
        elif any(kw in col_lower for kw in ['phone', 'mobile', 'tel']):
            return 'phone'
        elif any(kw in col_lower for kw in ['date', 'time', 'timestamp']):
            return 'datetime'
        elif any(kw in col_lower for kw in ['address', 'street', 'city']):
            return 'address'
        elif any(kw in col_lower for kw in ['price', 'cost', 'amount', 'salary']):
            return 'monetary'
        
        # Fallback to dtype-based inference
        if 'int' in str(dtype) or 'float' in str(dtype):
            return 'numeric'
        elif 'object' in str(dtype) or 'string' in str(dtype):
            return 'text'
        elif 'datetime' in str(dtype):
            return 'datetime'
        elif 'bool' in str(dtype):
            return 'boolean'
        
        return 'unknown'
    
    def compare_schemas(
        self, 
        profile1: Dict[str, Any], 
        profile2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two schema profiles
        
        Returns:
            Dictionary containing comparison results
        """
        cols1 = set(profile1['columns'].keys())
        cols2 = set(profile2['columns'].keys())
        
        return {
            'common_columns': list(cols1 & cols2),
            'only_in_first': list(cols1 - cols2),
            'only_in_second': list(cols2 - cols1),
            'row_count_diff': profile2['row_count'] - profile1['row_count'],
            'column_count_diff': profile2['column_count'] - profile1['column_count'],
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'full_name': ['John Doe', 'Jane Smith', None, 'Bob Wilson', 'Alice Brown'],
        'email': ['john@example.com', 'jane@test.com', 'invalid', None, 'alice@demo.com'],
        'phone': ['+1234567890', '+9876543210', None, '+1122334455', '+5566778899'],
        'purchase_amount': [99.99, 149.50, 200.00, None, 75.25],
    })
    
    profiler = SchemaProfiler(sample_size=100)
    profile = profiler.profile(sample_data)
    
    print("Schema Profile:")
    print(f"Rows: {profile['row_count']}, Columns: {profile['column_count']}")
    print("\nColumn Details:")
    for col_name, col_info in profile['columns'].items():
        print(f"\n{col_name}:")
        print(f"  Type: {col_info['dtype']}")
        print(f"  Semantic: {col_info['semantic_type']}")
        print(f"  NULLs: {col_info['null_percentage']:.1f}%")
        print(f"  Unique: {col_info['unique_count']}")
        if col_info['detected_patterns']:
            print(f"  Patterns: {col_info['detected_patterns']}")
