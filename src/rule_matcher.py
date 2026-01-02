"""
Rule-Based Matcher - Performs deterministic column matching using rules
"""
import re
from typing import Dict, List, Tuple, Any
from Levenshtein import ratio as levenshtein_ratio
import config


class RuleMatcher:
    """Performs rule-based column matching"""
    
    def __init__(self):
        self.master_schema = config.MASTER_SCHEMA
        self.rules_config = config.RULE_MATCHING
        self.type_compatibility = config.TYPE_COMPATIBILITY
        self.patterns = config.REGEX_PATTERNS
    
    def match_column(
        self, 
        source_column: str,
        source_profile: Dict[str, Any]
    ) -> List[Tuple[str, float, str]]:
        """
        Match source column against master schema using rules
        
        Args:
            source_column: Name of source column
            source_profile: Profile metadata for the column
            
        Returns:
            List of (master_column, score, match_reason) tuples
        """
        matches = []
        
        for master_column, master_metadata in self.master_schema.items():
            score, reason = self._calculate_rule_score(
                source_column,
                source_profile,
                master_column,
                master_metadata
            )
            
            if score > 0:
                matches.append((master_column, score, reason))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def _calculate_rule_score(
        self,
        source_column: str,
        source_profile: Dict[str, Any],
        master_column: str,
        master_metadata: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Calculate rule-based matching score
        
        Returns:
            Tuple of (score, reason)
        """
        scores = []
        reasons = []
        
        # Rule 1: Exact name match (case-insensitive)
        if self._exact_match(source_column, master_column):
            scores.append(self.rules_config["exact_match"]["weight"])
            reasons.append("exact_name_match")
        
        # Rule 2: Fuzzy name match (Levenshtein)
        if self.rules_config["fuzzy_match"]["enabled"]:
            fuzzy_score = self._fuzzy_match(source_column, master_column)
            if fuzzy_score >= self.rules_config["fuzzy_match"]["threshold"]:
                weighted_score = fuzzy_score * self.rules_config["fuzzy_match"]["weight"]
                scores.append(weighted_score)
                reasons.append(f"fuzzy_match({fuzzy_score:.2f})")
        
        # Rule 3: Contains match (substring)
        if self.rules_config["contains_match"]["enabled"]:
            if self._contains_match(source_column, master_column):
                scores.append(self.rules_config["contains_match"]["weight"])
                reasons.append("contains_match")
        
        # Rule 4: Type compatibility
        if self.rules_config["type_match"]["enabled"]:
            if self._type_compatible(source_profile, master_metadata):
                scores.append(self.rules_config["type_match"]["weight"])
                reasons.append("type_compatible")
        
        # Rule 5: Regex pattern match
        if self.rules_config["regex_match"]["enabled"]:
            if self._regex_match(source_profile, master_metadata):
                scores.append(self.rules_config["regex_match"]["weight"])
                reasons.append("regex_pattern_match")
        
        # Return best score and combined reasons
        if scores:
            best_score = max(scores)
            reason_str = " + ".join(reasons)
            return best_score, reason_str
        
        return 0.0, "no_match"
    
    def _exact_match(self, source_col: str, master_col: str) -> bool:
        """Check for exact name match (case-insensitive)"""
        if not self.rules_config["exact_match"]["case_sensitive"]:
            return source_col.lower() == master_col.lower()
        return source_col == master_col
    
    def _fuzzy_match(self, source_col: str, master_col: str) -> float:
        """Calculate fuzzy match score using Levenshtein distance"""
        # Normalize column names
        source_normalized = self._normalize_column_name(source_col)
        master_normalized = self._normalize_column_name(master_col)
        
        # Calculate similarity ratio
        similarity = levenshtein_ratio(source_normalized, master_normalized)
        
        return similarity
    
    def _normalize_column_name(self, col_name: str) -> str:
        """Normalize column name for comparison"""
        # Convert to lowercase
        normalized = col_name.lower()
        
        # Remove common prefixes/suffixes
        prefixes = ['customer_', 'user_', 'client_', 'cust_', 'usr_']
        suffixes = ['_id', '_name', '_number', '_date', '_status']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        # Replace underscores and hyphens with spaces
        normalized = normalized.replace('_', ' ').replace('-', ' ')
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _contains_match(self, source_col: str, master_col: str) -> bool:
        """Check if one column name contains the other"""
        source_lower = source_col.lower()
        master_lower = master_col.lower()
        
        # Check both directions
        return (
            source_lower in master_lower or 
            master_lower in source_lower or
            self._normalize_column_name(source_col) in self._normalize_column_name(master_col) or
            self._normalize_column_name(master_col) in self._normalize_column_name(source_col)
        )
    
    def _type_compatible(
        self,
        source_profile: Dict[str, Any],
        master_metadata: Dict[str, Any]
    ) -> bool:
        """Check if source and master types are compatible"""
        source_semantic = source_profile.get("semantic_type", "unknown")
        master_type = master_metadata.get("type", "string")
        
        # Get compatible semantic types for master type
        compatible_types = self.type_compatibility.get(master_type, [])
        
        return source_semantic in compatible_types
    
    def _regex_match(
        self,
        source_profile: Dict[str, Any],
        master_metadata: Dict[str, Any]
    ) -> bool:
        """Check if detected patterns match master column requirements"""
        detected_patterns = source_profile.get("detected_patterns", {})
        master_regex = master_metadata.get("regex")
        
        if not master_regex:
            return False
        
        # Check if any detected pattern aligns with master requirements
        # For example, if master expects email and source has high email pattern match
        if "email" in str(master_regex) and "email" in detected_patterns:
            return detected_patterns["email"] > 80
        
        if "phone" in str(master_regex) and "phone" in detected_patterns:
            return detected_patterns["phone"] > 80
        
        return False
    
    def batch_match(
        self,
        source_columns: Dict[str, Dict[str, Any]],
        top_k: int = 3
    ) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Match multiple source columns
        
        Args:
            source_columns: Dict of column_name -> column_profile
            top_k: Number of top matches to return per column
            
        Returns:
            Dict of source_column -> [(master_column, score, reason), ...]
        """
        results = {}
        
        for source_col, profile in source_columns.items():
            matches = self.match_column(source_col, profile)
            results[source_col] = matches[:top_k]
        
        return results
    
    def explain_match(
        self,
        source_column: str,
        master_column: str,
        score: float,
        reason: str
    ) -> str:
        """Generate human-readable explanation for a match"""
        explanation = f"Rule-based match: '{source_column}' → '{master_column}'\n"
        explanation += f"Score: {score:.3f}\n"
        explanation += f"Reasons: {reason}\n"
        
        if "exact_name_match" in reason:
            explanation += "  • Column names are identical\n"
        if "fuzzy_match" in reason:
            explanation += "  • Column names are very similar\n"
        if "contains_match" in reason:
            explanation += "  • One column name contains the other\n"
        if "type_compatible" in reason:
            explanation += "  • Data types are compatible\n"
        if "regex_pattern_match" in reason:
            explanation += "  • Values match expected pattern\n"
        
        return explanation


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    from schema_profiler import SchemaProfiler
    
    # Sample data with various column naming styles
    test_data = pd.DataFrame({
        'cust_id': [1, 2, 3, 4, 5],
        'customer_name': ['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown', 'Eve Davis'],
        'email': ['alice@email.com', 'bob@email.com', None, 'david@email.com', 'eve@email.com'],
        'phone': ['+1-555-0101', '+1-555-0102', '+1-555-0103', None, '+1-555-0105'],
        'reg_date': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05', '2024-05-12']),
        'total_spend': [1250.50, 890.25, 2100.00, 450.75, 1680.30],
        'active': ['yes', 'yes', 'no', 'yes', 'yes']
    })
    
    # Profile the data
    profiler = SchemaProfiler()
    profile = profiler.profile(test_data)
    
    # Initialize rule matcher
    matcher = RuleMatcher()
    
    # Match columns
    print("="*70)
    print("RULE-BASED MATCHING RESULTS")
    print("="*70)
    
    matches = matcher.batch_match(profile['columns'], top_k=3)
    
    for source_col, candidates in matches.items():
        print(f"\n'{source_col}':")
        if candidates:
            for master_col, score, reason in candidates:
                print(f"  → {master_col:25s} score={score:.3f} ({reason})")
        else:
            print("  No matches found")
    
    # Show detailed explanation for best match
    if matches['cust_id']:
        print("\n" + "="*70)
        best_match = matches['cust_id'][0]
        print(matcher.explain_match('cust_id', best_match[0], best_match[1], best_match[2]))
