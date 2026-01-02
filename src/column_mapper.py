"""
Confidence Scorer - Combines rule-based and embedding-based scores
"""
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import config


@dataclass
class MatchResult:
    """Represents a column matching result"""
    source_column: str
    master_column: str
    embedding_score: float
    rule_score: float
    quality_score: float
    combined_score: float
    recommendation: str
    match_reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'source_column': self.source_column,
            'master_column': self.master_column,
            'embedding_score': round(self.embedding_score, 3),
            'rule_score': round(self.rule_score, 3),
            'quality_score': round(self.quality_score, 3),
            'combined_score': round(self.combined_score, 3),
            'recommendation': self.recommendation,
            'match_reason': self.match_reason
        }


class ConfidenceScorer:
    """Combines multiple matching signals into final confidence score"""
    
    def __init__(self):
        self.weights = config.CONFIDENCE_WEIGHTS
        self.thresholds = config.CONFIDENCE_THRESHOLDS
        self.master_schema = config.MASTER_SCHEMA
    
    def calculate_combined_score(
        self,
        source_column: str,
        master_column: str,
        embedding_score: float,
        rule_score: float,
        source_profile: Dict[str, Any]
    ) -> MatchResult:
        """
        Calculate final combined confidence score
        
        Args:
            source_column: Name of source column
            master_column: Name of master column
            embedding_score: Score from embedding matcher (0-1)
            rule_score: Score from rule matcher (0-1)
            source_profile: Source column profile metadata
            
        Returns:
            MatchResult object with all scores and recommendation
        """
        # Calculate quality score (inverse of null percentage)
        null_percentage = source_profile.get('null_percentage', 0)
        quality_score = 1.0 - (null_percentage / 100.0)
        
        # Apply null threshold penalty
        master_metadata = self.master_schema.get(master_column, {})
        null_threshold = master_metadata.get('null_threshold', 1.0)
        
        if null_percentage / 100.0 > null_threshold:
            # Significant penalty for exceeding threshold
            quality_penalty = 1.0 - ((null_percentage / 100.0 - null_threshold) / (1.0 - null_threshold))
            quality_score = quality_score * quality_penalty
        
        # Calculate weighted combined score
        combined_score = (
            self.weights['embedding'] * embedding_score +
            self.weights['rule'] * rule_score +
            self.weights['quality'] * quality_score
        )
        
        # Determine recommendation
        recommendation = self._get_recommendation(combined_score, null_percentage, null_threshold)
        
        # Generate match reason
        match_reason = self._generate_match_reason(
            embedding_score,
            rule_score,
            quality_score
        )
        
        return MatchResult(
            source_column=source_column,
            master_column=master_column,
            embedding_score=embedding_score,
            rule_score=rule_score,
            quality_score=quality_score,
            combined_score=combined_score,
            recommendation=recommendation,
            match_reason=match_reason
        )
    
    def _get_recommendation(
        self,
        combined_score: float,
        null_percentage: float,
        null_threshold: float
    ) -> str:
        """Determine recommendation based on score and constraints"""
        # Check null threshold violation
        if null_percentage / 100.0 > null_threshold:
            return "REVIEW_NEEDED (NULL threshold exceeded)"
        
        # Check combined score thresholds
        if combined_score >= self.thresholds['auto_accept']:
            return "AUTO_ACCEPT"
        elif combined_score >= self.thresholds['review_needed']:
            return "REVIEW_NEEDED"
        else:
            return "REJECT"
    
    def _generate_match_reason(
        self,
        embedding_score: float,
        rule_score: float,
        quality_score: float
    ) -> str:
        """Generate human-readable match reason"""
        reasons = []
        
        if embedding_score >= 0.8:
            reasons.append("high semantic similarity")
        elif embedding_score >= 0.6:
            reasons.append("moderate semantic similarity")
        
        if rule_score >= 0.8:
            reasons.append("strong rule match")
        elif rule_score >= 0.6:
            reasons.append("moderate rule match")
        
        if quality_score >= 0.9:
            reasons.append("excellent data quality")
        elif quality_score >= 0.7:
            reasons.append("good data quality")
        elif quality_score < 0.5:
            reasons.append("poor data quality")
        
        return " + ".join(reasons) if reasons else "weak match"
    
    def batch_score(
        self,
        embedding_matches: Dict[str, List[Tuple[str, float]]],
        rule_matches: Dict[str, List[Tuple[str, float, str]]],
        source_profiles: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[MatchResult]]:
        """
        Score multiple columns combining embedding and rule matches
        
        Args:
            embedding_matches: Dict from embedding matcher
            rule_matches: Dict from rule matcher
            source_profiles: Column profiles
            
        Returns:
            Dict of source_column -> [MatchResult, ...]
        """
        results = {}
        
        for source_col in source_profiles.keys():
            # Get matches from both matchers
            emb_matches = {col: score for col, score in embedding_matches.get(source_col, [])}
            rule_matches_dict = {col: score for col, score, _ in rule_matches.get(source_col, [])}
            
            # Find all candidate master columns
            all_candidates = set(emb_matches.keys()) | set(rule_matches_dict.keys())
            
            column_results = []
            
            for master_col in all_candidates:
                emb_score = emb_matches.get(master_col, 0.0)
                rule_score = rule_matches_dict.get(master_col, 0.0)
                
                result = self.calculate_combined_score(
                    source_column=source_col,
                    master_column=master_col,
                    embedding_score=emb_score,
                    rule_score=rule_score,
                    source_profile=source_profiles[source_col]
                )
                
                column_results.append(result)
            
            # Sort by combined score
            column_results.sort(key=lambda x: x.combined_score, reverse=True)
            results[source_col] = column_results
        
        return results
    
    def get_best_matches(
        self,
        scoring_results: Dict[str, List[MatchResult]],
        min_score: float = 0.0
    ) -> Dict[str, MatchResult]:
        """
        Extract best match for each source column
        
        Args:
            scoring_results: Results from batch_score
            min_score: Minimum score threshold
            
        Returns:
            Dict of source_column -> best MatchResult
        """
        best_matches = {}
        
        for source_col, results in scoring_results.items():
            if results and results[0].combined_score >= min_score:
                best_matches[source_col] = results[0]
        
        return best_matches
    
    def generate_mapping_report(
        self,
        best_matches: Dict[str, MatchResult]
    ) -> str:
        """Generate human-readable mapping report"""
        report = []
        report.append("=" * 80)
        report.append("COLUMN MAPPING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Group by recommendation
        auto_accept = []
        review_needed = []
        rejected = []
        
        for source_col, result in best_matches.items():
            if "AUTO_ACCEPT" in result.recommendation:
                auto_accept.append((source_col, result))
            elif "REVIEW_NEEDED" in result.recommendation:
                review_needed.append((source_col, result))
            else:
                rejected.append((source_col, result))
        
        # Auto-accept section
        if auto_accept:
            report.append("✓ AUTO-ACCEPTED MAPPINGS:")
            report.append("-" * 80)
            for source_col, result in auto_accept:
                report.append(f"  {source_col:25s} → {result.master_column:25s} "
                            f"[score: {result.combined_score:.3f}]")
            report.append("")
        
        # Review needed section
        if review_needed:
            report.append("? REVIEW NEEDED:")
            report.append("-" * 80)
            for source_col, result in review_needed:
                report.append(f"  {source_col:25s} → {result.master_column:25s} "
                            f"[score: {result.combined_score:.3f}]")
                report.append(f"    Reason: {result.match_reason}")
            report.append("")
        
        # Rejected section
        if rejected:
            report.append("✗ REJECTED MAPPINGS:")
            report.append("-" * 80)
            for source_col, result in rejected:
                report.append(f"  {source_col:25s} → {result.master_column:25s} "
                            f"[score: {result.combined_score:.3f}]")
            report.append("")
        
        # Summary
        report.append("=" * 80)
        report.append(f"SUMMARY: {len(auto_accept)} auto-accepted, "
                     f"{len(review_needed)} need review, "
                     f"{len(rejected)} rejected")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_unmapped_columns(
        self,
        source_columns: List[str],
        best_matches: Dict[str, MatchResult]
    ) -> List[str]:
        """Find source columns that have no acceptable mapping"""
        mapped = set(best_matches.keys())
        unmapped = [col for col in source_columns if col not in mapped]
        return unmapped


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from schema_profiler import SchemaProfiler
    from embedding_matcher import EmbeddingMatcher
    from rule_matcher import RuleMatcher
    
    # Sample data
    test_data = pd.DataFrame({
        'cust_id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', None, 'David Brown', 'Eve Davis'],
        'email': ['alice@email.com', 'bob@email.com', None, 'david@email.com', 'eve@email.com'],
        'mobile': ['+1-555-0101', None, '+1-555-0103', None, '+1-555-0105'],
        'signup': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05', '2024-05-12']),
        'ltv': [1250.50, 890.25, 2100.00, 450.75, 1680.30],
        'active': ['yes', 'yes', 'no', 'yes', 'yes']
    })
    
    # Profile
    profiler = SchemaProfiler()
    profile = profiler.profile(test_data)
    
    # Embedding matches
    emb_matcher = EmbeddingMatcher()
    emb_matcher.build_master_embeddings(config.MASTER_SCHEMA)
    emb_matches = emb_matcher.batch_match(profile['columns'], threshold=0.0)
    
    # Rule matches
    rule_matcher = RuleMatcher()
    rule_matches = rule_matcher.batch_match(profile['columns'])
    
    # Combined scoring
    scorer = ConfidenceScorer()
    results = scorer.batch_score(emb_matches, rule_matches, profile['columns'])
    
    # Get best matches
    best_matches = scorer.get_best_matches(results, min_score=0.5)
    
    # Print report
    print(scorer.generate_mapping_report(best_matches))
    
    # Show detailed scores for one column
    print("\nDetailed scores for 'cust_id':")
    for result in results['cust_id'][:3]:
        print(f"  {result.master_column}:")
        print(f"    Embedding: {result.embedding_score:.3f}")
        print(f"    Rules:     {result.rule_score:.3f}")
        print(f"    Quality:   {result.quality_score:.3f}")
        print(f"    Combined:  {result.combined_score:.3f}")
        print(f"    → {result.recommendation}")
