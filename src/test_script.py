"""
Comprehensive test suite for MasterData pipeline
Run this to validate all components are working correctly
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from schema_profiler import SchemaProfiler
from embedding_matcher import EmbeddingMatcher
from rule_matcher import RuleMatcher
from confidence_scorer import ConfidenceScorer
from main_pipeline import MasterDataPipeline


class TestRunner:
    """Test runner for MasterData pipeline"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name: str, condition: bool, details: str = ""):
        """Record test result"""
        status = "✓ PASS" if condition else "✗ FAIL"
        self.tests.append((name, condition, details))
        
        if condition:
            self.passed += 1
            print(f"{status}: {name}")
        else:
            self.failed += 1
            print(f"{status}: {name}")
            if details:
                print(f"  Details: {details}")
    
    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} ({pass_rate:.1f}%)")
        print(f"Failed: {self.failed}")
        print("=" * 80)
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED - Please review above")
        
        return self.failed == 0


def generate_test_data():
    """Generate diverse test datasets"""
    
    # Dataset 1: Perfect match (easy)
    perfect_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'full_name': ['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown', 'Eve Davis'],
        'email_address': ['alice@email.com', 'bob@email.com', 'carol@email.com', 'david@email.com', 'eve@email.com'],
        'phone_number': ['+1-555-0101', '+1-555-0102', '+1-555-0103', '+1-555-0104', '+1-555-0105'],
        'registration_date': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05', '2024-05-12']),
        'total_purchases': [1250.50, 890.25, 2100.00, 450.75, 1680.30],
        'account_status': ['active', 'active', 'inactive', 'active', 'active']
    })
    
    # Dataset 2: Abbreviated names (medium)
    abbrev_data = pd.DataFrame({
        'cust_id': [101, 102, 103, 104, 105],
        'name': ['Frank Wilson', 'Grace Lee', 'Henry Chen', 'Iris Martinez', 'Jack Taylor'],
        'email': ['frank@test.com', 'grace@test.com', 'henry@test.com', 'iris@test.com', 'jack@test.com'],
        'phone': ['+1-555-0201', '+1-555-0202', '+1-555-0203', '+1-555-0204', '+1-555-0205'],
        'reg_date': pd.to_datetime(['2024-06-01', '2024-06-15', '2024-07-20', '2024-08-10', '2024-09-05']),
        'purchase_total': [750.00, 1100.50, 890.25, 1450.75, 920.00],
        'status': ['active', 'active', 'suspended', 'active', 'inactive']
    })
    
    # Dataset 3: Very different names + nulls (hard)
    difficult_data = pd.DataFrame({
        'ID': [201, 202, 203, 204, 205],
        'CustomerName': ['Kate Anderson', None, 'Leo Thomas', 'Mia Robinson', 'Noah Walker'],
        'EmailAddr': ['kate@demo.com', 'mike@demo.com', None, 'mia@demo.com', 'noah@demo.com'],
        'ContactNumber': [None, '+1-555-0302', '+1-555-0303', '+1-555-0304', None],
        'SignupTimestamp': pd.to_datetime(['2024-09-20', '2024-10-01', '2024-10-15', '2024-11-01', '2024-11-20']),
        'LTV': [3200.00, None, 1890.50, 2450.25, None],
        'Active': ['yes', 'yes', 'no', 'yes', 'yes']
    })
    
    # Dataset 4: Type mismatches
    mixed_types_data = pd.DataFrame({
        'user_id': ['U001', 'U002', 'U003', 'U004', 'U005'],  # String IDs
        'fullname': ['Olivia Garcia', 'Paul Martinez', 'Quinn Rodriguez', 'Rachel Lee', 'Sam Wilson'],
        'email_contact': ['olivia@test.com', 'paul@test.com', 'quinn@test.com', 'rachel@test.com', 'sam@test.com'],
        'tel': ['5550401', '5550402', '5550403', '5550404', '5550405'],  # No country code
        'created_at': ['2024-12-01', '2024-12-05', '2024-12-10', '2024-12-15', '2024-12-20'],  # String dates
        'total_revenue': ['1250.50', '890.25', '2100.00', '450.75', '1680.30'],  # String numbers
        'is_active': [1, 1, 0, 1, 0]  # Numeric instead of string
    })
    
    return {
        'perfect': perfect_data,
        'abbreviated': abbrev_data,
        'difficult': difficult_data,
        'mixed_types': mixed_types_data
    }


def run_tests():
    """Run all tests"""
    runner = TestRunner()
    
    print("=" * 80)
    print("MASTERDATA PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test 1: Configuration
    print("TEST GROUP 1: Configuration")
    print("-" * 80)
    try:
        config.validate_config()
        runner.test("Configuration validation", True)
    except Exception as e:
        runner.test("Configuration validation", False, str(e))
    
    runner.test(
        "Master schema defined",
        len(config.MASTER_SCHEMA) == 7,
        f"Found {len(config.MASTER_SCHEMA)} columns"
    )
    
    runner.test(
        "Confidence weights sum to 1.0",
        abs(sum(config.CONFIDENCE_WEIGHTS.values()) - 1.0) < 0.001
    )
    
    print()
    
    # Test 2: Schema Profiler
    print("TEST GROUP 2: Schema Profiler")
    print("-" * 80)
    
    test_data = generate_test_data()
    profiler = SchemaProfiler()
    
    try:
        profile = profiler.profile(test_data['perfect'])
        runner.test("Profile generation", True)
        runner.test(
            "Profile contains all columns",
            len(profile['columns']) == len(test_data['perfect'].columns)
        )
        runner.test(
            "NULL percentage calculated",
            all('null_percentage' in col for col in profile['columns'].values())
        )
        runner.test(
            "Semantic types inferred",
            all('semantic_type' in col for col in profile['columns'].values())
        )
    except Exception as e:
        runner.test("Profile generation", False, str(e))
    
    print()
    
    # Test 3: Embedding Matcher
    print("TEST GROUP 3: Embedding Matcher")
    print("-" * 80)
    
    try:
        emb_matcher = EmbeddingMatcher(cache_dir=str(config.CACHE_DIR))
        emb_matcher.build_master_embeddings(config.MASTER_SCHEMA)
        runner.test("Master embeddings built", True)
        
        runner.test(
            "Embeddings cached",
            len(emb_matcher.master_embeddings) == len(config.MASTER_SCHEMA)
        )
        
        # Test matching
        profile = profiler.profile(test_data['perfect'])
        matches = emb_matcher.batch_match(profile['columns'], threshold=0.0)
        runner.test(
            "Batch matching works",
            len(matches) == len(test_data['perfect'].columns)
        )
        
        # Check if perfect matches have high scores
        perfect_score = matches.get('customer_id', [(None, 0)])[0][1]
        runner.test(
            "Perfect match has high score",
            perfect_score > 0.8,
            f"Score: {perfect_score:.3f}"
        )
        
    except Exception as e:
        runner.test("Embedding matcher", False, str(e))
    
    print()
    
    # Test 4: Rule Matcher
    print("TEST GROUP 4: Rule Matcher")
    print("-" * 80)
    
    try:
        rule_matcher = RuleMatcher()
        profile = profiler.profile(test_data['abbreviated'])
        rule_matches = rule_matcher.batch_match(profile['columns'])
        
        runner.test(
            "Rule matching works",
            len(rule_matches) == len(test_data['abbreviated'].columns)
        )
        
        # Check exact matches
        if 'name' in rule_matches:
            name_matches = rule_matches['name']
            has_full_name = any(m[0] == 'full_name' for m in name_matches)
            runner.test(
                "Rule matcher finds 'name' → 'full_name'",
                has_full_name
            )
        
    except Exception as e:
        runner.test("Rule matcher", False, str(e))
    
    print()
    
    # Test 5: Confidence Scorer
    print("TEST GROUP 5: Confidence Scorer")
    print("-" * 80)
    
    try:
        scorer = ConfidenceScorer()
        
        # Get matches from both matchers
        emb_matches = emb_matcher.batch_match(profile['columns'], threshold=0.0)
        rule_matches = rule_matcher.batch_match(profile['columns'])
        
        # Calculate combined scores
        results = scorer.batch_score(emb_matches, rule_matches, profile['columns'])
        
        runner.test(
            "Combined scoring works",
            len(results) == len(test_data['abbreviated'].columns)
        )
        
        # Check that recommendations are assigned
        has_recommendations = all(
            len(matches) > 0 and hasattr(matches[0], 'recommendation')
            for matches in results.values() if matches
        )
        runner.test("Recommendations assigned", has_recommendations)
        
    except Exception as e:
        runner.test("Confidence scorer", False, str(e))
    
    print()
    
    # Test 6: Complete Pipeline
    print("TEST GROUP 6: Complete Pipeline")
    print("-" * 80)
    
    try:
        pipeline = MasterDataPipeline(verbose=False)
        
        # Test with perfect data
        results = pipeline.process_source_data(test_data['perfect'], "Perfect")
        runner.test("Pipeline processes perfect data", True)
        runner.test(
            "Perfect data has matches",
            len(results['best_matches']) > 0
        )
        
        # Test with difficult data
        results_difficult = pipeline.process_source_data(test_data['difficult'], "Difficult")
        runner.test("Pipeline processes difficult data", True)
        
        # Test mapping export
        mapping = pipeline.export_mapping(results)
        runner.test("Mapping export works", isinstance(mapping, dict))
        
        # Test mapping application
        transformed = pipeline.apply_mapping(test_data['perfect'], mapping)
        runner.test("Mapping application works", len(transformed) > 0)
        
        # Check transformed columns are in master schema
        valid_columns = all(col in config.MASTER_SCHEMA for col in transformed.columns)
        runner.test("Transformed columns valid", valid_columns)
        
    except Exception as e:
        runner.test("Complete pipeline", False, str(e))
    
    print()
    
    # Test 7: Edge Cases
    print("TEST GROUP 7: Edge Cases")
    print("-" * 80)
    
    try:
        # Empty dataframe
        empty_df = pd.DataFrame()
        profile_empty = profiler.profile(empty_df)
        runner.test("Handles empty dataframe", True)
        
    except Exception as e:
        runner.test("Handles empty dataframe", False, str(e))
    
    try:
        # All nulls
        all_nulls = pd.DataFrame({'col1': [None, None, None]})
        profile_nulls = profiler.profile(all_nulls)
        runner.test("Handles all-null column", True)
        
    except Exception as e:
        runner.test("Handles all-null column", False, str(e))
    
    try:
        # Single row
        single_row = pd.DataFrame({'col1': [1]})
        profile_single = profiler.profile(single_row)
        runner.test("Handles single row", True)
        
    except Exception as e:
        runner.test("Handles single row", False, str(e))
    
    print()
    
    # Print summary
    success = runner.summary()
    
    return success


if __name__ == "__main__":
    print("\n🧪 Starting comprehensive test suite...\n")
    success = run_tests()
    
    if success:
        print("\n✅ All systems operational! You're ready to use MasterData.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)
