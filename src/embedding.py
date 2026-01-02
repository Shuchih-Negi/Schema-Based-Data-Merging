"""
Embedding-Based Matcher - Uses ML embeddings for semantic column matching
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


class EmbeddingMatcher:
    """Semantic column matching using sentence embeddings"""
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        cache_dir: str = './embeddings_cache'
    ):
        """
        Args:
            model_name: HuggingFace model for embeddings
            cache_dir: Directory to cache embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache for master schema embeddings
        self.master_embeddings = {}
    
    def generate_embedding(
        self, 
        column_name: str, 
        column_description: str = "", 
        sample_values: List[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for a column using name, description, and samples
        
        Args:
            column_name: Name of the column
            column_description: Optional semantic description
            sample_values: Optional sample values from the column
            
        Returns:
            Embedding vector as numpy array
        """
        # Construct rich text representation
        text_parts = [column_name]
        
        if column_description:
            text_parts.append(column_description)
        
        if sample_values and len(sample_values) > 0:
            # Take first 5 unique samples
            samples_str = ", ".join(str(v) for v in sample_values[:5])
            text_parts.append(f"Example values: {samples_str}")
        
        combined_text = " | ".join(text_parts)
        
        # Generate embedding
        embedding = self.model.encode(combined_text, convert_to_numpy=True)
        
        return embedding
    
    def build_master_embeddings(
        self, 
        master_schema: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Pre-compute and cache embeddings for master schema
        
        Args:
            master_schema: Dictionary of column_name -> column_metadata
        """
        print(f"Building embeddings for {len(master_schema)} master columns...")
        
        for col_name, col_metadata in master_schema.items():
            description = col_metadata.get('description', '')
            
            embedding = self.generate_embedding(
                column_name=col_name,
                column_description=description
            )
            
            self.master_embeddings[col_name] = embedding
        
        # Save to cache
        cache_file = os.path.join(self.cache_dir, 'master_embeddings.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(self.master_embeddings, f)
        
        print(f"✓ Master embeddings cached to {cache_file}")
    
    def load_master_embeddings(self) -> bool:
        """
        Load master embeddings from cache
        
        Returns:
            True if successfully loaded, False otherwise
        """
        cache_file = os.path.join(self.cache_dir, 'master_embeddings.pkl')
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.master_embeddings = pickle.load(f)
            print(f"✓ Loaded {len(self.master_embeddings)} master embeddings from cache")
            return True
        
        return False
    
    def match_column(
        self, 
        source_column: str,
        source_profile: Dict[str, Any],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find best matching master columns for a source column
        
        Args:
            source_column: Name of source column
            source_profile: Profile metadata for the column
            top_k: Number of top matches to return
            
        Returns:
            List of (master_column_name, similarity_score) tuples
        """
        if not self.master_embeddings:
            raise ValueError("Master embeddings not built. Call build_master_embeddings() first.")
        
        # Generate embedding for source column
        sample_values = source_profile.get('sample_values', [])
        source_embedding = self.generate_embedding(
            column_name=source_column,
            column_description="",  # Source columns don't have descriptions
            sample_values=sample_values
        )
        
        # Compute similarities with all master columns
        similarities = []
        
        for master_col, master_emb in self.master_embeddings.items():
            similarity = cosine_similarity(
                source_embedding.reshape(1, -1),
                master_emb.reshape(1, -1)
            )[0][0]
            
            similarities.append((master_col, float(similarity)))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def batch_match(
        self,
        source_columns: Dict[str, Dict[str, Any]],
        threshold: float = 0.7,
        top_k: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Match multiple source columns against master schema
        
        Args:
            source_columns: Dict of column_name -> column_profile
            threshold: Minimum similarity threshold
            top_k: Number of matches per column
            
        Returns:
            Dict of source_column -> [(master_column, score), ...]
        """
        matches = {}
        
        for source_col, profile in source_columns.items():
            candidates = self.match_column(source_col, profile, top_k=top_k)
            
            # Filter by threshold
            filtered = [(col, score) for col, score in candidates if score >= threshold]
            
            matches[source_col] = filtered
        
        return matches
    
    def explain_match(
        self,
        source_column: str,
        master_column: str,
        similarity_score: float
    ) -> str:
        """
        Generate human-readable explanation for a match
        
        Returns:
            Explanation string
        """
        if similarity_score >= 0.9:
            confidence = "Very High"
        elif similarity_score >= 0.8:
            confidence = "High"
        elif similarity_score >= 0.7:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        explanation = (
            f"Column '{source_column}' matched to '{master_column}'\n"
            f"Similarity Score: {similarity_score:.3f} ({confidence} confidence)\n"
            f"This match is based on semantic similarity between column names and sample values."
        )
        
        return explanation


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from schema_profiler import SchemaProfiler
    
    # Define master schema
    master_schema = {
        "customer_id": {
            "type": "int64",
            "description": "unique customer identifier number"
        },
        "full_name": {
            "type": "string",
            "description": "complete name of the customer"
        },
        "email_address": {
            "type": "string",
            "description": "customer email contact information"
        },
        "phone_number": {
            "type": "string",
            "description": "customer phone number for contact"
        },
        "total_purchases": {
            "type": "float64",
            "description": "total amount of purchases made by customer"
        }
    }
    
    # Create sample source data with different column names
    source_data = pd.DataFrame({
        'cust_id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
        'email': ['john@example.com', 'jane@test.com', 'bob@demo.com'],
        'mobile': ['+1234567890', '+9876543210', '+1122334455'],
        'purchase_total': [99.99, 149.50, 200.00]
    })
    
    # Profile source data
    profiler = SchemaProfiler()
    source_profile = profiler.profile(source_data)
    
    # Initialize matcher
    matcher = EmbeddingMatcher()
    
    # Build master embeddings
    matcher.build_master_embeddings(master_schema)
    
    # Match all source columns
    print("\n" + "="*60)
    print("EMBEDDING-BASED MATCHING RESULTS")
    print("="*60)
    
    matches = matcher.batch_match(
        source_profile['columns'],
        threshold=0.6,
        top_k=3
    )
    
    for source_col, candidates in matches.items():
        print(f"\n'{source_col}' → ")
        if candidates:
            for master_col, score in candidates:
                print(f"  • {master_col}: {score:.3f}")
        else:
            print("  No matches found")
    
    # Show detailed explanation for best match
    if matches['cust_id']:
        best_match = matches['cust_id'][0]
        print("\n" + "="*60)
        print(matcher.explain_match('cust_id', best_match[0], best_match[1]))
