"""
Configuration module for MasterData MVP
Contains all constants, thresholds, and settings
"""
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_DIR = DATA_DIR / "master"
UPLOADS_DIR = DATA_DIR / "uploads"
CACHE_DIR = PROJECT_ROOT / "embeddings_cache"
METADATA_DB = DATA_DIR / "metadata.db"

# Create directories if they don't exist
for directory in [DATA_DIR, MASTER_DIR, UPLOADS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MASTER SCHEMA DEFINITION
# ============================================================================

MASTER_SCHEMA = {
    "customer_id": {
        "type": "int64",
        "nullable": False,
        "description": "unique customer identifier number",
        "constraints": ["unique", "primary_key"],
        "null_threshold": 0.0  # 0% nulls allowed
    },
    "full_name": {
        "type": "string",
        "nullable": False,
        "description": "complete name of the customer first and last",
        "constraints": [],
        "null_threshold": 0.05  # 5% nulls allowed
    },
    "email_address": {
        "type": "string",
        "nullable": True,
        "description": "customer email contact information address",
        "regex": r"^[\w\.-]+@[\w\.-]+\.\w+$",
        "constraints": [],
        "null_threshold": 0.30  # 30% nulls allowed
    },
    "phone_number": {
        "type": "string",
        "nullable": True,
        "description": "customer telephone phone number for contact",
        "regex": r"^\+?[\d\s\-\(\)]{10,}$",
        "constraints": [],
        "null_threshold": 0.40  # 40% nulls allowed
    },
    "registration_date": {
        "type": "datetime64",
        "nullable": False,
        "description": "date when customer registered signed up enrolled",
        "constraints": [],
        "null_threshold": 0.10  # 10% nulls allowed
    },
    "total_purchases": {
        "type": "float64",
        "nullable": True,
        "description": "total amount value of all purchases made by customer",
        "constraints": ["non_negative"],
        "null_threshold": 0.50  # 50% nulls allowed
    },
    "account_status": {
        "type": "string",
        "nullable": False,
        "description": "current status of customer account active inactive suspended",
        "constraints": ["enum"],
        "allowed_values": ["active", "inactive", "suspended"],
        "null_threshold": 0.0  # 0% nulls allowed
    }
}

# ============================================================================
# MATCHING CONFIGURATION
# ============================================================================

# Embedding model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
EMBEDDING_SAMPLE_SIZE = 5  # Number of sample values to include

# Confidence score weights
CONFIDENCE_WEIGHTS = {
    "embedding": 0.40,  # Semantic similarity weight
    "rule": 0.40,       # Rule-based matching weight
    "quality": 0.20     # Data quality weight (null penalty)
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "auto_accept": 0.85,   # >= 0.85: automatically accept
    "review_needed": 0.70,  # 0.70-0.84: human review required
    "reject": 0.70          # < 0.70: reject mapping
}

# Rule-based matching configuration
RULE_MATCHING = {
    "exact_match": {
        "weight": 1.0,
        "case_sensitive": False
    },
    "fuzzy_match": {
        "weight": 0.7,
        "threshold": 0.85,  # Minimum Levenshtein similarity
        "enabled": True
    },
    "contains_match": {
        "weight": 0.6,
        "enabled": True
    },
    "regex_match": {
        "weight": 0.8,
        "enabled": True
    },
    "type_match": {
        "weight": 0.5,
        "enabled": True
    }
}

# Type compatibility mapping
TYPE_COMPATIBILITY = {
    "int64": ["identifier", "numeric"],
    "float64": ["numeric", "monetary"],
    "string": ["name", "text", "email", "phone", "address", "identifier"],
    "datetime64": ["datetime", "date"],
    "bool": ["boolean"]
}

# ============================================================================
# DATA QUALITY CONFIGURATION
# ============================================================================

# Quality metric thresholds
QUALITY_THRESHOLDS = {
    "completeness": {
        "excellent": 95.0,  # >= 95% non-null
        "good": 85.0,       # >= 85% non-null
        "fair": 70.0,       # >= 70% non-null
        "poor": 0.0         # < 70% non-null
    },
    "uniqueness": {
        "high": 90.0,       # >= 90% unique for ID columns
        "medium": 50.0,     # >= 50% unique
        "low": 0.0          # < 50% unique
    }
}

# Columns that should be unique
UNIQUENESS_COLUMNS = ["customer_id"]

# ============================================================================
# REGEX PATTERNS
# ============================================================================

REGEX_PATTERNS = {
    "email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
    "phone": r"^\+?[\d\s\-\(\)]{10,}$",
    "url": r"^https?://[\w\.-]+\.\w+",
    "date_iso": r"^\d{4}-\d{2}-\d{2}",
    "date_us": r"^\d{2}/\d{2}/\d{4}",
    "zip_code": r"^\d{5}(-\d{4})?$",
    "credit_card": r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$",
}

# ============================================================================
# MERGE CONFIGURATION
# ============================================================================

MERGE_CONFIG = {
    "conflict_resolution": "last_write_wins",  # or "maintain_history"
    "type_casting_strict": False,  # Allow flexible type conversion
    "validate_before_merge": True,
    "backup_before_merge": True,
    "max_merge_batch_size": 100000  # Process in batches
}

# ============================================================================
# VERSIONING CONFIGURATION
# ============================================================================

VERSION_CONFIG = {
    "enabled": True,
    "max_versions_to_keep": 10,
    "version_format": "master_v{version}.parquet",
    "track_lineage": True
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": DATA_DIR / "masterdata.log"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_master_column_metadata(column_name: str) -> Dict[str, Any]:
    """Get metadata for a master schema column"""
    return MASTER_SCHEMA.get(column_name, {})

def get_null_threshold(column_name: str) -> float:
    """Get null threshold for a column"""
    return MASTER_SCHEMA.get(column_name, {}).get("null_threshold", 1.0)

def is_primary_key(column_name: str) -> bool:
    """Check if column is a primary key"""
    constraints = MASTER_SCHEMA.get(column_name, {}).get("constraints", [])
    return "primary_key" in constraints

def get_allowed_values(column_name: str) -> list:
    """Get allowed values for enum columns"""
    return MASTER_SCHEMA.get(column_name, {}).get("allowed_values", [])

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    # Check weights sum to 1.0
    total_weight = sum(CONFIDENCE_WEIGHTS.values())
    assert abs(total_weight - 1.0) < 0.001, f"Confidence weights must sum to 1.0, got {total_weight}"
    
    # Check thresholds are valid
    assert 0 <= CONFIDENCE_THRESHOLDS["reject"] <= 1.0
    assert CONFIDENCE_THRESHOLDS["reject"] <= CONFIDENCE_THRESHOLDS["review_needed"]
    assert CONFIDENCE_THRESHOLDS["review_needed"] <= CONFIDENCE_THRESHOLDS["auto_accept"]
    
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"\n✓ Project root: {PROJECT_ROOT}")
    print(f"✓ Data directory: {DATA_DIR}")
    print(f"✓ Master schema: {len(MASTER_SCHEMA)} columns")
    print(f"✓ Embedding model: {EMBEDDING_MODEL}")
