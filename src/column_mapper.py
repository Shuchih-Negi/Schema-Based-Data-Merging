
# column_mapper.py
import re
from typing import Dict, Tuple, Set


# ============================================================
# DTYPE COMPATIBILITY RULES
# ============================================================

DTYPE_COMPATIBILITY = {
    "string": {"string", "varchar"},
    "varchar": {"string", "varchar"},
    "int": {"int", "bigint"},
    "bigint": {"int", "bigint"},
    "double": {"double", "float"},
    "float": {"double", "float"},
    "date": {"date", "timestamp", "string"},
    "timestamp": {"timestamp", "date", "string"},
}


def dtypes_compatible(dt1: str, dt2: str) -> bool:
    return dt2 in DTYPE_COMPATIBILITY.get(dt1, set())


# ============================================================
# STEP 0 — SEMANTIC / REGEX-BASED MATCHING
# ============================================================

SEMANTIC_PATTERNS = {
    "email": {
        "regex": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
        "allowed_dtypes": {"string"},
        "master_column": "email"
    },
    "phone_number": {
        "regex": r"^\+?\d{1,3}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{6,10}$",
        "allowed_dtypes": {"string", "bigint"},
        "master_column": "phone_number"
    },
    "pin_code": {
        "regex": r"^\d{6}$",
        "allowed_dtypes": {"string", "int"},
        "master_column": "postal_code"
    },
}


def match_semantic(profile_uploaded: Dict, profile_master: Dict) -> Dict[str, str]:
    matches = {}
    used_master_cols = set()

    for u_col, u_meta in profile_uploaded.items():
        dtype = u_meta["type"]
        examples = [u_meta.get("ex0"), u_meta.get("ex1")]
        non_null_examples = [e for e in examples if e is not None]

        if not non_null_examples:
            continue

        for cfg in SEMANTIC_PATTERNS.values():
            if dtype not in cfg["allowed_dtypes"]:
                continue

            regex = re.compile(cfg["regex"])
            valid = [
                ex for ex in non_null_examples
                if isinstance(ex, str) and regex.match(ex)
            ]

            # Strong match: all samples conform
            if len(valid) == len(non_null_examples):
                master_col = cfg["master_column"]
                if master_col in profile_master and master_col not in used_master_cols:
                    matches[u_col] = master_col
                    used_master_cols.add(master_col)
                    break

    return matches


# ============================================================
# STEP 1 — EXACT NAME MATCH
# ============================================================

def match_exact(profile_uploaded: Dict, profile_master: Dict) -> Dict[str, str]:
    matches = {}
    master_lookup = {c.lower(): c for c in profile_master.keys()}

    for u_col, u_meta in profile_uploaded.items():
        key = u_col.lower()
        if key in master_lookup:
            m_col = master_lookup[key]
            if dtypes_compatible(u_meta["type"], profile_master[m_col]["type"]):
                matches[u_col] = m_col

    return matches


# ============================================================
# STEP 2 — PARTIAL / TOKEN MATCH
# ============================================================

def _normalize(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())


def match_partial(profile_uploaded: Dict, profile_master: Dict) -> Dict[str, str]:
    matches = {}
    master_norm = {
        _normalize(m_col): m_col
        for m_col in profile_master.keys()
    }

    for u_col, u_meta in profile_uploaded.items():
        u_norm = _normalize(u_col)

        for m_norm, m_col in master_norm.items():
            if u_norm in m_norm or m_norm in u_norm:
                if dtypes_compatible(u_meta["type"], profile_master[m_col]["type"]):
                    matches[u_col] = m_col
                    break

    return matches


# ============================================================
# STEP 3 — HISTORICAL MEMORY MATCH (STUB)
# ============================================================

def match_from_memory(profile_uploaded: Dict, memory_index: Dict) -> Dict[str, str]:
    """
    memory_index:
        uploaded_column_name -> master_column
        (later replaced with embedding similarity)
    """
    matches = {}

    for u_col in profile_uploaded.keys():
        if u_col in memory_index:
            matches[u_col] = memory_index[u_col]

    return matches


# ============================================================
# STEP 4 — EMBEDDING-BASED MATCH (PLACEHOLDER)
# ============================================================

def match_with_embeddings(profile_uploaded: Dict, profile_master: Dict) -> Dict[str, str]:
    # To be implemented: cosine similarity / LLM logic
    return {}


# ============================================================
# ORCHESTRATOR — COLUMN MAPPER
# ============================================================

class ColumnMapper:
    def __init__(
        self,
        profile_uploaded: Dict,
        profile_master: Dict,
        memory_index: Dict = None
    ):
        self.profile_uploaded = profile_uploaded
        self.profile_master = profile_master
        self.memory_index = memory_index or {}

        self.final_mapping = {}
        self.unmatched_uploaded = set(profile_uploaded.keys())
        self.unmatched_master = set(profile_master.keys())

    def _apply(self, matcher_fn):
        matches = matcher_fn(
            {k: self.profile_uploaded[k] for k in self.unmatched_uploaded},
            {k: self.profile_master[k] for k in self.unmatched_master},
        )

        for u_col, m_col in matches.items():
            self.final_mapping[u_col] = m_col
            self.unmatched_uploaded.discard(u_col)
            self.unmatched_master.discard(m_col)

    def run(self):
        self._apply(match_semantic)
        self._apply(match_exact)
        self._apply(match_partial)
        self._apply(lambda u, m: match_from_memory(u, self.memory_index))
        self._apply(match_with_embeddings)

        return {
            "mapping": self.final_mapping,
            "unmatched_uploaded": self.unmatched_uploaded,
            "unmatched_master": self.unmatched_master,
        }

