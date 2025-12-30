import re
import math
import logging
import json
from datetime import datetime, timezone
from dateutil import parser
from typing import List, Dict, Any

# Configure Production Logging for VerPoint AI
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - VerPoint AI [v40.1] - %(levelname)s - %(message)s'
)

class VerPointAI:
    """
    VerPoint AI Framework v1.7 (Internal v40.1 Master)
    Adversarial Information Integrity & Heuristic Audit Engine
    Purpose: Information -> Intelligence -> Wealth
    """
    def __init__(self, tavily_key: str = None, serper_key: str = None):
        self.version = "1.7.0"
        self.internal_v = "40.1"
        self.brand = "VerPoint AI"
        self.tavily_key = tavily_key
        self.serper_key = serper_key
        
        # V40.1: Entity Normalization (Prevents 'Identity' Hallucinations)
        self.alias_map = {
            "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
            "fed": "federal reserve", "the fed": "federal reserve",
            "cpi": "consumer price index", "gdp": "gross document product",
            "trump": "donald trump", "harris": "kamala harris", "biden": "joe biden",
            "plc": "programmable logic controller", "scada": "scada system"
        }
        
        # V40.1: Authority Tier Weighting
        self.tiers = {
            "critical": 2.5,   # .gov, .edu, Tier 1 Finance (Bloomberg/Reuters)
            "specialist": 1.5, # Industry specific journals/official whitepapers
            "news": 1.2,       # Established global news organizations
            "low": 0.3         # Unverified social media or personal blogs
        }

        # V40.1: Hardened Adversarial Contradiction Library
        self.contradiction_markers = [
            "evidence does not support", "lacks substantiation", "debunked",
            "considered misleading", "data contradicts", "assertion is questionable",
            "experts reject", "contrary to", "misinformation", "is not true",
            "false", "inaccurate", "unproven", "claims are exaggerated"
        ]

    def _normalize(self, text: str) -> str:
        """Standardizes terminology for cross-reference accuracy."""
        text = text.lower().strip()
        for alias, canonical in self.alias_map.items():
            text = re.sub(rf"\b{alias}\b", canonical, text)
        return text

    def _get_tier_multiplier(self, url: str) -> float:
        """Determines authority weight based on source domain reliability."""
        url = url.lower()
        if any(d in url for d in [".gov", ".edu", "reuters.com", "bloomberg.com", "apnews.com"]): 
            return self.tiers["critical"]
        if any(d in url for d in ["nature.com", "coindesk.com", "nih.gov", "sciencedirect.com"]): 
            return self.tiers["specialist"]
        if any(d in url for d in ["nytimes.com", "wsj.com", "theguardian.com", "bbc.co.uk"]): 
            return self.tiers["news"]
        if any(d in url for d in ["reddit.com", "x.com", "quora.com", "substack.com"]):
            return self.tiers["low"]
        return 0.8 # Standard Web Baseline

    def _calculate_recency(self, date_str: str) -> float:
        """Applies exponential decay to information value over time."""
        if not date_str: return 0.5 
        try:
            source_date = parser.parse(date_str).replace(tzinfo=timezone.utc)
            days_old = (datetime.now(timezone.utc) - source_date).days
            # Decay curve: Half-life of relevance is ~14 days for high-volatility data
            return math.exp(-0.05 * max(0, days_old)) 
        except: return 0.5

    def audit_claim(self, claim: str, evidence: List[Dict]) -> Dict:
        """Executes the full V40.1 gauntlet on a specific data point."""
        score = 0.30 # Skeptical Base Score
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', claim)
        normalized_claim = self._normalize(claim)
        
        for item in evidence:
            snippet = item.get('content', '').lower()
            recency = self._calculate_recency(item.get('published_date'))
            authority = self._get_tier_multiplier(item.get('url', ''))
            weight = recency * authority
            
            # Support Validation
            entity_hits = sum(1 for e in entities if self._normalize(e) in snippet)
            if entities and entity_hits >= len(entities) * 0.5:
                score += 0.15 * weight
            
            # Adversarial Check
            for marker in self.contradiction_markers:
                if re.search(rf"\b{marker}\b.{{0,60}}\b({normalized_claim[:15]})\b", snippet):
                    score -= 0.45 * authority # Hardened penalty for V40.1

        return {"claim": claim, "score": round(max(0.0, min(1.0, score)), 2)}

    def run_master_audit(self, response_text: str, evidence: List[Dict] = []):
        """The primary execution pipeline for VerPoint AI."""
        # 1. Decomposition
        claims = [s.strip() for s in response_text.split('.') if len(s) > 10]
        
        # 2. Gauntlet Execution
        results = [self.audit_claim(c, evidence) for c in claims]
        
        # 3. Logical Conflict Check (Internal Consistency)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
        conflicts = len(set(numbers)) < len(numbers) 
        
        integrity_score = sum(r['score'] for r in results) / len(results) if results else 0
        if conflicts: integrity_score *= 0.80 # V40.1 heightened penalty
        
        return {
            "brand": self.brand,
            "version": self.version,
            "internal_v": self.internal_v,
            "overall_integrity": round(integrity_score, 2),
            "claim_ledger": results,
            "timestamp": datetime.now().isoformat()
        }

# --- MASTER INITIALIZATION ---
if __name__ == "__main__":
    vp = VerPointAI()
    print(f"--- {vp.brand} SYSTEM ACTIVE ---")
    print(f"VERSION: {vp.version} | INTERNAL: {vp.internal_v}")

