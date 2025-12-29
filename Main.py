import re
import math
import logging
import requests
from datetime import datetime, timezone
from dateutil import parser
from typing import List, Dict, Any

# Configure Production Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - VerPoint - %(levelname)s - %(message)s')

class VerPointMaster:
    """
    VerPoint AI Framework v1.7 (V40 Master)
    Integrated Adversarial Information Integrity Engine
    """
    def __init__(self, tavily_key: str = None, serper_key: str = None):
        self.version = "1.7.0 (V40 Master)"
        self.tavily_key = tavily_key
        self.serper_key = serper_key
        
        # V40: Comprehensive Alias Normalization Map
        self.alias_map = {
            "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
            "fed": "federal reserve", "the fed": "federal reserve",
            "cpi": "consumer price index", "gdp": "gross domestic product",
            "trump": "donald trump", "harris": "kamala harris", "biden": "joe biden",
            "plc": "programmable logic controller", "scada": "scada system"
        }
        
        # V40: Authority Tier Multipliers
        self.tiers = {
            "critical": 2.5,   # .gov, .edu, Reuters, Bloomberg
            "specialist": 1.5, # Nature, CoinDesk, NIH
            "news": 1.2,       # NYT, WSJ, BBC
            "low": 0.3         # Reddit, X, personal blogs
        }

        # V40: Subtle Contradiction Library
        self.contradiction_markers = [
            "evidence does not support", "lacks substantiation", "debunked",
            "considered misleading", "data contradicts", "assertion is questionable",
            "experts reject", "contrary to", "misinformation", "is not true"
        ]

    # --- CORE UTILITIES ---
    def _normalize(self, text: str) -> str:
        """Applies entity normalization for consistent auditing."""
        text = text.lower().strip()
        for alias, canonical in self.alias_map.items():
            text = re.sub(rf"\b{alias}\b", canonical, text)
        return text

    def _get_tier_multiplier(self, url: str) -> float:
        """Determines authority weight based on source domain."""
        url = url.lower()
        if any(d in url for d in [".gov", ".edu", "reuters.com", "bloomberg.com", "apnews.com"]): return self.tiers["critical"]
        if any(d in url for d in ["nature.com", "coindesk.com", "nih.gov", "sciencedirect.com"]): return self.tiers["specialist"]
        if any(d in url for d in ["nytimes.com", "wsj.com", "theguardian.com", "bbc.co.uk"]): return self.tiers["news"]
        if any(d in url for d in ["reddit.com", "x.com", "quora.com", "substack.com"]): return self.tiers["low"]
        return 0.8 # General web default

    def _calculate_recency(self, date_str: str) -> float:
        """Applies exponential decay for temporal weighting."""
        if not date_str: return 0.5 # Default for undated sources
        try:
            source_date = parser.parse(date_str).replace(tzinfo=timezone.utc)
            days_old = (datetime.now(timezone.utc) - source_date).days
            return math.exp(-0.05 * max(0, days_old)) # Decay curve
        except: return 0.5

    # --- ENGINE LOGIC ---
    def audit_claim(self, claim: str, evidence: List[Dict]) -> Dict:
        """Runs the VerPoint gauntlet on a single atomic claim."""
        score = 0.30 # Skeptical Base Score
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', claim)
        normalized_claim = self._normalize(claim)
        
        for item in evidence:
            snippet = item.get('content', '').lower()
            recency = self._calculate_recency(item.get('published_date'))
            authority = self._get_tier_multiplier(item.get('url', ''))
            weight = recency * authority
            
            # Support Detection
            entity_hits = sum(1 for e in entities if self._normalize(e) in snippet)
            if entities and entity_hits >= len(entities) * 0.5:
                score += 0.15 * weight
            
            # Contradiction Detection (Proximity & Verbal Negation)
            for marker in self.contradiction_markers:
                if re.search(rf"\b{marker}\b.{{0,60}}\b({normalized_claim[:15]})\b", snippet):
                    score -= 0.40 * authority # Critical sources penalize harder

        return {"claim": claim, "score": round(max(0.0, min(1.0, score)), 2)}

    def run_master_audit(self, response_text: str):
        """Orchestrates the complete V40 verification pipeline."""
        # 1. Decomposition (Mock - typically requires a simple NLP split)
        claims = [s.strip() for s in response_text.split('.') if len(s) > 10]
        
        # 2. Evidence Retrieval (Placeholder for Tavily/Serper API call)
        # evidence = tavily.search(query=response_text)
        evidence = [] # Loaded from API
        
        # 3. Gauntlet Execution
        results = [self.audit_claim(c, evidence) for c in claims]
        
        # 4. Cross-Claim Consistency Check
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
        conflicts = len(set(numbers)) < len(numbers) # Basic logic
        
        integrity_score = sum(r['score'] for r in results) / len(results) if results else 0
        if conflicts: integrity_score *= 0.85 # Conflict penalty
        
        return {
            "version": self.version,
            "overall_integrity": round(integrity_score, 2),
            "claim_ledger": results,
            "timestamp": datetime.now().isoformat()
        }
