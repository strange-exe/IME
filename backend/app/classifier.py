"""Hybrid classification engine for shipping emails.

This module provides classification capabilities using a two-layer hybrid
approach:
1. Rule-based path leveraging pre-defined weighted keyword signals.
2. Machine Learning path utilizing a TF-IDF + LogisticRegression pipeline.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

from sklearn.pipeline import Pipeline


_TONNAGE_SIGNALS = [
    (r"\bMV\b", 3),
    (r"\bM\.V\.", 3),
    (r"\bM/V\b", 3),
    (r"\bOPEN\b", 2),
    (r"\bDWT\b", 3),
    (r"\bMTDW\b", 3),
    (r"\bSUPRAMAX\b", 3),
    (r"\bULTRAMAX\b", 3),
    (r"\bPANAMAX\b", 3),
    (r"\bKAMSARMAX\b", 3),
    (r"\bHANDYSIZE\b", 3),
    (r"\bHANDYMAX\b", 3),
    (r"\bCAPESIZE\b", 3),
    (r"\bSDBC\b", 3),
    (r"\bSDSTBC\b", 3),
    (r"\bVESSEL\b", 1),
    (r"\bTONNAGE\b", 2),
    (r"\bOPEN POSITION\b", 3),
    (r"\b\d{2,3}K?\s*DWT\b", 3),
    (r"\bBUILT\s+\d{4}\b", 2),
    (r"\bBLT\b", 2),
    (r"\bSEEKING EMPLOYMENT\b", 3),
    (r"\bOPEN PORT\b", 3),
    (r"\bOPEN DATE\b", 3),
    (r"\bO/A\b", 3),
    (r"\bOWS\s+OPEN\b", 4),
    (r"\bPPSE\s+SUIT\b", 3),
    (r"\bTONNAGE\s+LIST\b", 4),
    (r"\bFLAG\b", 1),
    (r"\bHO/HA\b", 2),
    (r"\bGRAIN\s*CAP\b", 2),
    (r"\bLOA\b", 1),
    (r"\bBEAM\b", 1),
    (r"\bSCRUBBER\b", 2),
    (r"\bSSW\b", 1),
]

_CARGO_VC_SIGNALS = [
    (r"\bVOYAGE CHARTER\b", 4),
    (r"\bVC\b", 2),
    (r"\bV/?C\b", 2),
    (r"\bVC\s+CARGO\b", 4),
    (r"\bLOAD PORT\b", 3),
    (r"\bLOADING PORT\b", 3),
    (r"\bLOADING:\s", 2),
    (r"\bLP\s*:", 3),
    (r"\bPOL\s*:", 3),
    (r"\bDISCHARGE PORT\b", 3),
    (r"\bDISCHARGING PORT\b", 3),
    (r"\bDISCH(ARGE)?:\s", 2),
    (r"\bDP\s*:", 3),
    (r"\bPOD\s*:", 3),
    (r"\bFROM\b.*\bTO\b", 2),
    (r"\bLAYCAN\b", 2),
    (r"\bWORLDSCALE\b", 3),
    (r"\bFREIGHT\b", 2),
    (r"\bLUMPSUM\b", 2),
    (r"\bCOAL\b", 1),
    (r"\bGRAIN\b", 1),
    (r"\bIRON ORE\b", 1),
    (r"\bCARGO ENQUIRY\b", 2),
    (r"\bCARGO OFFER\b", 2),
    (r"\bFIRM CARGO\b", 3),
    (r"\bOFFER FIRM\b", 3),
    (r"\bQUANTITY\b", 2),
    (r"\bQTY\b", 2),
    (r"\bMTS\b", 2),
    (r"\bCOMMODITY\b", 1),
    (r"\bFIOS\b", 3),
    (r"\bPWWD\b", 3),
    (r"\bSSHEX\b", 2),
    (r"\bLOAD\s+RATE\b", 3),
    (r"\bDISCHARGE\s+RATE\b", 3),
    (r"\bPCT\s+TTL\b", 1),
]

_CARGO_TC_SIGNALS = [
    (r"\bTIME CHARTER\b", 5),
    (r"\bT/?C\b", 2),
    (r"\bTCT\b", 4),
    (r"\bTC\s+CARGO\b", 4),
    (r"\bPERIOD CHARTER\b", 4),
    (r"\bDELIVERY\b", 3),
    (r"\bDELY\b", 3),
    (r"\bREDELIVERY\b", 5),
    (r"\bREDEL\b", 5),
    (r"\bDURATION\b", 2),
    (r"\bCHARTER PERIOD\b", 3),
    (r"\bHIRE PERIOD\b", 3),
    (r"\bPERIOD\b", 1),
    (r"\bMONTHS?\b", 1),
    (r"\bYEARS?\b", 1),
    (r"\bDAYS\s+WOG\b", 3),
    (r"\bT\.C\.\b", 3),
    (r"\bON HIRE\b", 3),
    (r"\bOFF HIRE\b", 3),
    (r"\bHIRE RATE\b", 3),
    (r"\bDELIVERY PORT\b", 4),
    (r"\bREDELIVERY PORT\b", 5),
    (r"\bCHARTERER\b", 2),
    (r"\bPRINCIPAL\b", 1),
    (r"\bADDCOM\b", 3),
    (r"\bADDOM\b", 3),
    (r"\bTCT\s+WITH\b", 5),
    (r"\bSMX[-/]UMX\b", 2),
    (r"\bDELY\s+(?:TO\s+MAKE|WW)\b", 4),
]

_SIGNAL_GROUPS = {
    "tonnage": _TONNAGE_SIGNALS,
    "cargo_vc": _CARGO_VC_SIGNALS,
    "cargo_tc": _CARGO_TC_SIGNALS,
}


def _rule_scores(text: str) -> Dict[str, float]:
    """Computes the raw accumulated rule-based scores for each email category.

    Args:
        text (str): The raw email content.

    Returns:
        Dict[str, float]: Mapping of category names to rule signal scores.
    """
    upper_text = text.upper()
    scores = {category: 0.0 for category in _SIGNAL_GROUPS}
    for category, signals in _SIGNAL_GROUPS.items():
        for pattern, weight in signals:
            if re.search(pattern, upper_text):
                scores[category] += weight
    return scores


def classify(text: str, pipeline: Pipeline) -> Tuple[str, float]:
    """Classifies a shipping email using a blended keyword-rule and ML approach.

    Blends rule-based keyword match normalization (55% weight) with ML model
    probability normalization (45% weight) to determine the best-fit shipping
    category and calculate confidence score.

    Args:
        text (str): The raw email text to classify.
        pipeline (Pipeline): The pre-trained scikit-learn classification pipeline.

    Returns:
        Tuple[str, float]: The classified category string and the confidence score.
    """
    rule_scores = _rule_scores(text)
    total_rule_score = sum(rule_scores.values())

    ml_probabilities = pipeline.predict_proba([text])[0]
    ml_classes = list(pipeline.classes_)
    ml_scores = {cls: prob for cls, prob in zip(ml_classes, ml_probabilities)}

    blended_scores: Dict[str, float] = {}
    for cat in ["tonnage", "cargo_vc", "cargo_tc"]:
        rule_normalized = rule_scores[cat] / (total_rule_score + 1e-9)
        ml_normalized = ml_scores.get(cat, 0.0)
        blended_scores[cat] = 0.55 * rule_normalized + 0.45 * ml_normalized

    best_category = max(blended_scores, key=blended_scores.__getitem__)
    best_blended_score = blended_scores[best_category]

    total_blended_score = sum(blended_scores.values())
    confidence = round(best_blended_score / (total_blended_score + 1e-9), 4)

    non_winning_scores = [v for k, v in rule_scores.items() if k != best_category]
    max_alternative_score = max(non_winning_scores) if non_winning_scores else 0.0

    if rule_scores[best_category] > 0.0 and rule_scores[best_category] >= 2.0 * max_alternative_score:
        confidence = min(1.0, confidence + 0.10)

    if confidence < 0.5:
        return "unclassified", round(float(confidence), 4)

    return best_category, round(float(confidence), 4)
