"""Structured data extraction engine for shipping broker emails.

This module provides dedicated extraction functions for:
1. Tonnage records (vessel availability reports).
2. Voyage Charter (VC) cargo requirements.
3. Time Charter (TC) requirement details.

Each extractor scans the email body and returns a list of dictionaries
containing the structured data fields.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

MONTH_MAP = {
    "JAN": "JAN", "JANUARY": "JAN",
    "FEB": "FEB", "FEBRUARY": "FEB",
    "MAR": "MAR", "MARCH": "MAR",
    "APR": "APR", "APRIL": "APR",
    "MAY": "MAY",
    "JUN": "JUN", "JUNE": "JUN",
    "JUL": "JUL", "JULY": "JUL",
    "AUG": "AUG", "AUGUST": "AUG",
    "SEP": "SEP", "SEPT": "SEP", "SEPTEMBER": "SEP",
    "OCT": "OCT", "OCTOBER": "OCT",
    "NOV": "NOV", "NOVEMBER": "NOV",
    "DEC": "DEC", "DECEMBER": "DEC",
}

MONTH_RE = (
    r"JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|"
    r"JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|"
    r"OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?"
)

VESSEL_TYPE_KEYWORDS = [
    "SUPRAMAX", "ULTRAMAX", "PANAMAX", "KAMSARMAX",
    "HANDYSIZE", "HANDYMAX", "HMAX", "CAPESIZE", "CAPE", "MINI CAPESIZE",
    "POST-PANAMAX", "POST PANAMAX", "NEWCASTLEMAX", "VLOC", "VLCC",
    "BULK CARRIER", "SDSTBC", "SDBC",
]

CARGO_TYPE_KEYWORDS = {
    "BULK": [
        "COAL", "GRAIN", "IRON ORE", "BAUXITE", "FERTILIZER",
        "CEMENT", "CLINKER", "LIMESTONE", "SALT", "SUGAR",
        "WHEAT", "CORN", "SOYBEAN", "RICE", "NICKEL ORE",
        "MANGANESE", "PETCOKE", "GYPSUM", "PHOSPHATE", "SCRAP",
        "UREA", "SLAG", "IRON SLAG", "HRC", "STEEL",
        "MOLOCHOPT", "GRAINS", "STEELS",
    ],
    "TANKER": ["CRUDE OIL", "FUEL OIL", "DIESEL", "GASOLINE", "LNG", "LPG"],
    "CONTAINER": ["CONTAINERS", "BOXES", "FCLS"],
    "GENERAL": ["GENERAL CARGO", "BREAKBULK", "PIPES", "EQUIPMENT", "GENS", "LAWFULS"],
}


def _normalise_dwt(raw: str) -> str:
    """Normalizes the deadweight tonnage (DWT) string to a standard numeric string.

    Args:
        raw (str): The raw DWT string (e.g. "58K", "58,000").

    Returns:
        str: Normalized integer value representation.
    """
    cleaned = raw.upper().replace(",", "").replace(".", "").strip()
    match_k = re.match(r"(\d+(?:\.\d+)?)\s*K", cleaned)
    if match_k:
        return str(int(float(match_k.group(1)) * 1000))
    match_int = re.match(r"(\d+)", cleaned)
    if match_int:
        return match_int.group(1)
    return cleaned


def _normalise_date(raw: str) -> str:
    """Normalizes a raw date string by mapping months to uppercase three-letter abbreviations.

    Args:
        raw (str): The raw date string.

    Returns:
        str: Normalized date string.
    """
    normalized = raw.strip().upper()
    normalized = re.sub(r"(\d+)(?:ST|ND|RD|TH)\b", r"\1", normalized)
    for long_name, short_name in MONTH_MAP.items():
        normalized = re.sub(r"\b" + long_name + r"\b", short_name, normalized)
    return normalized.strip()


def _normalise_port(raw: str) -> str:
    """Cleans a raw port string by removing trailing delimiters and operational suffixes.

    Args:
        raw (str): The raw port name.

    Returns:
        str: Cleaned and normalized port name.
    """
    cleaned = raw.strip().upper()
    cleaned = re.sub(r"\s+(?:O/?A|ONW|EX\b).*$", "", cleaned)
    cleaned = cleaned.rstrip(",.;:")
    return cleaned.strip()


def _detect_cargo_type(cargo_name: str, text: str) -> str:
    """Infers the category class of a cargo from its name and email context.

    Args:
        cargo_name (str): The extracted cargo name.
        text (str): The email block text.

    Returns:
        str: The categorized cargo type (e.g. BULK, TANKER, CONTAINER, GENERAL).
    """
    upper_cargo = cargo_name.upper()
    upper_text = text.upper()
    for category_type, keywords in CARGO_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in upper_cargo:
                return category_type
    return ""


def _extract_account(text: str) -> str:
    """Attempts to match and extract the account/company name from the email text.

    Args:
        text (str): The context text block.

    Returns:
        str: The extracted account name, or empty string if not found.
    """
    patterns = [
        r"(?:A/?C|ACC)\s+([A-Z][A-Z &.\-]+(?:\s+(?:COMPANY|CO|LTD|LIMITED|DMCC|INC|LLC|CORP|SHIPPING|MARITIME|OCEAN|BROKERS)\.?)*)",
        r"(?:ACCOUNT|CHARTERER|SHIPPER|PRINCIPAL|OFFERED BY|ON BEHALF OF|OWNER)\s*[:\-]?\s*([A-Z][A-Z &.\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip().upper()
            extracted = re.sub(r"\s+$", "", extracted)
            extracted = re.sub(r"^ACC\s+", "", extracted)
            return extracted
    return ""


_INLINE_VESSEL_PATTERN = re.compile(
    r"(?:MV|M\.V\.)\s+"
    r"([A-Z][A-Z0-9 \-]+?)"
    r"\s+(?:DWT\s*)?(\d[\d,\.]+)\s*K?"
    r"(?:\s*(?:MT(?:DW)?|DWT))?"
    r"\s+OPEN\s+"
    r"([A-Z][A-Z ,\.]+?)"
    r"\s+O/?A\s+"
    r"((?:\d{1,2}(?:ST|ND|RD|TH)?\s+)?(?:" + MONTH_RE + r")(?:\s+\d{4})?)",
    re.IGNORECASE | re.VERBOSE,
)

_COMPACT_VESSEL_PATTERN = re.compile(
    r"(?:MV|M\.V\.)\s+"
    r"([A-Z][A-Z0-9 \-]+?)"
    r"\s*/\s*"
    r"(\d+)K\s*"
    r"/?\s*"
    r"(?:\d{2,4}\s*-?\s*)?"
    r"((?:[A-Z][A-Z ,\.]+?))"
    r"\s,?\s+"
    r"((?:\d{1,2}(?:ST|ND|RD|TH)?\s+)?(?:" + MONTH_RE + r")(?:\s+\w+)?)",
    re.IGNORECASE,
)

_VESSEL_TYPE_PATTERN = re.compile(
    "|".join(re.escape(vt) for vt in VESSEL_TYPE_KEYWORDS),
    re.IGNORECASE,
)


def extract_tonnage(text: str) -> List[Dict[str, Any]]:
    """Scans and extracts vessel availability details from a tonnage email.

    Args:
        text (str): The raw email text.

    Returns:
        List[Dict[str, Any]]: Extracted structured tonnage records.
    """
    upper_text = text.upper()
    records: List[Dict[str, Any]] = []
    used_positions: set = set()

    for match in _INLINE_VESSEL_PATTERN.finditer(text):
        vessel_name = match.group(1).strip().upper()
        dwt_raw = match.group(2).replace(",", "").replace(".", "")
        port = _normalise_port(match.group(3))
        date = _normalise_date(match.group(4))

        port_city = port.split(",")[0].strip()

        records.append({
            "vessel_name": vessel_name,
            "account_name": _extract_account(text),
            "open_port": port_city,
            "open_date": date,
            "vessel_type": "",
            "vessel_size": dwt_raw,
        })
        used_positions.add(match.start())

    for match in _COMPACT_VESSEL_PATTERN.finditer(text):
        if match.start() in used_positions:
            continue
        vessel_name = match.group(1).strip().upper()
        dwt = str(int(match.group(2)) * 1000)
        port = _normalise_port(match.group(3))
        date = _normalise_date(match.group(4))

        records.append({
            "vessel_name": vessel_name,
            "account_name": _extract_account(text),
            "open_port": port,
            "open_date": date,
            "vessel_type": "",
            "vessel_size": dwt,
        })
        used_positions.add(match.start())

    vessel_anchors = []
    for match_name in re.finditer(
        r"(?:MV|M\.V\.|VESSEL\s*[:\-]|NAME\s*[:\-]|M/V[:\s])\s*([A-Z][A-Z0-9 \-]+)",
        upper_text,
    ):
        already_used = False
        for used_pos in used_positions:
            if abs(match_name.start() - used_pos) < 5:
                already_used = True
                break
        if not already_used:
            vessel_anchors.append((match_name.start(), match_name.group(1).strip()))

    for idx, (anchor_pos, vessel_name) in enumerate(vessel_anchors):
        block_end = vessel_anchors[idx + 1][0] if idx + 1 < len(vessel_anchors) else len(upper_text)
        block = upper_text[anchor_pos:block_end]

        record: Dict[str, Any] = {
            "vessel_name": vessel_name.strip(),
            "account_name": "",
            "open_port": "",
            "open_date": "",
            "vessel_type": "",
            "vessel_size": "",
        }

        open_patterns = [
            r"OPEN\s+([A-Z][A-Z ,\.]*?)\s+((?:\d{1,2}(?:\s*[-/]\s*\d{1,2})?\s+)?(?:" + MONTH_RE + r")(?:\s+\d{4})?)",
            r"OPEN\s*(?:PORT)?\s*[:\-]?\s*([A-Z][A-Z ,\.]+?)(?:\s+((?:\d{1,2}(?:ST|ND|RD|TH)?\s+)?(?:" + MONTH_RE + r"))|\s*$|\s*\n)",
            r"CURRENTLY\s+OPEN\s+AT\s+([A-Z][A-Z ,\.]+?)(?:\s*$|\s*\n)",
        ]
        for pattern in open_patterns:
            open_match = re.search(pattern, block)
            if open_match:
                record["open_port"] = _normalise_port(open_match.group(1))
                record["open_port"] = record["open_port"].split(",")[0].strip()
                if open_match.lastindex and open_match.lastindex >= 2 and open_match.group(2):
                    record["open_date"] = _normalise_date(open_match.group(2))
                break

        if not record["open_date"]:
            date_match = re.search(
                r"(?:DATE\s*(?:OPEN)?\s*[:\-]?\s*|O/?A\s+)"
                r"((?:\d{1,2}(?:ST|ND|RD|TH)?\s+)?(?:" + MONTH_RE + r")(?:\s+\d{4})?)",
                block,
            )
            if date_match:
                record["open_date"] = _normalise_date(date_match.group(1))

        dwt_patterns = [
            r"(\d{2,3}(?:[,\.]\d{3})?(?:\.\d+)?)\s*K?\s*(?:MT\s*)?DWT",
            r"DWT\s*[:\-]?\s*(\d{2,3}(?:[,\.]\d{3})?(?:\.\d+)?)\s*K?",
            r"DWT\s*[:\-]?\s*(\d{4,6}(?:\.\d+)?)\s*(?:MT)?",
            r"(\d{4,6}(?:\.\d+)?)\s*MT\s*(?:DW|DWT)",
            r"(\d{2,3})K\s*DWT",
        ]
        for pattern in dwt_patterns:
            dwt_match = re.search(pattern, block)
            if dwt_match:
                raw_val = dwt_match.group(1).replace(",", "")
                if re.search(r"\d+K", dwt_match.group(0)):
                    record["vessel_size"] = str(int(float(raw_val) * 1000))
                elif "." in raw_val and float(raw_val) < 300:
                    record["vessel_size"] = str(int(float(raw_val) * 1000))
                else:
                    record["vessel_size"] = raw_val.split(".")[0]
                break

        type_match = _VESSEL_TYPE_PATTERN.search(block)
        if type_match:
            record["vessel_type"] = type_match.group(0).upper()

        record["account_name"] = _extract_account(block) or _extract_account(text)

        if record["vessel_name"] or record["open_port"] or record["vessel_size"]:
            existing_names = {rec["vessel_name"] for rec in records}
            if record["vessel_name"] not in existing_names:
                records.append(record)

    return records


_LAYCAN_PATTERN = re.compile(
    r"(?:LAYCAN|LAYDAYS|WINDOW|LAYCAN\s*RANGE|LC)\s*[:\-]?\s*"
    r"("
    r"(?:MID|EARLY|END|FULL|LATE)\s+(?:" + MONTH_RE + r")(?:\s+\d{4})?"
    r"|"
    r"(?:\d{1,2}\s*[-/]\s*\d{1,2}\s+)?(?:" + MONTH_RE + r")(?:\s+\d{4})?"
    r"|"
    r"(?:\d{1,2}\s+)?(?:" + MONTH_RE + r")\s*(?:[-/]\s*(?:\d{1,2}\s+)?(?:" + MONTH_RE + r"))?(?:\s+\d{4})?"
    r"|PROMPT|IMMEDIATE"
    r")"
    r"(?:\s+(?:TRY\s+LATER))?",
    re.IGNORECASE,
)

_LOAD_PORT_PATTERN = re.compile(
    r"(?:LOAD(?:ING)?\s*(?:PORT)?|LOAD\s*PORT|LP|POL|FROM)\s*[:\-]?\s*"
    r"([A-Z][A-Z ,.]+?)"
    r"(?:\n|$|\s{2,}|(?:\s*,\s*[A-Z]+\s*$))",
    re.IGNORECASE,
)

_DISCH_PORT_PATTERN = re.compile(
    r"(?:DISCH(?:ARGE)?(?:ING)?\s*(?:PORT)?|DISCHARGE\s*PORT|DP|POD|TO)\s*[:\-]?\s*"
    r"([A-Z][A-Z ,.\+]+?)"
    r"(?:\n|$|\s{2,})",
    re.IGNORECASE,
)

_CARGO_NAME_PATTERN = re.compile(
    r"(?:CARGO|COMMODITY|PRODUCT)\s*[:\-]?\s*(?:\d[\d,. ]*\s*(?:MTS?|MT)\s+(?:OF\s+)?)?([A-Z][A-Z0-9 /]+?)(?:\s+IN\s+BULK|\n|$|\s{2,}|,)",
    re.IGNORECASE,
)

_QTY_CARGO_INLINE = re.compile(
    r"(\d[\d,. ]+)\s*(?:[-/]\s*\d[\d,. ]+\s*)?(?:MTS?|MT|TONS?)\s+"
    r"(?:\d+\s*PCT\s+)?"
    r"([A-Z][A-Z ]+?)(?:\s+(?:IN\s+BULK|MAX|FIOS|FHINC|CQD)|\n|$)",
    re.IGNORECASE,
)

_PORT_PAIR_INLINE = re.compile(
    r"([A-Z][A-Z]+(?:\s+[A-Z]+)?)\s*/\s*([A-Z][A-Z]+(?:\s+[A-Z]+)?)",
    re.IGNORECASE,
)


def _split_vc_blocks(text: str) -> List[str]:
    """Divides voyage charter email text into individual requirement blocks.

    Args:
        text (str): Raw multi-cargo email text.

    Returns:
        List[str]: Individual structured blocks.
    """
    blocks = re.split(
        r"(?:\n\s*\+{4,}\s*\n|\n\s*-{4,}\s*\n|\n\s*={4,}\s*\n)",
        text,
    )
    if len(blocks) > 1:
        return [b for b in blocks if b.strip() and len(b.strip()) > 20]
    return [text]


def extract_cargo_vc(text: str) -> List[Dict[str, Any]]:
    """Scans and extracts voyage charter cargo details from a VC cargo email.

    Args:
        text (str): The raw email text.

    Returns:
        List[Dict[str, Any]]: Extracted structured voyage charter cargo records.
    """
    blocks = _split_vc_blocks(text)
    records: List[Dict[str, Any]] = []

    for block in blocks:
        block_upper = block.upper()
        record: Dict[str, Any] = {
            "account_name": "",
            "cargo_name": "",
            "loading_port": "",
            "discharge_port": "",
            "laycan": "",
            "cargo_type": "",
        }

        cargo_match = _CARGO_NAME_PATTERN.search(block)
        if cargo_match:
            record["cargo_name"] = cargo_match.group(1).strip().upper()
        else:
            qty_match = _QTY_CARGO_INLINE.search(block)
            if qty_match:
                record["cargo_name"] = qty_match.group(2).strip().upper()

        if not record["cargo_name"]:
            for category_type, keywords in CARGO_TYPE_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in block_upper:
                        record["cargo_name"] = keyword
                        break
                if record["cargo_name"]:
                    break

        load_match = _LOAD_PORT_PATTERN.search(block)
        if load_match:
            port = load_match.group(1).strip()
            port = re.split(r"\s*,\s*(?:[A-Z]{2,})", port.upper())[0]
            record["loading_port"] = _normalise_port(port)

        disch_match = _DISCH_PORT_PATTERN.search(block)
        if disch_match:
            record["discharge_port"] = _normalise_port(disch_match.group(1))

        if not record["loading_port"] or not record["discharge_port"]:
            for line in block.splitlines():
                line_s = line.strip()
                pair_match = re.match(
                    r"^([A-Z][A-Z ]+?)\s*/\s*([A-Z][A-Z ,+]+?)$",
                    line_s,
                    re.IGNORECASE,
                )
                if pair_match:
                    if not record["loading_port"]:
                        record["loading_port"] = _normalise_port(pair_match.group(1))
                    if not record["discharge_port"]:
                        record["discharge_port"] = pair_match.group(2).strip().upper()
                    break

        laycan_match = _LAYCAN_PATTERN.search(block)
        if laycan_match:
            record["laycan"] = _normalise_date(laycan_match.group(1).strip())
        else:
            date_range_match = re.search(
                r"(\d{1,2}\s+(?:" + MONTH_RE + r")\s*[-/]\s*\d{1,2}\s+(?:" + MONTH_RE + r")(?:\s+\d{4})?)"
                r"|(\d{1,2}\s*[-/]\s*\d{1,2}\s+(?:" + MONTH_RE + r")(?:\s+\d{4})?)"
                r"|(?:MID|EARLY|END|LATE|FULL)\s+(?:" + MONTH_RE + r")(?:\s+\d{4})?",
                block_upper,
            )
            if date_range_match:
                record["laycan"] = _normalise_date(date_range_match.group(0).strip())

        record["account_name"] = _extract_account(block)
        record["cargo_type"] = _detect_cargo_type(record["cargo_name"], block)

        if record["cargo_name"] or record["loading_port"] or record["discharge_port"]:
            records.append(record)

    return records if records else [{}]


_DELIVERY_PATTERN = re.compile(
    r"(?:DELIVER[YI]?\s*(?:PORT)?|DELY|DEL)\s*[:\-]?\s*"
    r"(?:TO\s+(?:MAKE\s+)?)?"
    r"([A-Z][A-Z ,\.\(\)]+?)"
    r"(?:\n|$|\s{2,}|,(?:\s+[A-Z])?)",
    re.IGNORECASE,
)

_REDELIVERY_PATTERN = re.compile(
    r"(?:REDELIVERY?\s*(?:PORT)?|REDEL|RE-?DELIVERY)\s*[:\-]?\s*"
    r"([A-Z][A-Z ,\.\(\)]+?)"
    r"(?:\n|$|\s{2,}|,(?:\s+[A-Z])?)",
    re.IGNORECASE,
)

_DURATION_PATTERN = re.compile(
    r"(?:DURATION|PERIOD|HIRE\s*PERIOD|CHARTER\s*PERIOD|MINIMUM)\s*[:\-]?\s*"
    r"("
    r"(?:ABOUT\s+|ABT\s+)?"
    r"\d+(?:\s*[-\u2013TO]+\s*\d+)?\s*"
    r"(?:MONTHS?|YEARS?|DAYS?)"
    r")"
    r"(?:\s+(?:MINIMUM|WOG))?",
    re.VERBOSE | re.IGNORECASE,
)

_TCT_CARGO_PATTERN = re.compile(
    r"(\d+)\s*TCT\s+WITH\s+([A-Z][A-Z /]+?)(?:\s+TO\b|\n|$|\s{2,}|\.)",
    re.IGNORECASE,
)

_DURATION_INLINE = re.compile(
    r"(?:ABT|ABOUT)\s+(\d+(?:\s*[-/]\s*\d+)?)\s*(DAYS?|MONTHS?|YEARS?)\s*(?:WOG)?",
    re.IGNORECASE,
)

_PERIOD_INLINE = re.compile(
    r"(\d+(?:\s*[-\u2013TO]+\s*\d+)?)\s*(MONTHS?|YEARS?|DAYS?)",
    re.IGNORECASE,
)


def _split_tc_blocks(text: str) -> List[str]:
    """Divides time charter email text into individual requirement blocks.

    Args:
        text (str): Raw multi-cargo email text.

    Returns:
        List[str]: Individual structured blocks.
    """
    blocks = re.split(
        r"(?:\n\s*\+{4,}\s*\n|\n\s*-{20,}\s*\n|\n\s*={4,}\s*\n"
        r"|\u2014-{20,}\s*\n)",
        text,
    )
    if len(blocks) > 1:
        return [b for b in blocks if b.strip() and len(b.strip()) > 20]
    return [text]


def extract_cargo_tc(text: str) -> List[Dict[str, Any]]:
    """Scans and extracts time charter cargo details from a TC cargo email.

    Args:
        text (str): The raw email text.

    Returns:
        List[Dict[str, Any]]: Extracted structured time charter cargo records.
    """
    blocks = _split_tc_blocks(text)
    records: List[Dict[str, Any]] = []

    for block in blocks:
        block_upper = block.upper()

        record: Dict[str, Any] = {
            "account_name": "",
            "cargo_name": "",
            "cargo_type": "",
            "charter_type": "",
            "delivery_port": "",
            "redelivery_port": "",
            "duration": "",
            "laycan": "",
            "vessel_size": "",
            "vessel_type": "",
        }

        tct_match = _TCT_CARGO_PATTERN.search(block)
        if tct_match:
            record["cargo_name"] = tct_match.group(2).strip().upper()
        else:
            cargo_match = _CARGO_NAME_PATTERN.search(block)
            if cargo_match:
                record["cargo_name"] = cargo_match.group(1).strip().upper()

        if not record["cargo_name"]:
            for category_type, keywords in CARGO_TYPE_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in block_upper:
                        record["cargo_name"] = keyword
                        break
                if record["cargo_name"]:
                    break

        if re.search(r"\bTCT\b", block_upper):
            record["charter_type"] = "TCT"
        elif re.search(r"\bPERIOD\b", block_upper):
            record["charter_type"] = "PERIOD"

        del_match = _DELIVERY_PATTERN.search(block)
        if del_match:
            port = del_match.group(1).strip()
            port = re.sub(r"^TM\s+", "", port, flags=re.IGNORECASE)
            port = re.sub(r"\s*,?\s*(?:E\s+KALI|OF\s+INDONESIA).*$", "", port, flags=re.IGNORECASE)
            record["delivery_port"] = port.strip()

        redel_match = _REDELIVERY_PATTERN.search(block)
        if redel_match:
            port = redel_match.group(1).strip()
            record["redelivery_port"] = port.strip()

        duration_match = _DURATION_PATTERN.search(block)
        if duration_match:
            record["duration"] = duration_match.group(1).strip().upper()
        else:
            dur_inline = _DURATION_INLINE.search(block)
            if dur_inline:
                record["duration"] = f"ABT {dur_inline.group(1).strip()} {dur_inline.group(2).strip().upper()}"
            else:
                period_match = _PERIOD_INLINE.search(block)
                if period_match:
                    record["duration"] = f"{period_match.group(1).strip()} {period_match.group(2).strip().upper()}"

        laycan_match = _LAYCAN_PATTERN.search(block)
        if laycan_match:
            record["laycan"] = _normalise_date(laycan_match.group(1).strip())
        else:
            date_match = re.search(
                r"(?:FULL|MID|EARLY|END|LATE)\s+(?:" + MONTH_RE + r")(?:\s+\d{4})?",
                block_upper,
            )
            if date_match:
                record["laycan"] = _normalise_date(date_match.group(0).strip())
            else:
                date_range = re.search(
                    r"(\d{1,2}(?:\s*[-/]\s*\d{1,2})?\s+(?:" + MONTH_RE + r")(?:\s+\d{4})?)",
                    block_upper,
                )
                if date_range:
                    record["laycan"] = _normalise_date(date_range.group(1).strip())

        dwt_patterns = [
            r"(\d{2,3}(?:[,\.]\d{3})?(?:\.\d+)?)\s*K?\s*(?:MT\s*)?DWT",
            r"DWT\s*[:\-]?\s*(\d{2,3}(?:[,\.]\d{3})?(?:\.\d+)?)\s*K?",
            r"DWT\s*[:\-]?\s*(\d{4,6}(?:\.\d+)?)\s*(?:MT)?",
            r"(\d{4,6}(?:\.\d+)?)\s*MT\s*(?:DW|DWT)",
            r"(\d{2,3})K\s*DWT",
            r"\b(\d{2,3})K\b",
        ]
        for pattern in dwt_patterns:
            dwt_match = re.search(pattern, block_upper)
            if dwt_match:
                raw_val = dwt_match.group(1).replace(",", "")
                if re.search(r"\d+K", dwt_match.group(0)):
                    record["vessel_size"] = str(int(float(raw_val) * 1000))
                elif "." in raw_val and float(raw_val) < 300:
                    record["vessel_size"] = str(int(float(raw_val) * 1000))
                else:
                    record["vessel_size"] = raw_val.split(".")[0]
                break

        type_match = _VESSEL_TYPE_PATTERN.search(block)
        if type_match:
            record["vessel_type"] = type_match.group(0).upper()

        record["account_name"] = _extract_account(block)
        record["cargo_type"] = ""  # Do not invent cargo_type for TC

        if record["delivery_port"] or record["redelivery_port"] or record["cargo_name"] or record["charter_type"]:
            records.append(record)

    return records if records else [{}]


def extract_records(category: str, text: str) -> List[Dict[str, Any]]:
    """Routes the parsing request to the correct category extractor.

    Args:
        category (str): Classified category ('tonnage', 'cargo_vc', or 'cargo_tc').
        text (str): Raw email text to process.

    Returns:
        List[Dict[str, Any]]: List of extracted structured records.
    """
    if category == "tonnage":
        return extract_tonnage(text)
    elif category == "cargo_vc":
        return extract_cargo_vc(text)
    elif category == "cargo_tc":
        return extract_cargo_tc(text)
    return []
