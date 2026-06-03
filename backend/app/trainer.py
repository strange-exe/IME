"""Model training and serialization pipeline for the Shipping Email Classifier.

Generates structured synthetic and real-world email training datasets,
fits a scikit-learn TF-IDF vectorization and LogisticRegression pipeline,
and serializes the trained model to disk.
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

VESSEL_NAMES = [
    "SEA BRIGHT", "OCEAN STAR", "PACIFIC GLORY", "MARITIME QUEEN",
    "ATLANTIC BRAVE", "STELLAR WAVE", "BLUE HORIZON", "GOLDEN GATE",
    "NORDIC PEARL", "EMPRESS OF THE SEA", "CAPE FORTUNE", "ISLAND SPIRIT",
    "TRANSOCEANIC", "DELTA TRADER", "CARGO MASTER", "BULK PIONEER",
    "HORIZON CHASER", "SEA FALCON", "HARBOR KNIGHT", "TSUNAMI",
    "ALBATROSS", "WESTWARD", "EASTERN PROMISE", "ARCTIC WOLF",
    "DISCOVERY", "ENDEAVOUR", "RESOLUTION", "CHALLENGER",
    "SARONIC CHAMPION", "SHENG AN HAI", "FENG HUI HAI", "YUANPING SEA",
    "SHENG DE HAI", "YIN HUA 1", "BI JIA SHAN", "YUANNING SEA",
    "COS ORCHID", "TRUE FRIEND", "BLUE STAR", "DE SHENG HAI",
    "AN DING HAI", "JIAN GUO HAI",
]

PORTS = [
    "SINGAPORE", "ROTTERDAM", "HAMBURG", "SHANGHAI", "HONG KONG",
    "BUSAN", "JAKARTA", "PORT KLANG", "COLOMBO", "MUMBAI",
    "KANDLA", "KARACHI", "DUBAI", "JEDDAH", "AQABA",
    "RICHARDS BAY", "DURBAN", "CAPE TOWN", "LAGOS", "TEMA",
    "NEW YORK", "HOUSTON", "NEW ORLEANS", "SANTOS", "BUENOS AIRES",
    "CALLAO", "VALPARAISO", "VANCOUVER", "MONTREAL", "ANTWERP",
    "YOKOHAMA", "TOKYO", "TIANJIN", "QINGDAO", "GUANGZHOU",
    "PASIR GUDANG", "PORT LOUIS", "FREMANTLE", "SYDNEY", "AUCKLAND",
    "VUNG ANG", "XIAMEN", "MANILA", "SAMALAJU", "CHITTAGONG",
    "GWADAR", "SOHAR", "DAR ES SALAAM", "BEJAIA", "GABES",
    "MUCURIPE", "CASABLANCA", "KOH SI CHANG", "CHENNAI",
    "BILBAO", "BUSHEHR", "DOHA", "ISKENDERUN", "BIK",
    "SANGATTA", "HAIKOU",
]

VESSEL_TYPES = [
    "SUPRAMAX", "ULTRAMAX", "PANAMAX", "HANDYSIZE", "HANDYMAX",
    "KAMSARMAX", "CAPESIZE", "POST-PANAMAX", "MINI CAPESIZE",
    "BULK CARRIER", "SDSTBC", "SDBC",
]

VESSEL_SIZES = [
    "28000", "32000", "38000", "45000", "52000", "55000",
    "58000", "60000", "63000", "65000", "75000", "80000",
    "82000", "93000", "120000", "170000", "180000",
]

CARGOES = [
    "COAL", "GRAIN", "IRON ORE", "BAUXITE", "FERTILIZER",
    "CEMENT", "CLINKER", "LIMESTONE", "SALT", "SUGAR",
    "WHEAT", "CORN", "SOYBEAN", "RICE", "NICKEL ORE",
    "MANGANESE", "PETCOKE", "GYPSUM", "PHOSPHATE", "SCRAP METAL",
    "UREA", "HRC", "IRON SLAG", "MOLOCHOPT", "STEELS",
]

DATES = [
    "01 JAN", "10 FEB", "15 MAR", "20 APR", "05 MAY",
    "15 JUN", "18 JUN", "22 JUL", "10 AUG", "25 SEP",
    "01 OCT", "15 NOV", "20 DEC", "PROMPT", "END JUN",
    "EARLY JUL", "MID AUG", "1ST JUNE", "2ND JUNE",
    "3RD JUNE", "5TH JUNE", "6TH JUNE", "25 MAY",
]

LAYCANS = [
    "01-05 JAN", "10-15 FEB", "15-20 MAR", "20-25 APR",
    "05-10 MAY", "10-15 JUN", "15-20 JUN", "20-25 JUL",
    "01-10 AUG", "15-25 SEP", "PROMPT", "01-07 OCT",
    "MID JULY", "25 JUNE - 5 JULY", "16-20 JULY", "25-30 JULY",
    "10-17 JUNE", "29-2ND JUN", "FULL MAY", "15-18 JULY",
    "21-23 JULY",
]

DURATIONS = [
    "3 MONTHS", "4-6 MONTHS", "6-9 MONTHS", "12 MONTHS",
    "4 TO 6 MONTHS", "6 TO 9 MONTHS", "2-3 MONTHS",
    "MINIMUM 3 MONTHS", "ABOUT 4 MONTHS", "1 YEAR",
    "ABT 30 DAYS WOG", "ABT 35-40 DAYS", "ABT 50-55 DAYS",
    "1-3 YEARS",
]

COMPANIES = [
    "ABC SHIPPING", "XYZ BROKER", "GLOBAL MARITIME", "OCEAN FREIGHT",
    "SEA VENTURES", "COASTAL TRADING", "PACIFIC BROKERS", "ATLANTIC MARITIME",
    "NORDIC CHARTERING", "EAST WEST SHIPPING", "PREMIER MARITIME",
    "CONTINENTAL CHARTERING", "MERIDIAN MARINE", "APEX SHIPPING",
    "BLUE WATER BROKERS", "DELTA MARITIME", "SUMMIT CHARTERING",
    "HARBOR BROKERS", "NEXUS MARITIME", "TRIDENT SHIPPING",
    "PRIME MARITIME INC", "COSCO DALIAN SHIPYARD",
    "DAI AN OCEAN SHIPPING COMPANY LIMITED", "SEASCHIFFE DMCC",
    "MCD LIDOMAR",
]


def _rand(lst: List[str]) -> str:
    """Helper method to retrieve a random item from a string list."""
    return random.choice(lst)


def _generate_tonnage_sample() -> str:
    """Generates a synthetic tonnage email snippet using pre-defined layouts.

    Returns:
        str: Synthetic email snippet representing a vessel availability report.
    """
    templates = [
        lambda: (
            f"MV {_rand(VESSEL_NAMES)}\n"
            f"OPEN {_rand(PORTS)} {_rand(DATES)}\n"
            f"{random.choice([28,32,38,45,52,55,58,63,65,75])}K DWT {_rand(VESSEL_TYPES)}\n"
            f"PLS REVERT WITH SUITABLE CARGO"
        ),
        lambda: (
            f"VESSEL: {_rand(VESSEL_NAMES)}\n"
            f"TYPE: {_rand(VESSEL_TYPES)}\n"
            f"DWT: {random.choice([28,32,38,45,52,55,58,63,65,75])}000 MT\n"
            f"OPEN: {_rand(PORTS)}\n"
            f"DATE: {_rand(DATES)}\n"
            f"SEEKING EMPLOYMENT - CONTACT {_rand(COMPANIES)}"
        ),
        lambda: (
            f"TONNAGE POSITION\n"
            f"{_rand(VESSEL_NAMES)} / {_rand(VESSEL_TYPES)} / "
            f"{random.choice([28000,32000,38000,45000,52000,55000,58000,63000,65000,75000])} DWT\n"
            f"OPEN {_rand(PORTS)} AROUND {_rand(DATES)}\n"
            f"ACCOUNT: {_rand(COMPANIES)}"
        ),
        lambda: (
            f"DEAR BROKERS,\n\n"
            f"WE HAVE THE FOLLOWING VESSEL OPEN:\n"
            f"MV {_rand(VESSEL_NAMES)}\n"
            f"SIZE: {random.choice([28,32,38,45,52,55,58,63,65,75])}K DWT\n"
            f"TYPE: {_rand(VESSEL_TYPES)}\n"
            f"CURRENTLY OPEN AT {_rand(PORTS)}\n"
            f"DATE OPEN: {_rand(DATES)}\n"
            f"PLEASE OFFER SUITABLE BUSINESS"
        ),
        lambda: (
            f"OPEN POSITION:\n"
            f"NAME: {_rand(VESSEL_NAMES)}\n"
            f"BUILT: {random.randint(2005,2022)}\n"
            f"DWT: {random.choice([28000,32000,38000,45000,52000,55000,58000,63000,65000,75000])}\n"
            f"CLASS: {_rand(VESSEL_TYPES)}\n"
            f"OPEN PORT: {_rand(PORTS)}\n"
            f"OPEN DATE: {_rand(DATES)}"
        ),
        lambda: (
            f"good day,\n\nPLS PROPOSE FOR THE BELOW TONNAGE list:\n\n"
            f"PACIFIC OCEAN\n=======================\n\n"
            f"MV {_rand(VESSEL_NAMES)} DWT {random.choice([38000,45000,52000,55000,58000,63000])} OPEN {_rand(PORTS)}, {random.choice(['CHINA','INDIA','VIETNAM','PHILIPPINES','INDONESIA'])} O/A {_rand(DATES)}\n\n"
            f"MV {_rand(VESSEL_NAMES)} DWT {random.choice([38000,45000,52000,55000,58000,63000])} OPEN {_rand(PORTS)}, {random.choice(['CHINA','INDIA','VIETNAM'])} O/A {_rand(DATES)}\n\n"
            f"INDIAN OCEAN\n=======================\n\n"
            f"MV {_rand(VESSEL_NAMES)} DWT {random.choice([38000,45000,52000,55000,58000,63000])} OPEN {_rand(PORTS)}, {random.choice(['OMAN','PAKISTAN','BANGLADESH'])} O/A {_rand(DATES)}"
        ),
        lambda: (
            f"OUR CLOSE OWS OPEN ASF\n\nPLS PPSE SUIT\n\n"
            f"MV {_rand(VESSEL_NAMES)}/{random.choice([38,45,51,55,58,63,65,75,93])}K/ {random.choice(['09','11','13','15','17'])} - {_rand(PORTS)} , {_rand(DATES)} ONW - EX OUR CP\n\n"
            f"MV {_rand(VESSEL_NAMES)}\n==============\n\n"
            f"DWT: {random.choice([38000,45000,51241,55000,58000,63000,65000])}/\n"
            f"BUILT : {random.randint(2005,2022)}\n"
            f"FLAG: {random.choice(['BARBADOS','LIBERIA','PANAMA','HONG KONG','SINGAPORE'])}\n"
            f"BULK CARRIER\n"
            f"5/5 HO/HA\n"
            f"GRAIN : ABT {random.choice([48000,59000,65000,71000,78000])}"
        ),
        lambda: (
            f"DEAR SIRS\n\nGOOD DAY\n\nOUR DIRECT OWS OPEN AS FOLLOWS\n\nPLS PPSE SUIT\n\n"
            f"PACIFIC\n=======\n\n"
            f"MV {_rand(VESSEL_NAMES)} ({random.choice([38,55,58,63,75,93])}K - SCRUBBER FITTED / {random.randint(2008,2022)} ) - OPEN {_rand(PORTS)} {_rand(DATES)}\n\n"
            f"{_rand(VESSEL_NAMES)}\n\n"
            f"{random.choice(['LIBERIA','PANAMA','HONG KONG','SINGAPORE','PRC'])} FLAG\n\n"
            f"BUILT {random.randint(2005,2022)}\n\n"
            f"CLASS {random.choice(['LR','NK','CCS','ABS','BV'])}\n\n"
            f"ABT {random.choice([38000,55000,58000,63000,75000,93116])} DWT ON ABT {random.choice(['10.50','12.80','13.30','14.90'])} MTRS SSW\n\n"
            f"BEST REGARDS"
        ),
    ]
    return random.choice(templates)()


def _generate_cargo_vc_sample() -> str:
    """Generates a synthetic voyage charter cargo email snippet.

    Returns:
        str: Synthetic email snippet representing a voyage charter requirement.
    """
    port_a, port_b = random.sample(PORTS, 2)
    templates = [
        lambda: (
            f"CARGO ENQUIRY - VOYAGE CHARTER\n"
            f"CARGO: {_rand(CARGOES)}\n"
            f"LOAD PORT: {port_a}\n"
            f"DISCHARGE PORT: {port_b}\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"QUANTITY: {random.randint(30,80)}000 MT\n"
            f"ACCOUNT: {_rand(COMPANIES)}"
        ),
        lambda: (
            f"VOYAGE CARGO AVAILABLE\n"
            f"{_rand(CARGOES)} FROM {port_a} TO {port_b}\n"
            f"LAYCAN {_rand(LAYCANS)}\n"
            f"QTY: {random.randint(30,80)}K MT\n"
            f"FREIGHT: LUMPSUM / WORLDSCALE\n"
            f"SHIPPER: {_rand(COMPANIES)}"
        ),
        lambda: (
            f"DEAR ALL,\n\n"
            f"PLEASE BE ADVISED WE HAVE FOLLOWING CARGO:\n"
            f"COMMODITY: {_rand(CARGOES)}\n"
            f"LOADING: {port_a}\n"
            f"DISCHARGING: {port_b}\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"LOT SIZE: {random.randint(30,80)}000 MT\n"
            f"CHARTERER: {_rand(COMPANIES)}\n"
            f"PLEASE OFFER SUITABLE TONNAGE"
        ),
        lambda: (
            f"VC CARGO SEEKING VESSEL\n"
            f"PRODUCT: {_rand(CARGOES)} BULK\n"
            f"FROM: {port_a}\n"
            f"TO: {port_b}\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"ACCOUNT: {_rand(COMPANIES)}"
        ),
        lambda: (
            f"CARGO OFFER\n"
            f"TYPE: VOYAGE CHARTER / BULK CARGO\n"
            f"CARGO: {_rand(CARGOES)}\n"
            f"LOAD: {port_a}\n"
            f"DISCH: {port_b}\n"
            f"WINDOW: {_rand(LAYCANS)}\n"
            f"QUANTITY: ABOUT {random.randint(30,80)}000 MT\n"
            f"OFFERED BY: {_rand(COMPANIES)}"
        ),
        lambda: (
            f"PLEASE OFFER FIRM FOR FOLL FULY FIRM CARGO\n\n"
            f"{random.randint(10,30)},000 - {random.randint(30,60)},000 MTS 10PCT {_rand(CARGOES)}\n"
            f"LOAD PORT : {port_a}\n"
            f"DISCHARGE PORT: {port_b}\n"
            f"LOAD RATE: {random.choice([1000,2000,3000,5000])} MTS PWWD SSHEX\n"
            f"DISCHARGE RATE: {random.choice([1500,3000,5000])} MTS PWWD SSHEX\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"COM : 3.75 PCT TTL"
        ),
        lambda: (
            f"{port_a} / {port_b}\n\n"
            f"{random.randint(15,30)} 000  mt {_rand(CARGOES)}  max {random.choice(['25','28','28.5'])} mt\n\n"
            f"FIOS\n\n"
            f"{random.choice([3000,4000,5000])} mt fhinc / CQD disch\n\n"
            f"{_rand(LAYCANS)} try later\n\n"
            f"3.75% here"
        ),
        lambda: (
            f"PLS OFFER FIRM FOR FOLL OUR CLOSE AND DIR CHRTRS\n\n"
            f"{random.randint(15,40)}-{random.randint(30,60)},000 mts {random.choice(['iron slag','urea','clinker','HRC'])} in bulk\n"
            f"LP:{port_a}\n"
            f"DP: {port_b}\n"
            f"{random.choice([5000,10000,12000])}/{random.choice([8000,10000,12000])}\n"
            f"{_rand(LAYCANS)}\n"
            f"3.75% TTL"
        ),
        lambda: (
            f"Cargo:{random.randint(20,50)},000 mts of {_rand(CARGOES)} in bulk\n"
            f"POL: {port_a}\n"
            f"POD: {port_b}\n"
            f"5000/5000\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"COMM:1.25% TTL"
        ),
    ]
    return random.choice(templates)()


def _generate_cargo_tc_sample() -> str:
    """Generates a synthetic time charter cargo email snippet.

    Returns:
        str: Synthetic email snippet representing a time charter requirement.
    """
    port_a, port_b = random.sample(PORTS, 2)
    templates = [
        lambda: (
            f"TIME CHARTER REQUIREMENTS\n"
            f"CARGO: {_rand(CARGOES)}\n"
            f"DELIVERY: {port_a}\n"
            f"REDELIVERY: {port_b}\n"
            f"DURATION: {_rand(DURATIONS)}\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"CHARTERER: {_rand(COMPANIES)}"
        ),
        lambda: (
            f"TC REQUIREMENT\n"
            f"SEEKING {_rand(VESSEL_TYPES)} FOR TIME CHARTER\n"
            f"DELIVERY PORT: {port_a}\n"
            f"REDELIVERY PORT: {port_b}\n"
            f"PERIOD: {_rand(DURATIONS)}\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"CARGO: {_rand(CARGOES)}\n"
            f"ACCOUNT: {_rand(COMPANIES)}"
        ),
        lambda: (
            f"DEAR OWNERS,\n\n"
            f"WE ARE SEEKING VESSEL ON TIME CHARTER BASIS:\n"
            f"DELIVERY: {port_a}\n"
            f"REDELIVERY: {port_b}\n"
            f"CHARTER PERIOD: {_rand(DURATIONS)}\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"COMMODITY: {_rand(CARGOES)}\n"
            f"PRINCIPAL: {_rand(COMPANIES)}"
        ),
        lambda: (
            f"T/C CARGO ENQUIRY\n"
            f"COMMODITY: {_rand(CARGOES)}\n"
            f"DELIVERY: {port_a}\n"
            f"REDELIVERY: {port_b}\n"
            f"HIRE PERIOD: {_rand(DURATIONS)}\n"
            f"LAYCAN: {_rand(LAYCANS)}\n"
            f"OFFERED BY {_rand(COMPANIES)}"
        ),
        lambda: (
            f"TIME CHARTER CARGO\n"
            f"PERIOD CHARTER / {_rand(DURATIONS)}\n"
            f"LOADING AREA: {port_a}\n"
            f"REDELIVERY AREA: {port_b}\n"
            f"LAYDAYS: {_rand(LAYCANS)}\n"
            f"CARGO: {_rand(CARGOES)} BULK\n"
            f"ON BEHALF OF {_rand(COMPANIES)}"
        ),
        lambda: (
            f"{random.choice(['CHINA / NOPAC','SEASIE','WORLDWIDE','ECI / MED'])}\n"
            f"ACC {_rand(COMPANIES)}\n"
            f"DELIVERY TM {port_a}\n"
            f"LC {_rand(LAYCANS)}\n"
            f"SMX-UMX, PREF UMX\n"
            f"1 TCT WITH {_rand(CARGOES)}\n"
            f"REDELIVERY {port_b}\n"
            f"3.75 ADDCOM PUS"
        ),
        lambda: (
            f"SEASIE\n"
            f"ACC {_rand(COMPANIES)}\n"
            f"SMX-UMX MAX 20 YRS.\n"
            f"DELY TO MAKE {port_a}\n"
            f"{_rand(LAYCANS)}\n"
            f"1 TCT WITH {_rand(CARGOES)} TO {random.choice(['BDESH','INDIA','CHINA'])}.\n"
            f"DURATION ABT {random.randint(25,55)} DAYS WOG.\n"
            f"3.75PCT ADDOM PUS\n"
            f"TRY PERIOD\n"
            f"PLS ADV THE OUTREACH"
        ),
        lambda: (
            f"Please provide suitable, rated vessels for our following firm requirements.\n"
            f"* A/C {_rand(COMPANIES)}\n"
            f"* 1 TCT with {_rand(CARGOES)}\n"
            f"* {random.choice([22,28,33,38,45,55,63])}k dwt upto HMAX\n"
            f"* Delivery: {port_a}\n"
            f"* Laycan: {_rand(LAYCANS)}\n"
            f"* Redel: {port_b} via {random.choice(['GOA','COGH','SUEZ'])} transit\n"
            f"* Duration: abt {random.randint(30,60)}-{random.randint(40,70)} days wog\n"
            f"* 3.75% Adc"
        ),
        lambda: (
            f"WORLDWIDE\n"
            f"ACC {_rand(COMPANIES)}\n"
            f"SUPRA/ULTRA DELY WW\n"
            f"FULL MAY\n"
            f"1-3 YEARS TRY SHORT PERIOD\n"
            f"FLAT OR INDEX BOTH WORKABLE\n"
            f"3.75 ADDCOM PUS"
        ),
    ]
    return random.choice(templates)()


REAL_TONNAGE_SAMPLES = [
    (
        "good day,\n\nPLS PROPOSE FOR THE BELOW TONNAGE list:\n\n"
        "PACIFIC OCEAN\n=======================\n\n"
        "MV SHENG AN HAI DWT 56564 OPEN XIAMEN, china O/A 2ND JUNE 2026\n\n"
        "MV FENG HUI HAI DWT 63260 OPEN GUANGZHOU, CHINA O/A 6TH JUNE 2026\n\n"
        "MV YUANPING SEA DWT 55646 OPEN MANILA, PHI O/A 3RD JUNE 2026\n\n"
        "MV SHENG DE HAI DWT 56721 OPEN SAMALAJU, MALAYSIA O/A 3RD JUNE 2026\n\n"
        "INDIAN OCEAN\n=======================\n\n"
        "MV YIN HUA 1 DWT 46613 OPEN CHITTAGONG, B.DESH O/A 5TH JUNE 2026\n\n"
        "MV BI JIA SHAN DWT 56623 OPEN GWADAR, PAKISTAN O/A 2ND JUNE 2026\n\n"
        "MV yuanning sea DWT 55580 OPEN SOHAR, OMAN O/A 30TH may 2026\n\n"
        "MV COS ORCHID DWT 55550 OPEN DAR ES SALAAM, TANZANIA O/A 1ST JUNE 2026"
    ),
    (
        "DEAR SIRS\n\nGOOD DAY\n\nOUR DIRECT OWS OPEN AS FOLLOWS\n\nPLS PPSE SUIT\n\n"
        "PACIFIC\n=======\n\n"
        "SARONIC CHAMPION (93K - SCRUBBER FITTED / 2011 ) - OPEN VUNG ANG, VIETNAM 08-12 JUNE\n\n"
        "SARONIC CHAMPION\nLIBERIA FLAG\nBUILT 2011\nCLASS LR\n"
        "ABT 93.116 DWT ON ABT 14.90 MTRS SSW (SCANTLING)\n"
        "LOA 229.253 MTRS / BEAM 38.00 MTRS\n"
        "GRAIN CAP ABT 110.330 CBM\n7/7 HO/HA - SCRUBBER FITTED"
    ),
    (
        "OUR CLOSE OWS OPEN ASF\n\nPLS PPSE SUIT\n\n"
        "MV TRUE FRIEND/51K/ 09 - BEJAIA , 1ST JUNE ONW - EX OUR CP\n\n"
        "MV TRUE FRIEND\n==============\nOPEN HATC BOX\nDWT: 51.241/\n"
        "BUILT : 2009\nFLAG: BARBADOS\nBULK CARRIER\n5/5 HO/HA\n"
        "GRAIN : ABT 59,676"
    ),
    (
        "MV BLUE STAR (38K DWT) - OPEN 25 MAY GABES, TUNISIA\n\n"
        "GEARED SELF-TRIMMING SINGLE DECK BULK CARRIER\n"
        "BUILT 2011 SAMHO SHIPBUILDING CO LTD, KOREA\nLIBERIAN FLAG\n"
        "37,947 MTDWT ON 10,63 M SSW (TPC 49,16)\nLOA 179,98 / BEAM 30,00M\n"
        "5 H/H\nCARGO HOLDS CUBIC 48.133,5/46.689,5 CBM (GRAIN/BALE)\n"
        "4 X 35MT CRANES"
    ),
    (
        "PLS PROPOSE FOR THE BELOW TONNAGE list:\n\n"
        "ECSA + W. AFRICA\n\n"
        "MV DE SHENG HAI  DWT 38,821.5 MT OPEN MUCURIPE, BRAZIL  O/A 24-25 MAY 2026\n\n"
        "CONTI+MED\n\n"
        "M/V AN DING HAI DWT 38,800 MT  - OPEN  CASABLANCA  O/A  28-30 MAY 2026"
    ),
]

REAL_CARGO_VC_SAMPLES = [
    (
        "PLEASE OFFER FIRM FOR FOLL FULY FIRM CARGO\n\n"
        "15,000 - 20,000 MTS 10PCT MOLOCHOPT\n"
        "LOAD PORT : KOH SI CHANG , THAILAND\n"
        "DISCHARGE PORT: KANDLA + CHENNAI\n"
        "LOAD RATE: 1,000 MTS PWWD SSHEX\n"
        "DISCHARGE RATE: 1500 MTS PWWD SSHEX\n"
        "LAYCAN: MID JULY 2026\n"
        "COM : 3.75 PCT TTL"
    ),
    (
        "Jeddah  / Bilbao\n\n"
        "20 000  mt HRC  max 28,5 mt\n\n"
        "FIOS\n\n4000 mt fhinc / CQD disch\n\n"
        "25 June - 5 July try later\n\n3,75% here"
    ),
    (
        "PLS OFFER FIRM FOR FOLL OUR CLOSE AND DIR CHRTRS\n\n"
        "20-30,000 mts iron slag in bulk\nLP:Bushehr\nDP: Doha\n"
        "10000/12000\n25-30 july\n3.75% TTL"
    ),
    (
        "Cargo:30,000 mts of Urea in bulk\nPOL: Bik\n"
        "POD: Iskenderun or Durban\n5000/5000\n"
        "LAYCAN: 16-20 July\nCOMM:1.25% TTL"
    ),
]

REAL_CARGO_TC_SAMPLES = [
    (
        "CHINA / NOPAC\n"
        "ACC DAI AN OCEAN SHIPPING COMPANY LIMITED\n"
        "DELIVERY TM VANCOUVER\nLC 10-17 JUNE\n"
        "SMX-UMX, PREF UMX\n1 TCT WITH GRAINS\n"
        "REDELIVERY CHITTAGONG\n3.75 ADDCOM PUS"
    ),
    (
        "SEASIE\nACC DAI AN OCEAN SHIPPING COMPANY LIMITED\n"
        "SMX-UMX MAX 20 YRS.\n"
        "DELY TO MAKE SANGATTA (NEAR TO TJ BARA), E KALI OF INDONESIA.\n"
        "29-2ND JUN\n1 TCT WITH CLINKER TO BDESH.\n"
        "DURATION ABT 30 DAYS WOG.\n3.75PCT ADDOM PUS\nTRY PERIOD"
    ),
    (
        "WORLDWIDE\nACC DAI AN OCEAN SHIPPING COMPANY LIMITED\n"
        "SUPRA/ULTRA DELY WW\nFULL MAY\n"
        "1-3 YEARS TRY SHORT PERIOD\nFLAT OR INDEX BOTH WORKABLE\n"
        "3.75 ADDCOM PUS"
    ),
    (
        "Please provide suitable, rated vessels for our following firm requirements.\n"
        "* A/C SeaSchiffe\n* 1 TCT with Steels/Gens/lawfuls\n"
        "* 22k dwt upto HMAX\n* Delivery: ECI\n* Laycan: 15-18 July\n"
        "* Redel: Med via GOA transit\n* Duration: abt 35-40 days wog\n"
        "* 3.75% Adc"
    ),
    (
        "* A/C SeaSchiffe\n* 1 TCT with Steels/Gens/lawfuls\n"
        "* 33k dwt upto HMAX\n* Delivery: ECI\n* Laycan: 21-23 July\n"
        "* Redel: ARAG via COGH transit\n* Duration: abt 50-55 days wog\n"
        "* 3.75% Adc"
    ),
]


def generate_training_data(n_per_class: int = 400) -> Tuple[List[str], List[str]]:
    """Synthesizes and compiles labelled training datasets for ML model training.

    Generates n_per_class synthetic samples per classification category and augments
    them with oversampled real-world broker email templates for higher accuracy.

    Args:
        n_per_class (int): Number of synthetic emails to generate for each category.

    Returns:
        Tuple[List[str], List[str]]: Returns a pair containing the dataset inputs
            (X, email text) and the corresponding labels (y, classification class).
    """
    random.seed(42)
    texts: List[str] = []
    labels: List[str] = []

    generators = {
        "tonnage": _generate_tonnage_sample,
        "cargo_vc": _generate_cargo_vc_sample,
        "cargo_tc": _generate_cargo_tc_sample,
    }

    for label, gen in generators.items():
        for _ in range(n_per_class):
            texts.append(gen())
            labels.append(label)

    for _ in range(3):
        for sample in REAL_TONNAGE_SAMPLES:
            texts.append(sample)
            labels.append("tonnage")
        for sample in REAL_CARGO_VC_SAMPLES:
            texts.append(sample)
            labels.append("cargo_vc")
        for sample in REAL_CARGO_TC_SAMPLES:
            texts.append(sample)
            labels.append("cargo_tc")

    return texts, labels


def train_model(n_per_class: int = 400) -> Pipeline:
    """Trains the classification pipeline (TF-IDF vectorizer + LogisticRegression).

    Performs a stratified split, validates model performance, prints the validation
    classification report metrics, and serializes the pipeline object as a pickle file.

    Args:
        n_per_class (int): Number of synthetic records to generate per class.

    Returns:
        Pipeline: The trained scikit-learn pipeline instance.
    """
    logger.info("Generating training data from vocabulary pools and samples.")
    texts, labels = generate_training_data(n_per_class)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    clf = LogisticRegression(
        max_iter=1000,
        C=3.0,
        solver="lbfgs",
        random_state=42,
    )

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])

    logger.info("Fitting classification pipeline.")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    logger.info("Pipeline validation classification report:\n%s", classification_report(y_test, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info("Trained model serialized to disk successfully at %s", MODEL_PATH)
    return pipeline


def load_model() -> Pipeline:
    """Loads the serialized classification model pipeline or initiates training.

    Returns:
        Pipeline: The active scikit-learn model pipeline instance.
    """
    if MODEL_PATH.exists():
        logger.info("Deserializing classification model from %s", MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    logger.info("No serialized model found on disk. Initiating training pipeline.")
    return train_model()


if __name__ == "__main__":
    train_model()
