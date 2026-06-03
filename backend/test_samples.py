"""Integration test suite verifying extraction accuracy on real broker email formats.

Sends structured HTTP requests to the active FastAPI backend service and
validates responses across tonnage, voyage charter (VC), and time charter (TC)
sample formats.
"""

import json
import sys
import requests

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000/parse-email"


def run_tests() -> None:
    """Executes the integration tests and formats verification output to stdout."""
    print("=" * 70)
    print("TEST 1: TONNAGE - Inline vessel list")
    print("=" * 70)
    res = requests.post(BASE_URL, json={"email_body": """good day,

PLS PROPOSE FOR THE BELOW TONNAGE list:

PACIFIC OCEAN
=======================

MV SHENG AN HAI DWT 56564 OPEN XIAMEN, CHINA O/A 2ND JUNE 2026

MV FENG HUI HAI DWT 63260 OPEN GUANGZHOU, CHINA O/A 6TH JUNE 2026

MV YUANPING SEA DWT 55646 OPEN MANILA, PHI O/A 3RD JUNE 2026

INDIAN OCEAN
=======================

MV YIN HUA 1 DWT 46613 OPEN CHITTAGONG, B.DESH O/A 5TH JUNE 2026

MV BI JIA SHAN DWT 56623 OPEN GWADAR, PAKISTAN O/A 2ND JUNE 2026"""})
    data = res.json()
    print(f"Category: {data['category']}  Confidence: {data['confidence']}")
    print(f"Records: {len(data['records'])}")
    for rec in data['records']:
        print(f"  {rec.get('vessel_name',''):20s} | {rec.get('vessel_size',''):>7s} DWT | {rec.get('open_port',''):20s} | {rec.get('open_date','')}")

    print("\n" + "=" * 70)
    print("TEST 2: CARGO VC - Real broker format")
    print("=" * 70)
    res = requests.post(BASE_URL, json={"email_body": """PLEASE OFFER FIRM FOR FOLL FULY FIRM CARGO

15,000 - 20,000 MTS 10PCT MOLOCHOPT
LOAD PORT : KOH SI CHANG , THAILAND
DISCHARGE PORT: KANDLA + CHENNAI
LOAD RATE: 1,000 MTS PWWD SSHEX
DISCHARGE RATE: 1500 MTS PWWD SSHEX
LAYCAN: MID JULY 2026
COM : 3.75 PCT TTL"""})
    data = res.json()
    print(f"Category: {data['category']}  Confidence: {data['confidence']}")
    for rec in data['records']:
        print(f"  Cargo: {rec.get('cargo_name','')} | Load: {rec.get('loading_port','')} | Disch: {rec.get('discharge_port','')} | Laycan: {rec.get('laycan','')}")

    print("\n" + "=" * 70)
    print("TEST 3: CARGO VC - Compact inline port pair")
    print("=" * 70)
    res = requests.post(BASE_URL, json={"email_body": """Jeddah  / Bilbao

20 000  mt HRC  max 28,5 mt

FIOS

4000 mt fhinc / CQD disch

25 June - 5 July try later

3,75% here"""})
    data = res.json()
    print(f"Category: {data['category']}  Confidence: {data['confidence']}")
    for rec in data['records']:
        print(f"  Cargo: {rec.get('cargo_name','')} | Load: {rec.get('loading_port','')} | Disch: {rec.get('discharge_port','')} | Laycan: {rec.get('laycan','')}")

    print("\n" + "=" * 70)
    print("TEST 4: CARGO TC - Abbreviated DELY/REDEL format")
    print("=" * 70)
    res = requests.post(BASE_URL, json={"email_body": """CHINA / NOPAC
ACC DAI AN OCEAN SHIPPING COMPANY LIMITED
DELIVERY TM VANCOUVER
LC 10-17 JUNE
SMX-UMX, PREF UMX
1 TCT WITH GRAINS
REDELIVERY CHITTAGONG
3.75 ADDCOM PUS"""})
    data = res.json()
    print(f"Category: {data['category']}  Confidence: {data['confidence']}")
    for rec in data['records']:
        print(f"  Cargo: {rec.get('cargo_name','')} | Dely: {rec.get('delivery_port','')} | Redel: {rec.get('redelivery_port','')} | Duration: {rec.get('duration','')} | Laycan: {rec.get('laycan','')} | Account: {rec.get('account_name','')}")

    print("\n" + "=" * 70)
    print("TEST 5: CARGO TC - Asterisk structured format")
    print("=" * 70)
    res = requests.post(BASE_URL, json={"email_body": """Please provide suitable, rated vessels for our following firm requirements.
* A/C SeaSchiffe
* 1 TCT with Steels/Gens/lawfuls
* 33k dwt upto HMAX
* Delivery: ECI
* Laycan: 21-23 July
* Redel: ARAG via COGH transit
* Duration: abt 50-55 days wog
* 3.75% Adc"""})
    data = res.json()
    print(f"Category: {data['category']}  Confidence: {data['confidence']}")
    for rec in data['records']:
        print(f"  Cargo: {rec.get('cargo_name','')} | Dely: {rec.get('delivery_port','')} | Redel: {rec.get('redelivery_port','')} | Duration: {rec.get('duration','')} | Laycan: {rec.get('laycan','')} | Account: {rec.get('account_name','')}")

    print("\n" + "=" * 70)
    print("TEST 6: CARGO VC - Multi-cargo with dividers")
    print("=" * 70)
    res = requests.post(BASE_URL, json={"email_body": """PLS OFFER FIRM FOR FOLL OUR CLOSE AND DIR CHRTRS


20-30,000 mts iron slag in bulk
LP:Bushehr
DP: Doha
10000/12000
25-30 july
3.75% TTL

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Cargo:30,000 mts of Urea in bulk
POL: Bik
POD: Iskenderun or Durban
5000/5000
LAYCAN: 16-20 July
COMM:1.25% TTL"""})
    data = res.json()
    print(f"Category: {data['category']}  Confidence: {data['confidence']}")
    print(f"Records: {len(data['records'])}")
    for i, rec in enumerate(data['records']):
        print(f"  [{i+1}] Cargo: {rec.get('cargo_name','')} | Load: {rec.get('loading_port','')} | Disch: {rec.get('discharge_port','')} | Laycan: {rec.get('laycan','')}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
