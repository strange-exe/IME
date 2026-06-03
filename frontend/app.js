/**
 * @fileoverview Main frontend application logic for the MailParser dashboard.
 * Provides user interaction, API request management, status checks,
 * sample loading, and responsive DOM rendering for parsed shipping details.
 */

(() => {
  "use strict";

  const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "https://mailparser.backend.abhinesh.me";

  const $emailInput     = document.getElementById("emailInput");
  const $charCount      = document.getElementById("charCount");
  const $btnParse       = document.getElementById("btnParse");
  const $btnPasteSample = document.getElementById("btnPasteSample");
  const $btnClear       = document.getElementById("btnClear");

  const $loadingOverlay = document.getElementById("loadingOverlay");
  const $loadingText    = document.getElementById("loadingText");
  const $resultsSection = document.getElementById("resultsSection");

  const $categoryBadge  = document.getElementById("categoryBadge");
  const $categoryIcon   = document.getElementById("categoryIcon");
  const $categoryLabel  = document.getElementById("categoryLabel");
  const $confidenceFill = document.getElementById("confidenceFill");
  const $confidenceVal  = document.getElementById("confidenceValue");

  const $metaRecords    = document.getElementById("metaRecordsCount");
  const $metaTime       = document.getElementById("metaTimeValue");
  const $recordsBody    = document.getElementById("recordsBody");
  const $jsonOutput     = document.getElementById("jsonOutput");
  const $btnDownloadJson = document.getElementById("btnDownloadJson");

  const $statusDot      = document.getElementById("statusDot");
  const $statusText     = document.getElementById("statusText");

  const SAMPLES = [
    `good day,

PLS PROPOSE FOR THE BELOW TONNAGE list:

PACIFIC OCEAN
=======================

MV SHENG AN HAI DWT 56564 OPEN XIAMEN, CHINA O/A 2ND JUNE 2026

MV FENG HUI HAI DWT 63260 OPEN GUANGZHOU, CHINA O/A 6TH JUNE 2026

MV YUANPING SEA DWT 55646 OPEN MANILA, PHI O/A 3RD JUNE 2026

INDIAN OCEAN
=======================

MV YIN HUA 1 DWT 46613 OPEN CHITTAGONG, B.DESH O/A 5TH JUNE 2026

MV BI JIA SHAN DWT 56623 OPEN GWADAR, PAKISTAN O/A 2ND JUNE 2026`,

    `PLEASE OFFER FIRM FOR FOLL FULY FIRM CARGO

15,000 - 20,000 MTS 10PCT MOLOCHOPT
LOAD PORT : KOH SI CHANG , THAILAND
DISCHARGE PORT: KANDLA + CHENNAI
LOAD RATE: 1,000 MTS PWWD SSHEX
DISCHARGE RATE: 1500 MTS PWWD SSHEX
LAYCAN: MID JULY 2026
COM : 3.75 PCT TTL`,

    `CHINA / NOPAC
ACC DAI AN OCEAN SHIPPING COMPANY LIMITED
DELIVERY TM VANCOUVER
LC 10-17 JUNE
SMX-UMX, PREF UMX
1 TCT WITH GRAINS
REDELIVERY CHITTAGONG
3.75 ADDCOM PUS`,

    `Jeddah  / Bilbao

20 000  mt HRC  max 28,5 mt

FIOS

4000 mt fhinc / CQD disch

25 June - 5 July try later

3,75% here`,

    `Please provide suitable, rated vessels for our following firm requirements.
* A/C SeaSchiffe
* 1 TCT with Steels/Gens/lawfuls
* 33k dwt upto HMAX
* Delivery: ECI
* Laycan: 21-23 July
* Redel: ARAG via COGH transit
* Duration: abt 50-55 days wog
* 3.75% Adc`,

    `DEAR SIRS

GOOD DAY

OUR DIRECT OWS OPEN AS FOLLOWS

PLS PPSE SUIT

PACIFIC
=======

MV SARONIC CHAMPION (93K - SCRUBBER FITTED / 2011) - OPEN VUNG ANG, VIETNAM 08-12 JUNE

SARONIC CHAMPION
LIBERIA FLAG
BUILT 2011
CLASS LR
ABT 93116 DWT ON ABT 14.90 MTRS SSW
LOA 229.253 MTRS / BEAM 38.00 MTRS
GRAIN CAP ABT 110330 CBM
7/7 HO/HA - SCRUBBER FITTED`,
  ];

  let sampleIndex = 0;

  const CATEGORY_META = {
    tonnage:   { 
      icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="3"/><line x1="12" y1="22" x2="12" y2="8"/><path d="M5 12H2a10 10 0 0 0 20 0h-3"/></svg>`, 
      label: "Tonnage", 
      cssClass: "category-badge--tonnage" 
    },
    cargo_vc:  { 
      icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>`, 
      label: "Cargo · Voyage Charter",  
      cssClass: "category-badge--cargo_vc"  
    },
    cargo_tc:  { 
      icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>`, 
      label: "Cargo · Time Charter",    
      cssClass: "category-badge--cargo_tc"  
    },
  };

  const FIELD_LABELS = {
    vessel_name:     "Vessel Name",
    account_name:    "Account",
    open_port:       "Open Port",
    open_date:       "Open Date",
    vessel_type:     "Vessel Type",
    vessel_size:     "Vessel Size (DWT)",
    cargo_name:      "Cargo",
    loading_port:    "Load Port",
    discharge_port:  "Discharge Port",
    delivery_port:   "Delivery Port",
    redelivery_port: "Redelivery Port",
    laycan:          "Laycan",
    duration:        "Duration",
    cargo_type:      "Cargo Type",
  };

  /**
   * Sets the visual loading overlay state and parsing button usability.
   * @param {boolean} on - The active state of the loading overlay.
   * @param {string} [msg] - Optional customized message to display inside overlay.
   */
  function setLoading(on, msg = "Classifying email…") {
    $loadingOverlay.hidden = !on;
    $loadingText.textContent = msg;
    $btnParse.disabled = on;
    if (on) $resultsSection.hidden = true;
  }

  /**
   * Updates the UI header status dot based on backend availability check.
   * @param {string} state - The status state string: "online", "offline", or "connecting".
   */
  function setStatus(state) {
    $statusDot.className = "status-dot";
    if (state === "online") {
      $statusDot.classList.add("status-dot--online");
      $statusText.textContent = "Backend online";
    } else if (state === "offline") {
      $statusDot.classList.add("status-dot--offline");
      $statusText.textContent = "Backend offline";
    } else {
      $statusDot.classList.add("status-dot--connecting");
      $statusText.textContent = "Connecting…";
    }
  }

  /**
   * Performs an asynchronous health check validation fetch against the API.
   * @returns {Promise<void>}
   */
  async function warmBackend() {
    setStatus("connecting");
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(30000) });
      if (res.ok) {
        setStatus("online");
      } else {
        setStatus("offline");
      }
    } catch {
      setStatus("offline");
    }
  }

  /**
   * Main orchestrator to transmit email payload to parse API and direct output render.
   * @returns {Promise<void>}
   */
  async function parseEmail() {
    const body = $emailInput.value.trim();
    if (!body) {
      $emailInput.focus();
      return;
    }

    setLoading(true, "Classifying email…");

    const msgs = [
      "Running rule engine…",
      "Evaluating ML model…",
      "Extracting records…",
      "Formatting response…",
    ];
    let msgIdx = 0;
    const msgTimer = setInterval(() => {
      msgIdx = Math.min(msgIdx + 1, msgs.length - 1);
      $loadingText.textContent = msgs[msgIdx];
    }, 800);

    try {
      const res = await fetch(`${API_BASE}/parse-email`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email_body: body }),
        signal: AbortSignal.timeout(60000),
      });

      clearInterval(msgTimer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      renderResults(data);

      // Save records to Cloudflare D1
      if (data.success && data.records && data.records.length > 0) {
        fetch("/api/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ category: data.category, records: data.records })
        }).catch(err => console.error("Failed to save to D1:", err));
      }
    } catch (err) {
      clearInterval(msgTimer);
      setLoading(false);
      alert(`Error: ${err.message}`);
    }
  }

  /**
   * Renders the parsed classification details, metadata metrics, and payload results.
   * @param {Object} data - The deserialized API response body.
   */
  function renderResults(data) {
    setLoading(false);
    $resultsSection.hidden = false;

    const meta = CATEGORY_META[data.category] || CATEGORY_META.tonnage;
    $categoryBadge.className = `category-badge ${meta.cssClass}`;
    $categoryIcon.innerHTML = meta.icon;
    $categoryLabel.textContent = meta.label;

    const pct = Math.round(data.confidence * 100);
    $confidenceFill.style.width = `${pct}%`;
    $confidenceVal.textContent = `${pct}%`;

    $metaRecords.textContent = data.metadata?.records_found ?? data.records?.length ?? 0;
    $metaTime.textContent = data.metadata?.processing_time_ms?.toFixed(1) ?? "–";

    renderRecords(data.records, data.category);

    $jsonOutput.textContent = JSON.stringify(data, null, 2);

    $resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /**
   * Iterates and generates structural table grids for every extracted data record.
   * @param {Array<Object>} records - Extracted records array.
   * @param {string} category - Category class indicator.
   */
  function renderRecords(records, category) {
    if (!records || records.length === 0) {
      $recordsBody.innerHTML = `<p style="color: var(--clr-text-dim); padding: var(--space-lg); text-align:center;">No records extracted.</p>`;
      return;
    }

    $recordsBody.innerHTML = "";

    records.forEach((rec, i) => {
      const block = document.createElement("div");
      block.className = "record-block";
      block.style.animationDelay = `${i * 80}ms`;

      const label = document.createElement("div");
      label.className = "record-block__index";
      label.textContent = `Record ${i + 1}`;
      block.appendChild(label);

      const grid = document.createElement("div");
      grid.className = "record-grid";

      const fields = Object.keys(rec);

      fields.forEach((key) => {
        const val = rec[key];
        const field = document.createElement("div");
        field.className = "record-field";

        const keyEl = document.createElement("span");
        keyEl.className = "record-field__key";
        keyEl.textContent = FIELD_LABELS[key] || key.replace(/_/g, " ");

        const valEl = document.createElement("span");
        valEl.className = "record-field__val";
        if (val) {
          valEl.textContent = val;
        } else {
          valEl.textContent = "—";
          valEl.classList.add("record-field__val--empty");
        }

        field.appendChild(keyEl);
        field.appendChild(valEl);
        grid.appendChild(field);
      });

      block.appendChild(grid);
      $recordsBody.appendChild(block);
    });
  }

  $emailInput.addEventListener("input", () => {
    $charCount.textContent = `${$emailInput.value.length} characters`;
  });

  $btnParse.addEventListener("click", parseEmail);

  $emailInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      parseEmail();
    }
  });

  $btnPasteSample.addEventListener("click", () => {
    $emailInput.value = SAMPLES[sampleIndex % SAMPLES.length];
    sampleIndex++;
    $charCount.textContent = `${$emailInput.value.length} characters`;
    $emailInput.focus();
  });

  $btnClear.addEventListener("click", () => {
    $emailInput.value = "";
    $charCount.textContent = "0 characters";
    $resultsSection.hidden = true;
    $emailInput.focus();
  });

  $btnDownloadJson?.addEventListener("click", () => {
    const jsonText = $jsonOutput.textContent;
    if (!jsonText) return;

    const blob = new Blob([jsonText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mailparser-result.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  warmBackend();
})();
