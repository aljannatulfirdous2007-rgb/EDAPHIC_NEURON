# 🌱 Edaphic Neuron
### AI-Guided Field Force Intelligence — Syngenta IITM Hackathon 2026

> *"The rep still knocks on every door. They just never walk in blind again."*

---

## Project Overview

**Edaphic Neuron** is a field intelligence operating system for Syngenta sales representatives. It connects biological urgency — pest outbreaks, crop stress, satellite signals — directly to a rep's daily visit plan, automatically, before the farmer even knows the threat is there.

**Track:** Track 2 — AI-Guided Field Force Intelligence  
**Team Name:** Edaphic Neuron  
**Institution:** IIT Madras BS Programme  
**Season:** Rabi 2025-26 | Territory: TER_0164 — Amritsar West, Punjab

---

## The Problem

Syngenta field reps visit retailers on fixed rotations — same order, every week. Meanwhile:
- NPSS has pest outbreak data that nobody connects to rep routes
- Farmonaut satellites show crop stress days before damage appears
- Retailer stockouts create competitor gaps that expire in 48 hours
- 9,348 stockout events happened this Rabi season — most were missed

**Edaphic Neuron fixes this.** Every morning at 6am, it has already scanned pest data, weather signals, satellite imagery, and retailer inventory — and rebuilt the rep's day around what actually matters.

---

## Live Demo

**🌐 Live Deployment:** `[DEPLOYMENT_LINK]`  
*(Open in Chrome for full functionality including Voice Note feature)*

**📁 Source Code:** `[GITHUB_LINK]`

---

## Team

| Name | Role | Email |
|------|------|-------|
| A. Al Jannatul Firdous | ML Lead — M1/M2/M3/M7 Models | `[EMAIL]` |
| Oviya Sankarasivakumar | Backend — FastAPI + Claude API | `[EMAIL]` |
| Rakshna R | Frontend — React Dashboard + PWA | `[EMAIL]` |
| Kavya T | ML Partner — M4/M5/M6 Models + Data Analysis | `[EMAIL]` |

---

## Architecture

```
Signal Layer           Intelligence Engine        Action Layer
─────────────          ──────────────────         ────────────
NPSS Pest Alerts  →    M1: Risk Scorer        →   Ranked Visit List
IMD Weather       →    M2: Urgency Ranker     →   Action Card (Marathi/Hindi)
Farmonaut NDVI    →    M3: Route Optimizer    →   SHAP Explanation
AgriStack Farmers →    M4: Pest Predictor     →   Manager Dashboard
Retailer POS      →    M5: Anomaly Detector   →   Offline PWA
CRM Visit Logs    →    M6: Bandit Learner     →   Board Report
                        M7: SHAP Explainer    →   Voice-to-Action
```

---

## Key Results (Real Syngenta Dataset)

| Metric | Value |
|--------|-------|
| M2 Urgency Ranker AUC | **0.905** (30,000 real visits) |
| Product Rec → Sale Conversion | **85.7%** same-month rate |
| WhatsApp Open Rate | **23.2%** (4,479 messages) |
| Stockout Events Detected | **9,348** across Rabi season |
| High-Risk Retailers | **316** (7.9%) needing urgent visits |
| Smartphone Scan Rate | **25.8%** vs 0% keypad |

---

## 7 Unfair Advantage Features

1. **Human-in-the-Loop Override** — AI Sort / Manual Sort toggle; override reason fed back to M6 Bandit
2. **Inaction Cost Calculator** — Revenue at risk formula: Risk% × ₹41,600 avg order = ₹ at stake
3. **Model Confidence Tags** — High/Med/Low confidence per retailer based on data recency
4. **Voice-to-Action** — Web Speech API; rep speaks outcome note → Claude parses to Next Task
5. **Flash-Sync Animation** — Packet stream visualization when offline data syncs on reconnect
6. **Executive Board Report** — Claude generates 3-paragraph CEO-level territory summary
7. **Global Scalability Simulator** — Switch territory: Punjab (Wheat) → Maharashtra (Cotton) → Brazil (Soy) → Iowa (Corn)

---

## Running the Demo

### Option 1: Open the HTML file directly
```
Open EdaphicNeuron_ULTIMATE.html in Chrome
No server required — fully self-contained
```

### Option 2: Deploy locally
```bash
# No dependencies for the frontend demo
# Just open in Chrome — Leaflet loads from CDN
open EdaphicNeuron_ULTIMATE.html
```

### Option 3: Python ML pipeline
```bash
pip install pandas numpy scikit-learn joblib
python EdaphicNeuron_ML_Models.py
# Expected output: All 7 models passing, AUC scores printed
```

---

## Files in This Submission

```
edaphic-neuron/
├── README.md                          ← This file
├── EdaphicNeuron_ULTIMATE.html        ← Complete live demo (open in Chrome)
├── EdaphicNeuron_ML_Models.py         ← All 7 ML models, Python
├── EdaphicNeuron_Submission_Report.html ← 10-page solution document
├── EdaphicNeuron_Slides.pdf           ← Presentation slides
├── demo_video/
│   └── edaphic_neuron_demo.mp4        ← Demo video with all team members
└── docs/
    ├── API_Requirements.md
    └── Architecture_Diagram.png
```

---

## API & Environment Requirements

| API / Library | Purpose | Cost |
|---|---|---|
| Anthropic Claude API | Action card generation (claude-sonnet-4-20250514) | Pay-per-use |
| Leaflet.js v1.9.4 | Real geographic map | Free / CDN |
| OpenStreetMap | Map tiles | Free |
| Open-Meteo API | Weather data (IMD proxy) | Free |
| Web Speech API | Voice-to-action (browser native) | Free |
| scikit-learn, pandas, numpy | ML models | Free / Open Source |
| localStorage | Bandit weight persistence | Browser native |

**No backend server required for the demo.** The HTML demo is fully self-contained and works offline after initial load.

---

## Data Sources Used

| Source | Data | Access |
|---|---|---|
| Syngenta Synthetic Dataset | 30,000 visits · 235,042 POS · 4,000 retailers · 6,000 growers | Provided |
| NPSS (npss.icar.gov.in) | Pest surveillance — 432 pests, 66 crops | Public |
| IMD / Open-Meteo | District weather — temperature, humidity, rainfall | Public |
| Farmonaut | Satellite NDVI — crop stress index | Free tier |
| AgriStack | Farmer IDs + crop + location | Public |
| KDSS Crop Calendar | Growth stage by district and crop | Public |

---

## Declared Tools & Libraries

- **scikit-learn** — Random Forest (M2, M4), Isolation Forest (M5), Logistic Regression
- **SHAP** — TreeExplainer for M7 feature attribution
- **Leaflet.js** — Geographic map rendering
- **Anthropic Claude API** — Natural language action card generation
- **Web Speech API** — Browser-native voice transcription
- All code is original. Pre-trained models and open-source libraries declared above.

---

## Contact

**Team Lead:** A. Al Jannatul Firdous  
**Email:** `[EMAIL]`  
**Institution:** IIT Madras BS Programme  
**Submission:** Syngenta AgriTech Hackathon 2026 — Track 2

---

*Edaphic: relating to soil and ground-level biological conditions that shape ecosystems.*  
*Neuron: intelligence that connects signals to action.*  
*Edaphic Neuron reads what the earth is telling us.*
