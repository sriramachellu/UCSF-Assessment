# ePVS Group-Difference Analysis

UCSF Research Data Analyst assessment — Srirama Murthy Chellu

This project tests whether enlarged perivascular space (ePVS) measurements
differ across four study groups (A–D) in a sample of 124 children/adolescents,
after accounting for region size, age, sex, and handedness.

## Folder layout

```
epvs-assessment/
├── data/
│   └── pvs_epc_with_reports_filtered_for_interview.csv
├── analysis/
│   └── epvs_analysis.py        # the analysis (QC → outcomes → ANCOVA → regional → figures)
├── outputs/
│   ├── tables/                 # qc_summary + table1–table6 (CSV)
│   ├── figures/                # fig1–fig5 (PNG)
│   └── results.json            # key numbers from the analysis
├── README.md
├── requirements.txt
└── ePVS_Analysis_Report.pdf
```

## How to run

```bash
pip install -r requirements.txt
python analysis/epvs_analysis.py     # writes tables and figures to outputs/
```

The script reads the CSV from `data/` using a relative path and a fixed random
seed, so it reproduces the same tables and figures each time.

## Method summary

- **Primary analysis:** whole white-matter mask only, so there is one
  independent observation per subject.
- **Outcomes:** ePVS density (count / mask volume × 10,000), volume fraction
  (ePVS volume / mask volume × 100), and mean diameter. Density and volume
  fraction were log-transformed because they were right-skewed.
- **Region size:** ePVS density and volume fraction are already normalized by
  mask volume; Figure 3 shows why this matters.
- **Models:** ANCOVA `outcome ~ group + age + sex + handedness` (Type II,
  partial η²), with one-way ANOVA and Kruskal–Wallis as unadjusted references,
  and Shapiro–Wilk / Levene assumption checks.
- **Multiple comparisons:** Benjamini–Hochberg FDR across the three outcomes.
- **Post-hoc:** Tukey HSD and Cohen's d for any significant outcome.
- **Regional analysis:** left and right masks combined into five bilateral
  regions; density ANCOVA per region, FDR-corrected. Secondary and exploratory.

## Main result

ePVS density differed across groups (ANCOVA p = 0.016, FDR p = 0.049, partial
η² = 0.084), driven by Group B (highest density; Cohen's d ≈ 1.2 vs Group A) and
strongest in the frontal region (exploratory FDR p = 0.046). Volume fraction and
diameter did not differ. Density increased with age (r = 0.44). Group B is small
(n = 11), so its result is interpreted cautiously.
