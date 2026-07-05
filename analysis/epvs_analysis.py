"""
ePVS group-difference analysis
UCSF Research Data Analyst assessment - Srirama Murthy Chellu

Primary analysis uses whole-white-matter measurements (one row per subject).
Regional analyses are secondary and exploratory.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "pvs_epc_with_reports_filtered_for_interview.csv"
TABLES = PROJECT_DIR / "outputs" / "tables"
FIGURES = PROJECT_DIR / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

GROUPS = ["A", "B", "C", "D"]
sns.set_theme(style="whitegrid")


# Load data
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"Age:": "age", "Handedness:": "handedness",
                        "Sex:": "sex", "new_group": "group"})
df["group"] = pd.Categorical(df["group"], categories=GROUPS)


# Basic QC
n_zero = int((df["pvs_count"] == 0).sum())
piv = df.pivot_table(index="sub_id", columns="mask", values="pvs_count")
qc = {
    "Rows": len(df),
    "Columns": df.shape[1],
    "Unique subjects": df["sub_id"].nunique(),
    "Unique masks": df["mask"].nunique(),
    "Missing values": int(df.isna().sum().sum()),
    "Duplicate subject-mask rows": int(df.duplicated(["sub_id", "mask"]).sum()),
    "Negative numeric values": int(
        (df[["mask_volume", "pvs_count", "pvs_total_volume", "pvs_diameter_mean"]] < 0).sum().sum()),
    "Modality (constant, not used as covariate)": ", ".join(df["modality"].unique()),
    "Age range (years)": f"{df['age'].min():.2f} to {df['age'].max():.2f}",
    "All subjects have 13 masks": bool((df.groupby("sub_id")["mask"].nunique() == 13).all()),
    "Demographics constant per subject": bool(
        (df.groupby("sub_id")[["age", "sex", "handedness", "group"]].nunique() == 1).all().all()),
    "Whole-brain count = Left + Right": bool(
        np.allclose(piv["wmmask"], piv["L-wmmask"] + piv["R-wmmask"])),
    "Zero-ePVS rows (no ePVS detected)": f"{n_zero} ({100*n_zero/len(df):.1f}%)",
    "Subjects per group": "; ".join(
        f"{g}={int((df.drop_duplicates('sub_id')['group'] == g).sum())}" for g in GROUPS),
}
pd.DataFrame(qc.items(), columns=["Check", "Result"]).to_csv(TABLES / "qc_summary.csv", index=False)
df = df.drop(columns=["modality"])
print("QC done:", qc["Rows"], "rows,", qc["Unique subjects"], "subjects")


# Size-normalized outcomes
df["density"] = df["pvs_count"] / df["mask_volume"] * 1e4
df["vol_fraction"] = df["pvs_total_volume"] / df["mask_volume"] * 100
df["diameter"] = df["pvs_diameter_mean"]
# 0 = none detected
df["diameter_nozeros"] = df["pvs_diameter_mean"].where(df["pvs_count"] > 0, np.nan)


# Whole WM only
whole = df[df["mask"] == "wmmask"].copy()
whole["log_density"] = np.log(whole["density"])
whole["log_vol_fraction"] = np.log(whole["vol_fraction"])


# Summary tables
# Table 1: demographics
demo = []
for g in GROUPS:
    s = whole[whole.group == g]
    demo.append({
        "Group": g, "N": len(s),
        "Age, mean (SD)": f"{s.age.mean():.1f} ({s.age.std():.1f})",
        "Female, n (%)": f"{(s.sex == 'Female').sum()} ({100*(s.sex == 'Female').mean():.0f}%)",
        "Right-handed, n (%)": f"{(s.handedness == 'Right').sum()} ({100*(s.handedness == 'Right').mean():.0f}%)",
        "WM volume, mean (SD)": f"{s.mask_volume.mean():.0f} ({s.mask_volume.std():.0f})",
    })
pd.DataFrame(demo).to_csv(TABLES / "table1_demographics.csv", index=False)

# Covariate balance check
age_p = stats.f_oneway(*[whole[whole.group == g].age for g in GROUPS]).pvalue
sex_p = stats.chi2_contingency(pd.crosstab(whole.group, whole.sex))[1]
hand_p = stats.chi2_contingency(pd.crosstab(whole.group, whole.handedness))[1]

# Table 2: imaging
img = []
for col, lab in [("pvs_count", "ePVS count"), ("density", "ePVS density (/10k vox)"),
                 ("vol_fraction", "ePVS volume fraction (%)"), ("diameter", "ePVS mean diameter (vox)")]:
    row = {"Outcome": lab}
    for g in GROUPS:
        s = whole[whole.group == g][col]
        row[g] = f"{s.mean():.2f} ({s.std():.2f})"
    img.append(row)
pd.DataFrame(img).to_csv(TABLES / "table2_imaging_outcomes_by_group.csv", index=False)


# Skew check
dist = []
for col in ["pvs_count", "density", "vol_fraction", "diameter"]:
    x = whole[col].dropna()
    skew = stats.skew(x)
    p_raw = stats.shapiro(x)[1]
    p_log = stats.shapiro(np.log(x[x > 0]))[1]
    decision = "log-transform" if (abs(skew) > 0.5 and p_log > p_raw) else "keep raw"
    dist.append({"outcome": col, "skewness": round(skew, 2),
                 "shapiro_p_raw": round(p_raw, 4), "shapiro_p_log": round(p_log, 4),
                 "decision": decision})
dist_df = pd.DataFrame(dist)
dist_df.to_csv(TABLES / "table3_distribution_checks.csv", index=False)

# Count vs size
r_count = stats.pearsonr(whole.mask_volume, whole.pvs_count)[0]
r_density = stats.pearsonr(whole.mask_volume, whole.density)[0]
age_density_r = stats.pearsonr(whole.age, whole.density)[0]
print(f"mask volume vs raw count r={r_count:.2f}, vs density r={r_density:.2f}")


# Primary ANCOVA models
ancova = []
for col, lab in [("log_density", "ePVS density"), ("log_vol_fraction", "ePVS volume fraction"),
                 ("diameter_nozeros", "ePVS diameter")]:
    d = whole.dropna(subset=[col])
    groups_data = [d[d.group == g][col] for g in GROUPS]

    # Unadjusted, for reference
    anova_p = stats.f_oneway(*groups_data).pvalue
    kruskal_p = stats.kruskal(*groups_data).pvalue

    # Adjusted model
    model = smf.ols(f"{col} ~ C(group) + age + C(sex) + C(handedness)", data=d).fit()
    table = sm.stats.anova_lm(model, typ=2)
    ss_group = table.loc["C(group)", "sum_sq"]
    ss_resid = table.loc["Residual", "sum_sq"]
    partial_eta2 = ss_group / (ss_group + ss_resid)

    ancova.append({
        "outcome": lab, "col": col,
        "oneway_anova_p": anova_p, "kruskal_p": kruskal_p,
        "ancova_group_F": table.loc["C(group)", "F"],
        "ancova_group_p": table.loc["C(group)", "PR(>F)"],
        "partial_eta2": partial_eta2,
        "resid_shapiro_p": stats.shapiro(model.resid)[1],
        "levene_p": stats.levene(*groups_data).pvalue,
        "age_p": table.loc["age", "PR(>F)"],
        "sex_p": table.loc["C(sex)", "PR(>F)"],
        "handedness_p": table.loc["C(handedness)", "PR(>F)"],
    })
ancova = pd.DataFrame(ancova)


# FDR correction
ancova["ancova_p_fdr"] = multipletests(ancova["ancova_group_p"], method="fdr_bh")[1]
ancova.to_csv(TABLES / "table4_global_ancova_results.csv", index=False)
print("ANCOVA (FDR-adjusted):", dict(zip(ancova.outcome, ancova.ancova_p_fdr.round(3))))

# Post-hoc if significant
posthoc = []
for _, r in ancova.iterrows():
    if r["ancova_p_fdr"] >= 0.10:
        continue
    col = r["col"]
    d = whole.dropna(subset=[col])
    tukey = pairwise_tukeyhsd(d[col], d["group"])
    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    for _, pair in tukey_df.iterrows():
        a = d[d.group == pair["group1"]][col]
        b = d[d.group == pair["group2"]][col]
        pooled_sd = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
        posthoc.append({
            "outcome": r["outcome"], "group1": pair["group1"], "group2": pair["group2"],
            "mean_diff": pair["meandiff"], "p_adj": pair["p-adj"], "reject_H0": pair["reject"],
            "cohens_d": round((a.mean() - b.mean()) / pooled_sd, 2),
        })
posthoc = pd.DataFrame(posthoc) if posthoc else pd.DataFrame(
    [{"note": "No outcome met the threshold for post-hoc testing."}])
posthoc.to_csv(TABLES / "table5_posthoc_if_applicable.csv", index=False)


# Regional (exploratory)
# Combine L+R regions
lobar = df[df["mask"].str.match(r"^[LR]-(front|par|occ|temp|ins)$")].copy()
lobar["region"] = lobar["mask"].str.split("-").str[1].map(
    {"front": "Frontal", "par": "Parietal", "occ": "Occipital", "temp": "Temporal", "ins": "Insula"})
bilateral = (lobar.groupby(["sub_id", "region", "group"], observed=True)
                  .agg(pvs_count=("pvs_count", "sum"), mask_volume=("mask_volume", "sum"),
                       age=("age", "first"), sex=("sex", "first"),
                       handedness=("handedness", "first")).reset_index())
bilateral["density"] = bilateral["pvs_count"] / bilateral["mask_volume"] * 1e4
bilateral["log_density"] = np.log(bilateral["density"])

regional = []
for region in ["Frontal", "Parietal", "Temporal", "Occipital", "Insula"]:
    rd = bilateral[bilateral.region == region]
    model = smf.ols("log_density ~ C(group) + age + C(sex) + C(handedness)", data=rd).fit()
    table = sm.stats.anova_lm(model, typ=2)
    ss_group = table.loc["C(group)", "sum_sq"]
    ss_resid = table.loc["Residual", "sum_sq"]
    regional.append({
        "region": region, "ancova_group_F": table.loc["C(group)", "F"],
        "ancova_group_p": table.loc["C(group)", "PR(>F)"],
        "partial_eta2": ss_group / (ss_group + ss_resid),
    })
regional = pd.DataFrame(regional)
regional["p_fdr"] = multipletests(regional["ancova_group_p"], method="fdr_bh")[1]
regional.to_csv(TABLES / "table6_regional_exploratory_results.csv", index=False)
print("Regional (frontal FDR p):", round(regional.loc[regional.region == "Frontal", "p_fdr"].iloc[0], 3))


# Figures
colors = dict(zip(GROUPS, sns.color_palette("Set2", 4)))

# Figure 1: distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
for ax, (col, lab) in zip(axes, [("pvs_count", "ePVS count"),
                                 ("vol_fraction", "ePVS volume fraction (%)"),
                                 ("diameter", "ePVS diameter (vox)")]):
    x = whole[col].dropna()
    sns.histplot(x, kde=True, ax=ax, color="#4C9AA0", edgecolor="white")
    ax.axvline(x.mean(), color="firebrick", ls="--", lw=2, label=f"mean {x.mean():.2f}")
    ax.set_title(f"{lab} (skew {stats.skew(x):.2f})", fontsize=12)
    ax.set_xlabel(lab)
    ax.legend(fontsize=9)
fig.suptitle("Figure 1. Distributions of ePVS outcomes (n = 124)", fontsize=15)
fig.savefig(FIGURES / "fig1_distributions.png", dpi=200)
plt.close(fig)

# Figure 2: by group
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
for ax, (col, lab) in zip(axes, [("density", "ePVS density (/10k vox)"),
                                 ("vol_fraction", "ePVS volume fraction (%)"),
                                 ("diameter", "ePVS diameter (vox)")]):
    sns.boxplot(data=whole, x="group", y=col, order=GROUPS, hue="group",
                palette=colors, ax=ax, width=.6, fliersize=0, legend=False)
    sns.stripplot(data=whole, x="group", y=col, order=GROUPS, color="0.3", size=3, alpha=.5, ax=ax)
    ax.set(xlabel="Group", ylabel=lab, title=lab)
fig.suptitle("Figure 2. ePVS outcomes by study group", fontsize=15)
fig.savefig(FIGURES / "fig2_boxplots_by_group.png", dpi=200)
plt.close(fig)

# Figure 3: mask size
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
for g in GROUPS:
    s = whole[whole.group == g]
    axes[0].scatter(s.mask_volume/1e3, s.pvs_count, color=colors[g], s=40, alpha=.8, label=f"Group {g}")
    axes[1].scatter(s.mask_volume/1e3, s.density, color=colors[g], s=40, alpha=.8)
axes[0].set(xlabel="WM mask volume (x1000 vox)", ylabel="Raw ePVS count",
            title=f"A. Raw count vs mask size (r = {r_count:.2f})")
axes[1].set(xlabel="WM mask volume (x1000 vox)", ylabel="ePVS density (/10k vox)",
            title=f"B. Density vs mask size (r = {r_density:.2f})")
axes[0].legend(fontsize=9)
fig.suptitle("Figure 3. Why mask size must be accounted for", fontsize=15)
fig.savefig(FIGURES / "fig3_masksize_scatter.png", dpi=200)
plt.close(fig)

# Figure 4: regional
fig, ax = plt.subplots(figsize=(9.5, 5))
sns.barplot(data=bilateral, x="region", y="density", hue="group",
            order=["Frontal", "Parietal", "Temporal", "Occipital", "Insula"],
            hue_order=GROUPS, palette=colors, ax=ax, errorbar="se")
ax.set(xlabel="Brain region (bilateral)", ylabel="ePVS density (/10k vox)",
       title="Figure 4. ePVS density by region and group (exploratory)")
ax.legend(title="Group", fontsize=9, ncol=4)
fig.savefig(FIGURES / "fig4_regional_density.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure 5: age
fig, ax = plt.subplots(figsize=(8, 5))
for g in GROUPS:
    s = whole[whole.group == g]
    ax.scatter(s.age, s.density, color=colors[g], s=40, alpha=.8, label=f"Group {g}")
slope, intercept = np.polyfit(whole.age, whole.density, 1)
xs = np.linspace(whole.age.min(), whole.age.max(), 50)
ax.plot(xs, slope*xs + intercept, color="0.3", ls="--", lw=2, label="trend")
ax.set(xlabel="Age (years)", ylabel="ePVS density (/10k vox)",
       title=f"Figure 5. ePVS density vs age (r = {age_density_r:.2f})")
ax.legend(fontsize=9)
fig.savefig(FIGURES / "fig5_age_vs_density.png", dpi=200)
plt.close(fig)


# Save results
results = {
    "qc": {k: str(v) for k, v in qc.items()},
    "baseline_balance": {"age_p": float(age_p), "sex_p": float(sex_p), "handedness_p": float(hand_p)},
    "mask_size": {"corr_volume_rawcount": float(r_count), "corr_volume_density": float(r_density)},
    "age_density_r": float(age_density_r),
    "distribution_checks": dist_df.to_dict("records"),
    "global_ancova": ancova.to_dict("records"),
    "posthoc": posthoc.to_dict("records"),
    "regional_exploratory": regional.to_dict("records"),
}
with open(PROJECT_DIR / "outputs" / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Done. Tables and figures saved to outputs/.")
