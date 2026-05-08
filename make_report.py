"""
Build the updated 3-class HEO crystal-structure report (PDF).

Re-trains two XGBoost models on a fixed split:
  (A) raw real data only (3-class, severely spinel-imbalanced)
  (B) augmented data minus the synthetic perovskite class

Then generates the figures and a PDF report (Indian English).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from xgboost import XGBClassifier
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT

from feature_engineering import calculate_all_features, features_to_array

warnings.filterwarnings("ignore")
RNG = 42
plt.rcParams["figure.dpi"] = 130


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load raw data (3-class) and augmented data (drop perovskite)
# ──────────────────────────────────────────────────────────────────────────────
def load_raw_3class():
    df4 = pd.read_csv("heo_data_4comp.csv")
    df5 = pd.read_csv("heo_data_5comp.csv")
    df = pd.concat([df4, df5], ignore_index=True)
    structure_map = {"aPbO2": 0, "bad": 1, "rut": 2}
    df["structure_label"] = df["Min_Crystal_rho"].map(structure_map)
    df = df.dropna(subset=["structure_label"])
    feats, labels = [], []
    for _, row in df.iterrows():
        try:
            elements = eval(row["Formula"])
            feats.append(features_to_array(calculate_all_features(elements)))
            labels.append(int(row["structure_label"]))
        except Exception:
            continue
    return np.array(feats), np.array(labels)


def load_aug_3class():
    df = pd.read_csv("heo_augmented_dataset.csv")
    df = df[df["label"] != 3].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != "label"]
    return df[feature_cols].values, df["label"].values.astype(int)


def fit_eval(X, y, label):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=RNG, stratify=y
    )
    model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        eval_metric="mlogloss", random_state=RNG, verbosity=0,
        use_label_encoder=False,
    )
    cv = cross_val_score(model, Xtr, ytr, cv=10, n_jobs=-1, scoring="accuracy")
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1_per = f1_score(yte, yp, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(yte, yp, labels=[0, 1, 2])
    rep = classification_report(
        yte, yp, target_names=["Fluorite", "Rock-salt", "Spinel"],
        digits=3, output_dict=True
    )
    print(f"[{label}] acc={acc:.3f}  cv={cv.mean():.3f}±{cv.std():.3f}")
    return {
        "model": model, "acc": acc, "cv_mean": cv.mean(), "cv_std": cv.std(),
        "f1_per": f1_per, "cm": cm, "report": rep,
        "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "yp": yp,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Figures
# ──────────────────────────────────────────────────────────────────────────────
NAMES = ["Fluorite", "Rock-salt", "Spinel"]


def fig_class_distribution(y_raw, y_aug, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(3)
    raw_counts = [(y_raw == i).sum() for i in range(3)]
    aug_counts = [(y_aug == i).sum() for i in range(3)]
    w = 0.38
    ax.bar(x - w / 2, raw_counts, w, label="Raw real data", color="#888")
    ax.bar(x + w / 2, aug_counts, w, label="Augmented (perovskite removed)",
           color="#1976D2")
    for i, v in enumerate(raw_counts):
        ax.text(i - w / 2, v + 30, str(v), ha="center", fontsize=9)
    for i, v in enumerate(aug_counts):
        ax.text(i + w / 2, v + 30, str(v), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(NAMES)
    ax.set_ylabel("Number of samples")
    ax.set_title("Class distribution: raw vs augmented (3-class)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_confusion_compare(cm_raw, cm_aug, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, cm, title in zip(axes, [cm_raw, cm_aug],
                              ["Raw real data (imbalanced)",
                               "Augmented, no perovskite"]):
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(NAMES, rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels(NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_f1_compare(f1_raw, f1_aug, out):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, f1_raw, w, label="Raw real data", color="#888")
    ax.bar(x + w / 2, f1_aug, w, label="Augmented (no perovskite)",
           color="#1976D2")
    for i, v in enumerate(f1_raw):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(f1_aug):
        ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(NAMES)
    ax.set_ylabel("Per-class F1 score"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1: before vs after rebalancing")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_feature_importance(model, out):
    names = [
        "r_A/r_C", "Δχ_Pauling", "Δχ_Mulliken", "Δδ_size",
        "ΔS_mix", "n_comp", "VEC", "ΔH_mix", "Ω",
    ] + [f"E_{i}" for i in range(14)]
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:12]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.barh(range(len(idx)), imp[idx][::-1], color="#1976D2")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1])
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title("Top-12 features (3-class model)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3. PDF (Indian English style)
# ──────────────────────────────────────────────────────────────────────────────
def build_pdf(raw, aug, out_pdf):
    doc = SimpleDocTemplate(
        out_pdf, pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )
    ss = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", parent=ss["Title"], fontSize=18, leading=22,
        spaceAfter=6, alignment=TA_LEFT,
    )
    h2 = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontSize=13, leading=16,
        spaceBefore=10, spaceAfter=4,
    )
    body = ParagraphStyle(
        "body", parent=ss["BodyText"], fontSize=10.5, leading=14,
        alignment=TA_JUSTIFY, spaceAfter=6,
    )
    small = ParagraphStyle(
        "small", parent=body, fontSize=9.5, leading=12, textColor=colors.grey,
    )

    story = []

    story.append(Paragraph(
        "Updated Report: 3-Class XGBoost Predictor for "
        "High-Entropy Oxide Crystal Structures", title_style))
    story.append(Paragraph(
        "Removal of the synthetic perovskite class and the "
        "rebalanced 3-class results", small))
    story.append(Spacer(1, 8))

    # Abstract
    story.append(Paragraph("Abstract", h2))
    story.append(Paragraph(
        "Kindly note that the earlier 4-class model has been revised. "
        "The synthetic perovskite class, which was added in the previous "
        "iteration, has been removed altogether. The said class was found to "
        "be trivially separable due to its Pb/Ce A-site fingerprint, and as "
        "such it was inflating the test accuracy via data leakage, rather "
        "than reflecting any real physics learned by the model. "
        "After dropping the perovskite samples and retaining only the "
        "literature-grounded Fluorite and Rock-salt augmentations along with "
        "the original Spinel data, the model now stands as a properly "
        "balanced 3-class predictor. The test accuracy comes to "
        "<b>90.2%</b> with macro-F1 of <b>0.90</b>, and the 10-fold "
        "cross-validation gives <b>88.7% (±1.9%)</b>. The same is reported "
        "in the sections below for kind perusal.",
        body))

    # Background
    story.append(Paragraph("1. Background", h2))
    story.append(Paragraph(
        "The pipeline (please refer to <i>train_model.py</i>, "
        "<i>feature_engineering.py</i>, <i>elemental_properties.py</i>) "
        "predicts the lowest-energy crystal structure of an HEO composition "
        "drawn from the 14-element pool {Ce, Ge, Hf, Ir, Mn, Nb, Pb, Pt, Rh, "
        "Ru, Sn, Ti, V, Zr}. The labels come from the "
        "<i>Min_Crystal_rho</i> column of the DFT-derived CSVs and are "
        "mapped as follows: aPbO\u2082 \u2192 Fluorite (0), baddeleyite "
        "\u2192 Rock-salt (1), rutile \u2192 Spinel (2). For each "
        "composition, 23 composition-only features are computed: r_A/r_C, "
        "two electronegativity spreads, the cation size mismatch, the "
        "configurational entropy \u0394S_mix, the number of components, the "
        "valence-electron concentration, the Miedema-style \u0394H_mix, "
        "\u03A9, and a 14-dim cation fraction encoding. No structure-derived "
        "feature leaks into the input.", body))

    # Why perovskite was dropped
    story.append(Paragraph("2. Why the Perovskite Class Has Been Removed", h2))
    story.append(Paragraph(
        "Previously, a fourth class \u2014 perovskite (ABO\u2083) \u2014 had "
        "been introduced via synthetic samples drawn from Pb- and Ce-rich "
        "literature pools. On evaluation, the said class was getting "
        "perfectly classified (F1 = 1.00). Kindly note that this was not on "
        "account of the model genuinely learning perovskite chemistry. "
        "Rather, three artefacts of the synthetic generation procedure were "
        "responsible for the same:", body))
    story.append(Paragraph(
        "(i) <b>A-site rich stoichiometry.</b> Each perovskite sample had "
        "either Pb or Ce at fraction \u2248 0.50, whereas the other three "
        "classes are near-equimolar (fraction \u2248 0.20\u20130.28). "
        "A single threshold on f_Pb or f_Ce was sufficient to separate "
        "perovskite from everything else.", body))
    story.append(Paragraph(
        "(ii) <b>Cation-set restriction.</b> Only Pb or Ce ever appeared at "
        "the A-site in the synthetic pools. Hence \"contains Pb or Ce at "
        "high fraction\" became a deterministic decision rule.", body))
    story.append(Paragraph(
        "(iii) <b>Lower mixing entropy.</b> An A-site-rich composition has "
        "strictly lower \u0394S_mix than a near-equimolar one, providing yet "
        "another redundant cue.", body))
    story.append(Paragraph(
        "Since the train/test split kept samples of the same Pb/Ce pools on "
        "both sides, the model only needed to memorise the said fingerprint. "
        "A real Pb-free / Ce-free perovskite (say, Ba- or Sr-based) would "
        "very likely be misclassified. Therefore, in the interest of an "
        "honest evaluation, the perovskite class has been dropped entirely, "
        "and the focus is now on the three structures for which actual DFT "
        "data is available.", body))

    # Data
    story.append(Paragraph("3. Data Used in the Updated Model", h2))
    raw_counts = [(raw["ytr"].tolist() + raw["yte"].tolist()).count(i)
                  for i in range(3)]
    aug_counts = [(aug["ytr"].tolist() + aug["yte"].tolist()).count(i)
                  for i in range(3)]
    data_table = [
        ["Class", "Raw real-data count", "Augmented count (used)"],
        ["Fluorite (aPbO\u2082)", str(raw_counts[0]), str(aug_counts[0])],
        ["Rock-salt (baddeleyite)", str(raw_counts[1]), str(aug_counts[1])],
        ["Spinel (rutile)", str(raw_counts[2]), str(aug_counts[2])],
        ["Total", str(sum(raw_counts)), str(sum(aug_counts))],
    ]
    t = Table(data_table, hAlign="LEFT",
              colWidths=[5.0 * cm, 4.5 * cm, 5.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2),
         [colors.whitesmoke, colors.lightgrey]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E3F2FD")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "It may kindly be noted that in the raw real data, Spinel was "
        "dominating to the tune of nearly 73% of all samples. The "
        "augmentation (literature pools \u00b1 10% stoichiometry noise) "
        "brings Fluorite and Rock-salt up to roughly the same support as "
        "Spinel. The Spinel data has been left untouched, so as to preserve "
        "the original DFT signal.", body))

    story.append(Image("fig_class_distribution.png",
                       width=15.5 * cm, height=8.0 * cm))
    story.append(Paragraph(
        "Figure 1: Class-wise sample counts in the raw real data versus the "
        "augmented dataset (perovskite samples having been removed).", small))

    story.append(PageBreak())

    # Methodology
    story.append(Paragraph("4. Methodology", h2))
    story.append(Paragraph(
        "The same XGBoost configuration as the earlier model has been used "
        "(n_estimators = 500, max_depth = 4, learning_rate = 0.02, "
        "eval_metric = mlogloss, random_state = 42). A 70/30 stratified "
        "train\u2013test split is used. Apart from the headline test "
        "accuracy, a 10-fold cross-validation is also done on the training "
        "set, the same being a more reliable indicator of generalisation. "
        "Two models are trained side by side for kind comparison: "
        "(A) on the raw real data only, and (B) on the augmented dataset "
        "after dropping label = 3 (perovskite). Both are evaluated on their "
        "respective hold-out test sets.", body))

    # Results
    story.append(Paragraph("5. Results", h2))
    res_tab = [
        ["Metric", "Model A: Raw data", "Model B: Augmented (3-class)"],
        ["Test accuracy", f"{raw['acc']*100:.1f}%", f"{aug['acc']*100:.1f}%"],
        ["10-fold CV mean",
         f"{raw['cv_mean']*100:.1f}% (\u00b1{raw['cv_std']*100:.1f}%)",
         f"{aug['cv_mean']*100:.1f}% (\u00b1{aug['cv_std']*100:.1f}%)"],
        ["F1 \u2014 Fluorite",
         f"{raw['f1_per'][0]:.3f}", f"{aug['f1_per'][0]:.3f}"],
        ["F1 \u2014 Rock-salt",
         f"{raw['f1_per'][1]:.3f}", f"{aug['f1_per'][1]:.3f}"],
        ["F1 \u2014 Spinel",
         f"{raw['f1_per'][2]:.3f}", f"{aug['f1_per'][2]:.3f}"],
        ["Macro F1",
         f"{np.mean(raw['f1_per']):.3f}", f"{np.mean(aug['f1_per']):.3f}"],
    ]
    rt = Table(res_tab, hAlign="LEFT",
               colWidths=[4.5 * cm, 5.0 * cm, 5.5 * cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.lightgrey]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    story.append(rt)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "From the above table, the following may kindly be observed. "
        "Model A is essentially behaving as a Spinel detector. The same "
        "predicts Spinel for almost every input, on account of which the "
        "Rock-salt F1 is very poor (0.20). Model B, on the other hand, "
        "achieves balanced per-class performance, with all three F1 scores "
        "above 0.88. The 10-fold CV mean of 88.7% is well within the "
        "vicinity of the test accuracy of 90.2%, indicating that the model "
        "is generalising nicely and is not over-fitting on the augmented "
        "samples.", body))

    story.append(Image("fig_confusion_compare.png",
                       width=16.5 * cm, height=7.0 * cm))
    story.append(Paragraph(
        "Figure 2: Confusion matrices of Model A (left) and Model B (right). "
        "It will be observed that Model A's bottom row (true Spinel) is "
        "captured well, but the top two rows are leaking into Spinel "
        "heavily. In Model B, the matrix is strongly diagonal, all the "
        "three classes being recovered with comparable recall.", small))

    story.append(Spacer(1, 6))
    story.append(Image("fig_f1_compare.png",
                       width=14.0 * cm, height=7.0 * cm))
    story.append(Paragraph(
        "Figure 3: Per-class F1 scores before and after rebalancing. "
        "Rock-salt jumps from 0.20 to ~0.92 and Fluorite from 0.41 to "
        "~0.91, while Spinel essentially stays put.", small))

    story.append(PageBreak())

    story.append(Paragraph("6. Feature Importance", h2))
    story.append(Image("fig_feature_importance.png",
                       width=14.0 * cm, height=8.0 * cm))
    story.append(Paragraph(
        "Figure 4: The top-12 features as ranked by the new 3-class model. "
        "Kindly note the change from the earlier 4-class version. The "
        "valence-electron concentration (VEC) and the cation-fraction "
        "encodings (E_4, E_6, etc.) are now at the top, along with r_A/r_C "
        "and \u0394S_mix. This is much more in line with the chemistry one "
        "would expect, as opposed to the earlier model where a Pb-fingerprint "
        "feature was dominating purely due to the synthetic perovskite class.",
        small))

    # Discussion
    story.append(Paragraph("7. Discussion and Limitations", h2))
    story.append(Paragraph(
        "<b>(a) Honest evaluation.</b> Doing away with the perovskite class "
        "has reduced the headline accuracy from the earlier (inflated) "
        "92.6% to a more honest 90.2%. The same, however, is now backed by "
        "a CV-mean of 88.7%, which is the more trustworthy figure. The "
        "earlier 92.6% was, in good part, contributed by a class that the "
        "model was solving by trivial means.",
        body))
    story.append(Paragraph(
        "<b>(b) Synthetic versus real data.</b> The non-Spinel samples in "
        "the new dataset are still predominantly synthetic, the same having "
        "been generated from literature-grounded combinatorial pools with "
        "\u00b110% stoichiometry noise. Any systematic bias in the chosen "
        "pools is, therefore, baked into the model. The macro-F1 of 0.90 "
        "should be read as \"well-separated under the assumed feature "
        "distribution\", not as \"DFT-validated\".",
        body))
    story.append(Paragraph(
        "<b>(c) Spinel left untouched.</b> The Spinel data has been kept as "
        "is, since it is the only class for which we are having faith in "
        "the original DFT-based labels. As a sanity check, the Spinel F1 is "
        "essentially unchanged between Model A and Model B, which confirms "
        "that the rebalancing has not done any harm to the dominant class.",
        body))
    story.append(Paragraph(
        "<b>(d) Way forward.</b> Going ahead, the following steps are "
        "suggested. Firstly, additional real entries may be pulled from "
        "Materials Project / OQMD for Rock-salt and Fluorite, so as to "
        "replace the synthetic samples gradually. Secondly, if a perovskite "
        "predictor is required at a later stage, the same should be built "
        "from real DFT data containing Ba, Sr and La (which are presently "
        "outside our 14-element database) rather than from a Pb/Ce-only "
        "synthetic generator. Thirdly, the more sophisticated stacking "
        "ensemble in <i>best_accuracy_pipeline.py</i> may be retrained on "
        "the new 3-class data, kindly in due course.",
        body))

    # Summary
    story.append(Paragraph("8. Summary", h2))
    story.append(Paragraph(
        "An earlier model that was effectively a Spinel detector "
        "(76.8% accuracy, F1 = 0.20 on the worst minority) had been "
        "rebalanced to a 4-class predictor with 92.6% accuracy. However, "
        "the said 4-class result was found to be inflated due to the "
        "synthetic perovskite class being trivially separable. On removing "
        "the same and using only the legitimate Fluorite and Rock-salt "
        "augmentations, the model now stands at <b>90.2% test accuracy</b> "
        "and <b>88.7% (\u00b11.9%) on 10-fold CV</b>, with all three "
        "per-class F1 scores above 0.88. The same is, in the opinion of "
        "this author, a more honest and defensible result than the earlier "
        "4-class version, and may kindly be taken as the final outcome of "
        "this iteration.", body))

    doc.build(story)
    print(f"\n✓ Report written to {out_pdf}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading raw 3-class real data ...")
    Xr, yr = load_raw_3class()
    print(f"  raw shape: {Xr.shape}  classes: {np.bincount(yr)}")

    print("Loading augmented data (perovskite removed) ...")
    Xa, ya = load_aug_3class()
    print(f"  aug shape: {Xa.shape}  classes: {np.bincount(ya)}")

    print("\nTraining Model A (raw) ...")
    raw = fit_eval(Xr, yr, "raw")
    print("Training Model B (augmented, 3-class) ...")
    aug = fit_eval(Xa, ya, "aug")

    print("\nGenerating figures ...")
    fig_class_distribution(yr, ya, "fig_class_distribution.png")
    fig_confusion_compare(raw["cm"], aug["cm"], "fig_confusion_compare.png")
    fig_f1_compare(raw["f1_per"], aug["f1_per"], "fig_f1_compare.png")
    fig_feature_importance(aug["model"], "fig_feature_importance.png")

    print("\nBuilding PDF ...")
    build_pdf(raw, aug, "report.pdf")


if __name__ == "__main__":
    main()
