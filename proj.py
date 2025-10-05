# Capstone Project 1 – Working with NumPy Matrices (NHANES)
# Author: Debojyoti
# Notes:
# - Place the CSV files in the same folder as this script:
#     nhanes_adult_male_bmx_2020.csv
#     nhanes_adult_female_bmx_2020.csv
# - This script saves plots as PNG images in the current folder.

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utility functions
# -----------------------------

def load_and_clean_csv(path):
    """Load CSV, skip header, drop rows with NaN/Inf."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    data = data[np.all(np.isfinite(data), axis=1)]
    return data

def clean_column(arr):
    """Return only finite values from a column."""
    return arr[np.isfinite(arr)]

def zscore_matrix(X, ddof=0):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=ddof)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma_safe, mu, sigma_safe

def iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

def skewness(x):
    m = x.mean(); s = x.std(ddof=0)
    return 0.0 if s == 0 else np.mean((x - m) ** 3) / (s ** 3)

def excess_kurtosis(x):
    m = x.mean(); s = x.std(ddof=0)
    return 0.0 if s == 0 else np.mean((x - m) ** 4) / (s ** 4) - 3.0

def compute_basic_aggregates(x):
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)),
        "var": float(np.var(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "iqr": float(iqr(x)),
        "skewness": float(skewness(x)),
        "excess_kurtosis": float(excess_kurtosis(x)),
    }

def rankdata_tieavg(x):
    order = np.argsort(x, kind='mergesort')
    sorted_x = x[order]
    ranks = np.empty_like(sorted_x, dtype=float)
    n = len(sorted_x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[i:j + 1] = avg_rank
        i = j + 1
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    return ranks[inv_order]

def pearson_corr_matrix(X):
    return np.corrcoef(X, rowvar=False)

def spearman_corr_matrix(X):
    ranks = np.column_stack([rankdata_tieavg(X[:, j]) for j in range(X.shape[1])])
    return np.corrcoef(ranks, rowvar=False)

def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"[Saved] {filename}")
    plt.close(fig)

# -----------------------------
# 1–2. Load data
# -----------------------------

male = load_and_clean_csv("nhanes_adult_male_bmx_2020.csv")
female = load_and_clean_csv("nhanes_adult_female_bmx_2020.csv")

print("Shapes after cleaning:")
print(f"  male   : {male.shape}")
print(f"  female : {female.shape}")

# -----------------------------
# 3. Histograms of weights
# -----------------------------

female_weights = clean_column(female[:, 0])
male_weights = clean_column(male[:, 0])

w_min = min(female_weights.min(), male_weights.min())
w_max = max(female_weights.max(), male_weights.max())
pad = 0.05 * (w_max - w_min)
xlim = (w_min - pad, w_max + pad)

fig_hist, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].hist(female_weights, bins=30, color="#1f77b4", alpha=0.8, edgecolor="white")
axes[0].set_title("Female weight distribution")
axes[0].set_ylabel("Count")
axes[0].set_xlim(xlim)

axes[1].hist(male_weights, bins=30, color="#ff7f0e", alpha=0.8, edgecolor="white")
axes[1].set_title("Male weight distribution")
axes[1].set_xlabel("Weight (kg)")
axes[1].set_ylabel("Count")
axes[1].set_xlim(xlim)

save_figure(fig_hist, "hist_weights_female_top_male_bottom.png")

# -----------------------------
# 4. Boxplot: male vs female weights
# -----------------------------

fig_box_w, ax_box_w = plt.subplots(figsize=(7, 5))
ax_box_w.boxplot([female_weights, male_weights], labels=["Female", "Male"], patch_artist=True)
ax_box_w.set_title("Boxplot: Weight comparison (Female vs Male)")
ax_box_w.set_ylabel("Weight (kg)")
save_figure(fig_box_w, "boxplot_weights_female_vs_male.png")

female_agg = compute_basic_aggregates(female_weights)
male_agg = compute_basic_aggregates(male_weights)
print("\nWeight aggregates summary:")
print("  Female:", female_agg)
print("  Male  :", male_agg)

# -----------------------------
# 5. Distribution description
# -----------------------------

def describe_distribution(name, agg):
    print(f"\n{name} distribution characteristics:")
    print(f"  - Location: mean={agg['mean']:.2f}, median={agg['median']:.2f}")
    print(f"  - Dispersion: std={agg['std']:.2f}, IQR={agg['iqr']:.2f}, "
          f"range={agg['min']:.2f}–{agg['max']:.2f}")
    print(f"  - Shape: skewness={agg['skewness']:.3f}, "
          f"excess kurtosis={agg['excess_kurtosis']:.3f}")

describe_distribution("Female weight", female_agg)
describe_distribution("Male weight", male_agg)

# -----------------------------
# 6. Add BMI to female matrix
# -----------------------------

female_height_m = clean_column(female[:, 1]) / 100.0
female_bmi = female[:, 0] / (female_height_m ** 2)
female_with_bmi = np.column_stack([female, female_bmi])
print(f"\nFemale matrix with BMI added: shape={female_with_bmi.shape}")

# -----------------------------
# 7. Standardize female matrix
# -----------------------------

zfemale, female_mu, female_sigma = zscore_matrix(female_with_bmi, ddof=0)
print("zfemale computed (standardized columns).")

# -----------------------------
# 8. Scatterplot matrix + correlations
# -----------------------------

def rankdata_tieavg(x):
    order = np.argsort(x, kind='mergesort')
    sorted_x = x[order]
    ranks = np.empty_like(sorted_x, dtype=float)
    n = len(sorted_x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[i:j + 1] = avg_rank
        i = j + 1
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    return ranks[inv_order]

def pearson_corr_matrix(X):
    return np.corrcoef(X, rowvar=False)

def spearman_corr_matrix(X):
    ranks = np.column_stack([rankdata_tieavg(X[:, j]) for j in range(X.shape[1])])
    return np.corrcoef(ranks, rowvar=False)

cols_labels = ["Height (z)", "Weight (z)", "Waist (z)", "Hip (z)", "BMI (z)"]
cols_idx = [1, 0, 6, 5, 7]
Z = zfemale[:, cols_idx]

k = Z.shape[1]
fig_scm, axes = plt.subplots(k, k, figsize=(10, 10))
for i in range(k):
    for j in range(k):
        ax = axes[i, j]
        if i == j:
            ax.hist(Z[:, j], bins=25, color="#6baed6", alpha=0.9, edgecolor="white")
        else:
            ax.scatter(Z[:, j], Z[:, i], s=12, alpha=0.7, color="#2ca25f")
        if i == k - 1:
            ax.set_xlabel(cols_labels[j], fontsize=8)
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(cols_labels[i], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=7)

fig_scm.suptitle("Scatterplot matrix (standardized females: height, weight, waist, hip, BMI)", fontsize=12)
save_figure(fig_scm, "scatter_matrix_female_standardized.png")

pearson_mat = pearson_corr_matrix(Z)
spearman_mat = spearman_corr_matrix(Z)

print("\nPearson correlation matrix:")
print(cols_labels)
print(pearson_mat)

print("\nSpearman correlation matrix:")
print(cols_labels)
print(spearman_mat)

# -----------------------------
# 9. Add waist-to-height and waist-to-hip ratios
# -----------------------------

def add_waist_ratios(X):
    height_cm = X[:, 1]
    hip_cm = X[:, 5]
    waist_cm = X[:, 6]
    height_safe = np.where(height_cm == 0, np.nan, height_cm)
    hip_safe = np.where(hip_cm == 0, np.nan, hip_cm)
    whtr = waist_cm / height_safe
    whr = waist_cm / hip_safe
    return np.column_stack([X, whtr, whr])

male_with_ratios = add_waist_ratios(male)                 # adds cols [7]=WHtR, [8]=WHR
female_with_ratios = add_waist_ratios(female_with_bmi)    # adds cols [8]=WHtR, [9]=WHR

print("\nAdded ratios:")
print(f"  male_with_ratios shape   : {male_with_ratios.shape}")
print(f"  female_with_ratios shape : {female_with_ratios.shape}")

# -----------------------------
# 10. Boxplot: WHtR and WHR
# -----------------------------

male_whtr = male_with_ratios[:, 7]
male_whr = male_with_ratios[:, 8]
female_whtr = female_with_ratios[:, 8]
female_whr = female_with_ratios[:, 9]

fig_box_r, ax_box_r = plt.subplots(figsize=(8, 5))
ax_box_r.boxplot(
    [female_whtr, male_whtr, female_whr, male_whr],
    labels=["Female WHtR", "Male WHtR", "Female WHR", "Male WHR"],
    patch_artist=True
)
ax_box_r.set_title("Boxplot: Waist-to-Height and Waist-to-Hip Ratios (Female vs Male)")
ax_box_r.set_ylabel("Ratio (unitless)")
save_figure(fig_box_r, "boxplot_WHtR_WHR_female_vs_male.png")

# -----------------------------
# 11. Advantages and disadvantages
# -----------------------------

print("\nMeasures: advantages and disadvantages")

print("BMI:")
print("  + Simple to compute; widely used; correlates with overall adiposity.")
print("  - Does not distinguish fat vs muscle; ignores fat distribution; can misclassify athletic/elderly individuals.")

print("WHtR (waist/height):")
print("  + Captures central adiposity; useful across different heights; predictive of cardiometabolic risk.")
print("  - Requires accurate waist measurement; thresholds vary; does not separate visceral vs subcutaneous fat.")

print("WHR (waist/hip):")
print("  + Indicates fat distribution; associated with metabolic risk beyond BMI.")
print("  - Hip circumference measurement variability; influenced by body shape, sex differences; less direct for total adiposity.")

# -----------------------------
# 12. Lowest and highest BMI standardized rows
# -----------------------------

bmi_idx = 7  # BMI column in female_with_bmi and zfemale
order_bmi = np.argsort(female_with_bmi[:, bmi_idx])
lowest5_idx = order_bmi[:5]
highest5_idx = order_bmi[-5:]

print("\nStandardized female measurements (z-scores) for 5 lowest BMI:")
print("Rows (indices):", lowest5_idx.tolist())
print(zfemale[lowest5_idx, :])

print("\nStandardized female measurements (z-scores) for 5 highest BMI:")
print("Rows (indices):", highest5_idx.tolist())
print(zfemale[highest5_idx, :])

print("\nInterpretation (extreme BMI z-scores):")
print("  • Lowest BMI rows should show negative z-scores in weight/BMI/waist/hip relative to the cohort.")
print("  • Highest BMI rows should show positive z-scores in those measures.")
print("  • Height z-scores may vary independently, reinforcing BMI’s dependence on both weight and height.")

print("\nAll tasks completed. Generated figures:")
print("  - hist_weights_female_top_male_bottom.png")
print("  - boxplot_weights_female_vs_male.png")
print("  - scatter_matrix_female_standardized.png")
print("  - boxplot_WHtR_WHR_female_vs_male.png")