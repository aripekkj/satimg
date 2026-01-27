"""

Script to perform statistical test for class separability 

"""
import numpy as np
import rasterio
from itertools import combinations
from scipy.stats import kruskal, ttest_ind
import math

# input files
data_raster = "D:/BlueZan/S2/Chwaka/S2_2018_chwaka.tif" # Continuous raster
#class_raster = "D:/BlueZan/S2/Chwaka/obia/ZS_rasterized.tif" # Classification raster
class_raster = "D:/BlueZan/S2/Chwaka/obia/BlueZan_rasterized.tif"

# --------------------------- #

def read_raster_as_array(path):
    """Reads a raster file and returns its data as a NumPy array with NaNs for nodata."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr

def jeffries_matusita(vals1, vals2):
    """
    Computes Jeffries–Matusita separability index for two 1D arrays.
    """
    mu1, mu2 = np.mean(vals1), np.mean(vals2)
    var1, var2 = np.var(vals1, ddof=1), np.var(vals2, ddof=1)

    # Avoid division by zero
    if var1 == 0 or var2 == 0:
        return np.nan

    B = (1/8) * ((mu1 - mu2) ** 2) / (var1 + var2) \
        + 0.5 * math.log((var1 + var2) / (2 * math.sqrt(var1 * var2)))

    JM = 2 * (1 - math.exp(-B))
    return JM

def test_class_separability_kruskal_jm(data_raster_path, class_raster_path, equal_var=False):
    """
    Tests class separability in a single raster using:
      - Kruskal–Wallis H-test (overall difference)
      - Pairwise t-tests
      - Jeffries–Matusita separability index
    
    Returns:
        kruskal_result: (H_stat, p_value)
        pairwise_results: dict {(class1, class2): (t_stat, p_value, JM)}
    """
    data_arr = read_raster_as_array(data_raster_path)
    class_arr = read_raster_as_array(class_raster_path)

    if data_arr.shape != class_arr.shape:
        raise ValueError("Data raster and class raster must have the same dimensions.")

    # Get unique classes
    unique_classes = np.unique(class_arr[~np.isnan(class_arr)])

    # Collect values per class
    class_values = {}
    for cls in unique_classes:
        mask = (class_arr == cls) & ~np.isnan(data_arr)
        vals = data_arr[mask]
        if vals.size > 0:
            class_values[cls] = vals

    # Kruskal–Wallis test (non-parametric)
    if len(class_values) > 1:
        kruskal_result = kruskal(*class_values.values())
    else:
        raise ValueError("Not enough classes for Kruskal–Wallis test.")

    # Pairwise t-tests + JM
    pairwise_results = {}
    for cls1, cls2 in combinations(class_values.keys(), 2):
        vals1 = class_values[cls1]
        vals2 = class_values[cls2]
        if vals1.size > 1 and vals2.size > 1:
            t_stat, p_val = ttest_ind(vals1, vals2, equal_var=equal_var)
            jm_val = jeffries_matusita(vals1, vals2)
            pairwise_results[(cls1, cls2)] = (t_stat, p_val, jm_val)
        else:
            pairwise_results[(cls1, cls2)] = (np.nan, np.nan, np.nan)

    return kruskal_result, pairwise_results

if __name__ == "__main__":

    try:
        kruskal_result, pairwise_results = test_class_separability_kruskal_jm(data_raster, class_raster)

        print(f"Kruskal–Wallis → H-stat: {kruskal_result.statistic:.4f}, P-value: {kruskal_result.pvalue:.6f}")
        if kruskal_result.pvalue < 0.05:
            print("✅ Significant difference between at least two classes.")
        else:
            print("❌ No significant difference between classes.")

        print("\nPairwise Results (T-Test + JM):")
        for (cls1, cls2), (t_stat, p_val, jm_val) in pairwise_results.items():
            print(f"Class {int(cls1)} vs Class {int(cls2)} → "
                  f"T-stat: {t_stat:.4f}, P-value: {p_val:.6f}, JM: {jm_val:.4f}")

    except Exception as e:
        print(f"Error: {e}")
