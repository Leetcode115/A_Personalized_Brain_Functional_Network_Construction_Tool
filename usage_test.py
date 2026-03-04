#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""End-to-end smoke tests for the A_Personalized_Brain_Functional_Network_Construction_Tool.

This script exercises:
- BrainAtlasManager: listing and loading atlases
- PFBNMLPCorrelationEstimator:
  - basic ROI-level usage (using the provided dtseries example as pseudo-ROIs)
  - atlas-based usage (using a CIFTI dlabel atlas with synthetic vertex-level data)
- BrainVisualizer: visualization of connectivity matrices and basic network statistics

Run from the repository root:
    python usage_test.py
"""

import os
import numpy as np
import nibabel as nib

from atlas_manager import BrainAtlasManager
from correlation_estimator import PFBNMLPCorrelationEstimator
from visualization import BrainVisualizer


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def test_atlas_manager():
    print("=== Atlas manager ===")
    am = BrainAtlasManager()
    atlases = am.get_available_atlases()
    print("Available atlases:", atlases)

    # Try to load AD atlas if present
    if "AD" in atlases and am.atlas_exists("AD"):
        ad_atlas = am.load_atlas("AD")
        print("Loaded AD atlas; data shape:", ad_atlas.get_fdata().shape)
    else:
        ad_atlas = None
        print("AD atlas not found on disk; atlas-based tests will use another category or be skipped.")

    return ad_atlas


def test_pfbn_basic_with_dtseries():
    print("
=== PFBN distilled MLP: basic usage with dtseries example ===")

    dtseries_path = os.path.join(
        REPO_ROOT,
        "test_data",
        "sub-0012-gsp_run-01_DCANBOLDProc_v4.0.0_Atlas_resid.dtseries.nii",
    )
    if not os.path.exists(dtseries_path):
        print("⚠️ dtseries example not found:", dtseries_path)
        return None

    img = nib.load(dtseries_path)
    data = img.get_fdata()
    print("dtseries data shape:", data.shape)

    # For this smoke test, treat the first 200 rows as 200 pseudo-ROIs
    num_rois = 200
    vertices, time_points = data.shape
    if vertices < num_rois:
        print(
            f"⚠️ Not enough vertices ({vertices}) to form {num_rois} pseudo-ROIs; skipping PFBN basic test on dtseries."
        )
        return None

    fmri_data = data[:num_rois, :].astype(np.float32)

    estimator = PFBNMLPCorrelationEstimator(num_rois=num_rois)
    conn = estimator.estimate_correlation(fmri_data)
    print("Correlation matrix shape (basic PFBN usage):", conn.shape)

    return conn


def test_pfbn_with_atlas(ad_atlas):
    print("
=== PFBN distilled MLP with atlas (synthetic vertex-level data) ===")

    if ad_atlas is None:
        print("⚠️ No AD atlas loaded; skipping atlas-based PFBN test.")
        return None

    adata = ad_atlas.get_fdata()
    print("AD atlas data shape:", adata.shape)

    # For CIFTI dlabel (1, V) or volumetric cases, use the total number of elements as vertex count
    vertices = adata.size
    num_rois = 200

    fmri_vertex = np.random.randn(vertices, 200).astype(np.float32)

    estimator = PFBNMLPCorrelationEstimator(num_rois=num_rois)
    conn = estimator.compute_network_from_atlas(fmri_vertex, ad_atlas)
    print("Atlas-based connectivity matrix shape:", conn.shape)

    return conn


def test_visualization(connectivity_matrix):
    print("
=== BrainVisualizer on PFBN connectivity matrix ===")

    if connectivity_matrix is None:
        print("⚠️ No connectivity matrix provided; skipping visualization tests.")
        return

    viz = BrainVisualizer()

    try:
        viz.visualize_functional_network(
            connectivity_matrix,
            title="Test Functional Network (PFBN)",
        )
    except Exception as e:
        print("visualize_functional_network failed:", e)

    try:
        viz.plot_network_statistics(
            connectivity_matrix,
            title="Test Network Statistics (PFBN)",
        )
    except Exception as e:
        print("plot_network_statistics failed:", e)

    try:
        viz.plot_connectivity_histogram(
            connectivity_matrix,
            title="Test Connectivity Histogram (PFBN)",
        )
    except Exception as e:
        print("plot_connectivity_histogram failed:", e)


def main():
    ad_atlas = test_atlas_manager()

    conn_basic = test_pfbn_basic_with_dtseries()

    conn_atlas = test_pfbn_with_atlas(ad_atlas)

    # Prefer atlas-based connectivity for visualization if available; otherwise use basic one
    conn_for_viz = conn_atlas if conn_atlas is not None else conn_basic
    test_visualization(conn_for_viz)

    print("
All usage tests finished.")


if __name__ == "__main__":
    main()
