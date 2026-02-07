#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain Visualization Module

This module provides basic visualization methods for brain parcellations
and personalized brain functional networks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting


class BrainVisualizer:
    """
    Provides visualization methods for brain parcellations and functional networks.

    This class includes:
    - Visualization of brain atlases (surface and volume views)
    - Visualization of functional connectivity matrices
    - Basic statistical plots for brain network analysis
    """

    def __init__(self, figure_dir=None):
        """
        Initialize the brain visualizer.

        Args:
            figure_dir (str): Directory to save generated figures
        """
        self.figure_dir = figure_dir or os.path.join(
            os.path.dirname(__file__), 'figures'
        )
        os.makedirs(self.figure_dir, exist_ok=True)

    def visualize_atlas(self, atlas, title="Brain Atlas", output_path=None):
        """
        Visualize a brain atlas.

        Args:
            atlas (nibabel.nifti1.Nifti1Image): Loaded atlas image
            title (str): Visualization title
            output_path (str): Path to save the visualization (optional)
        """
        output_path = output_path or os.path.join(
            self.figure_dir, f"{title.lower().replace(' ', '_')}.png"
        )

        try:
            # Create a figure with multiple views
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Try simple volume visualization since surface view may fail
            self._visualize_atlas_volume(atlas, title, output_path)

        except Exception as e:
            print(f"❌ Error visualizing atlas: {e}")
            # Fallback to simple volume view if surface view fails
            self._visualize_atlas_volume(atlas, title, output_path)

    def _visualize_atlas_volume(self, atlas, title="Brain Atlas", output_path=None):
        """
        Fallback visualization method using volume view.

        Args:
            atlas (nibabel.nifti1.Nifti1Image): Loaded atlas image
            title (str): Visualization title
            output_path (str): Path to save the visualization (optional)
        """
        try:
            atlas_data = atlas.get_fdata()
            if atlas_data.ndim == 4:
                atlas_data = np.argmax(atlas_data, axis=-1)

            # For CIFTI files with shape (1, 64984), we need to handle differently
            if atlas_data.shape == (1, 64984):
                self._visualize_cifti_atlas(atlas, title, output_path)
                return

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Axial, coronal, sagittal views
            axial_slice = atlas_data.shape[2] // 2
            coronal_slice = atlas_data.shape[1] // 2
            sagittal_slice = atlas_data.shape[0] // 2

            axes[0].imshow(atlas_data[:, :, axial_slice], cmap='tab20', origin='lower')
            axes[0].set_title(f"{title} - Axial Slice")
            axes[0].axis('off')

            axes[1].imshow(atlas_data[:, coronal_slice, :], cmap='tab20', origin='lower')
            axes[1].set_title(f"{title} - Coronal Slice")
            axes[1].axis('off')

            axes[2].imshow(atlas_data[sagittal_slice, :, :], cmap='tab20', origin='lower')
            axes[2].set_title(f"{title} - Sagittal Slice")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Atlas visualization (volume view) saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error in fallback visualization: {e}")

    def _visualize_cifti_atlas(self, atlas, title="Brain Atlas", output_path=None):
        """
        Visualization method for CIFTI format atlases.

        Args:
            atlas (nibabel.Cifti2Image): Loaded CIFTI atlas image
            title (str): Visualization title
            output_path (str): Path to save the visualization (optional)
        """
        try:
            fig, ax = plt.subplots(figsize=(15, 6))

            # Get atlas data
            atlas_data = atlas.get_fdata()
            if atlas_data.shape == (1, 64984):
                atlas_data = atlas_data[0]

            # Plot the atlas data as a bar chart or histogram
            ax.plot(atlas_data, 'b-', linewidth=0.5, alpha=0.7)
            ax.set_title(f"{title} - Vertex-wise Parcellation")
            ax.set_xlabel('Vertex Index')
            ax.set_ylabel('Parcel Label')
            ax.set_ylim(0, int(atlas_data.max()) + 1)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ CIFTI atlas visualization saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error visualizing CIFTI atlas: {e}")

    def visualize_functional_network(self, connectivity_matrix, title="Functional Network",
                                     output_path=None, cmap='coolwarm', vmin=-1, vmax=1):
        """
        Visualize a functional brain network connectivity matrix.

        Args:
            connectivity_matrix (np.ndarray): Functional connectivity matrix
            title (str): Visualization title
            output_path (str): Path to save the visualization (optional)
            cmap (str): Colormap for the matrix
            vmin (float): Minimum value for color scale
            vmax (float): Maximum value for color scale
        """
        output_path = output_path or os.path.join(
            self.figure_dir, f"{title.lower().replace(' ', '_')}_matrix.png"
        )

        try:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot connectivity matrix
            im = ax.imshow(connectivity_matrix, cmap=cmap, vmin=vmin, vmax=vmax)

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient')

            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Regions of Interest', fontsize=12)
            ax.set_ylabel('Regions of Interest', fontsize=12)

            # Add grid and adjust layout
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Functional network visualization saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error visualizing functional network: {e}")

    def plot_network_statistics(self, connectivity_matrix, title="Network Statistics",
                                output_path=None):
        """
        Plot statistical properties of a functional brain network.

        Args:
            connectivity_matrix (np.ndarray): Functional connectivity matrix
            title (str): Visualization title
            output_path (str): Path to save the visualization (optional)
        """
        output_path = output_path or os.path.join(
            self.figure_dir, f"{title.lower().replace(' ', '_')}_statistics.png"
        )

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Histogram of connection strengths
            axes[0, 0].hist(connectivity_matrix.flatten(), bins=50, alpha=0.7)
            axes[0, 0].set_title('Connection Strength Distribution')
            axes[0, 0].set_xlabel('Correlation Coefficient')
            axes[0, 0].set_ylabel('Frequency')

            # Degree distribution
            degrees = np.sum(np.abs(connectivity_matrix) > 0.2, axis=1)  # Thresholded degree
            axes[0, 1].hist(degrees, bins=20, alpha=0.7)
            axes[0, 1].set_title('Node Degree Distribution')
            axes[0, 1].set_xlabel('Number of Connections')
            axes[0, 1].set_ylabel('Frequency')

            # Mean connectivity per node
            mean_connectivity = np.mean(connectivity_matrix, axis=1)
            axes[1, 0].plot(mean_connectivity, 'o-', markersize=2)
            axes[1, 0].set_title('Mean Connectivity per Node')
            axes[1, 0].set_xlabel('Node Index')
            axes[1, 0].set_ylabel('Mean Correlation')

            # Matrix sparsity
            sparsity = (np.abs(connectivity_matrix) > 0.2).mean()
            axes[1, 1].text(0.5, 0.5, f"Sparsity: {sparsity:.2%}",
                           fontsize=16, ha='center', va='center',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Network Sparsity')
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Network statistics plot saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error plotting network statistics: {e}")

    def compare_networks(self, network1, network2, title1="Network 1",
                         title2="Network 2", output_path=None):
        """
        Compare two functional brain networks.

        Args:
            network1 (np.ndarray): First connectivity matrix
            network2 (np.ndarray): Second connectivity matrix
            title1 (str): Title for first network
            title2 (str): Title for second network
            output_path (str): Path to save the visualization (optional)
        """
        output_path = output_path or os.path.join(
            self.figure_dir, "network_comparison.png"
        )

        try:
            fig, axes = plt.subplots(1, 3, figsize=(20, 7))

            # Network 1
            im1 = axes[0].imshow(network1, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0].set_title(title1)
            axes[0].set_xlabel('Regions of Interest')
            axes[0].set_ylabel('Regions of Interest')

            # Network 2
            im2 = axes[1].imshow(network2, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1].set_title(title2)
            axes[1].set_xlabel('Regions of Interest')
            axes[1].set_ylabel('Regions of Interest')

            # Difference
            diff = network1 - network2
            im3 = axes[2].imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
            axes[2].set_title('Difference (Network1 - Network2)')
            axes[2].set_xlabel('Regions of Interest')
            axes[2].set_ylabel('Regions of Interest')

            # Add colorbars
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Network comparison plot saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error comparing networks: {e}")

    def plot_connectivity_histogram(self, connectivity_matrix, title="Connectivity Histogram",
                                     output_path=None, threshold=0.2):
        """
        Plot histogram of connectivity strengths with thresholding.

        Args:
            connectivity_matrix (np.ndarray): Functional connectivity matrix
            title (str): Visualization title
            output_path (str): Path to save the visualization (optional)
            threshold (float): Threshold for significant connections
        """
        output_path = output_path or os.path.join(
            self.figure_dir, f"{title.lower().replace(' ', '_')}.png"
        )

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Flatten matrix and filter out diagonal
            upper_tri = np.triu(connectivity_matrix, k=1).flatten()

            # Create histogram
            ax.hist(upper_tri, bins=50, alpha=0.7, label='All Connections')

            # Highlight significant connections
            significant = upper_tri[np.abs(upper_tri) > threshold]
            ax.hist(significant, bins=50, alpha=0.7, label=f'Significant (|r| > {threshold})')

            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Correlation Coefficient', fontsize=12)
            ax.set_ylabel('Number of Connections', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Connectivity histogram saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error plotting connectivity histogram: {e}")