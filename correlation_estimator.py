#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Activity Correlation Estimator

This module implements the neural activity correlation estimation using
the lightweight two-layer MLP model. The model is trained to estimate
functional brain network connectivity from fMRI data.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from nilearn.connectome import ConnectivityMeasure

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


class LightweightMLPCorrelation(nn.Module):
    """
    Lightweight two-layer MLP for neural activity correlation estimation.

    This model distills the neural activity correlation estimation from
    the BrainMass model into a lightweight architecture.
    """

    def __init__(self, in_channels=90, hidden_size=256, out_channels=90):
        """
        Initialize the lightweight MLP model.

        Args:
            in_channels (int): Number of input channels (default: 90)
            hidden_size (int): Hidden layer size (default: 256)
            out_channels (int): Number of output channels (default: 90)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, out_channels)
        )

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input correlation matrix

        Returns:
            torch.Tensor: Output correlation matrix
        """
        return self.mlp(x)


class NeuralActivityCorrelationEstimator:
    """
    Estimates neural activity correlation using the lightweight MLP model.

    This class provides functionality to:
    - Load and initialize the correlation estimation model
    - Estimate functional connectivity from fMRI time series
    - Compute functional brain networks using parcellated data
    """

    def __init__(self, model_path=None):
        """
        Initialize the neural activity correlation estimator.

        Args:
            model_path (str): Path to the trained model checkpoint.
                If None, uses the default PPMI_Our.pth checkpoint.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型路径改为工具目录下的checkpoints文件夹
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'PPMI_Our.pth'
        )
        self.model = self._load_model()
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

        print(f"✅ Correlation estimator initialized (device: {self.device})")

    def _load_model(self):
        """
        Load the trained correlation estimation model.

        Returns:
            torch.nn.Module: Loaded model

        Raises:
            FileNotFoundError: If model checkpoint is not found
            RuntimeError: If model loading fails
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        try:
            # Load the complete BrainFly model
            sys.path.append(os.path.join(project_root, 'Stage3-Adaptive_Correlation_Calculation_Train'))

            from BrainMass.model import BrainFly, MLP1, MLP2
            from BrainMass.utils import BNTF

            from yaml import load
            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader

            # Load configuration
            config_path = os.path.join(
                project_root, 'Stage3-Adaptive_Correlation_Calculation_Train',
                'BrainMass', 'config', 'config.yaml'
            )
            config = load(open(config_path, 'r'), Loader=Loader)

            # Initialize model components
            mlp1 = MLP1(
                in_channels=config['model']['in_channels'],
                mlp_hidden_size=config['model']['hidden_size'],
                projection_size=config['model']['projection_size']
            )

            brain_mass = BNTF(
                feature_dim=config['model']['feature_dim'],
                depth=config['model']['depth'],
                heads=config['model']['heads'],
                dim_feedforward=config['model']['dim_feedforward']
            )

            mlp2 = MLP2(
                in_channels=config['model']['projection_size'],
                mlp_hidden_size=config['model']['hidden_size'],
                projection_size=config['model']['in_channels']
            )

            # Create complete model
            model = BrainFly(mlp1, brain_mass, mlp2)

            # Load checkpoint
            params = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(params)
            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def estimate_correlation(self, fmri_data):
        """
        Estimate neural activity correlation from fMRI time series data.

        Args:
            fmri_data (np.ndarray): fMRI time series data (vertices × time)

        Returns:
            np.ndarray: Functional connectivity matrix

        Raises:
            ValueError: If input data shape is invalid
        """
        if fmri_data.ndim != 2:
            raise ValueError(f"Invalid input shape: expected 2D (vertices × time), got {fmri_data.shape}")

        # Calculate initial correlation matrix
        correlation_matrix = self.correlation_measure.fit_transform([fmri_data.T])[0]

        # Convert to tensor and move to device
        correlation_tensor = torch.tensor(correlation_matrix, dtype=torch.float32, device=self.device)

        # Pass through the model
        with torch.no_grad():
            output = self.model.mlp1(correlation_tensor)
            for atten in self.model.brain_mass.attention_list:
                output = atten(output)
            output = self.model.mlp2(output)
            output = output.detach().cpu().numpy()

        # Calculate final correlation matrix
        final_correlation = self.correlation_measure.fit_transform([output.T])[0]

        return final_correlation

    def compute_network_from_atlas(self, fmri_data, atlas):
        """
        Compute functional brain network using a specific brain atlas.

        Args:
            fmri_data (np.ndarray): fMRI time series data (vertices × time)
            atlas (nibabel.nifti1.Nifti1Image): Brain atlas parcellation

        Returns:
            np.ndarray: Parcellated functional connectivity matrix

        Raises:
            ValueError: If input data or atlas is invalid
        """
        # Extract atlas labels
        atlas_data = atlas.get_fdata()

        # Support both volumetric atlases (3D/4D) and CIFTI dlabel-style vectors (1 x V or V x 1)
        if atlas_data.ndim in (3, 4):
            labels = np.argmax(atlas_data, axis=-1) if atlas_data.ndim == 4 else atlas_data
            atlas_labels = labels.flatten()
        elif atlas_data.ndim == 2 and 1 in atlas_data.shape:
            # CIFTI dlabel: e.g. (1, 64984) or (64984, 1)
            atlas_labels = atlas_data.reshape(-1)
        else:
            raise ValueError(
                f"Invalid atlas shape: expected 3D/4D volume or 1D CIFTI vector, got {atlas_data.shape}"
            )

        # Remove background and invalid labels
        valid_mask = atlas_labels > 0
        valid_labels = atlas_labels[valid_mask]
        valid_fmri = fmri_data[valid_mask]

        # Compute mean time series for each ROI
        unique_labels = np.unique(valid_labels)
        roi_time_series = []

        for label in unique_labels:
            roi_mask = valid_labels == label
            roi_data = valid_fmri[roi_mask]
            roi_mean = np.mean(roi_data, axis=0)
            roi_time_series.append(roi_mean)

        roi_time_series = np.array(roi_time_series)

        # Compute functional connectivity
        connectivity_matrix = self.correlation_measure.fit_transform([roi_time_series.T])[0]

        return connectivity_matrix

    def compute_batch_correlation(self, fmri_dataset):
        """
        Compute correlation matrices for a batch of fMRI datasets.

        Args:
            fmri_dataset (list): List of fMRI time series data arrays

        Returns:
            list: List of functional connectivity matrices
        """
        results = []
        for fmri_data in fmri_dataset:
            try:
                correlation = self.estimate_correlation(fmri_data)
                results.append(correlation)
            except Exception as e:
                print(f"Error processing data: {e}")
                results.append(None)
        return results


class PFBNMLPCorrelationEstimator:
    """Distilled PFBN two-layer MLP estimator.

    This estimator uses the lightweight two-layer MLP (LightweightMLPCorrelation)
    trained offline to approximate the PFBN personalized functional connectivity
    module. It takes ROI-level fMRI time series as input and outputs an enhanced
    functional connectivity matrix.
    """

    def __init__(self, model_path=None, num_rois: int = 200, hidden_size: int = 256):
        import torch
        from nilearn.connectome import ConnectivityMeasure

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_rois = num_rois
        self.hidden_size = hidden_size

        # Default to the distilled PFBN MLP checkpoint in the tool's checkpoints folder
        default_ckpt = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'checkpoints',
            'PFBN_Distilled_MLP.pth',
        )
        self.model_path = model_path or default_ckpt

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Initialize lightweight MLP and load distilled weights
        self.model = LightweightMLPCorrelation(
            in_channels=self.num_rois,
            hidden_size=self.hidden_size,
            out_channels=self.num_rois,
        ).to(self.device)

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.correlation_measure = ConnectivityMeasure(kind='correlation')

        print(f"✅ PFBN distilled MLP estimator initialized (device: {self.device})")

    def estimate_correlation(self, fmri_data):
        """Estimate functional connectivity from ROI-level fMRI time series.

        Args:
            fmri_data (np.ndarray): fMRI time series data (vertices/ROIs × time)

        Returns:
            np.ndarray: Enhanced functional connectivity matrix (num_rois × num_rois)
        """
        import numpy as np
        import torch

        if fmri_data.ndim != 2:
            raise ValueError(
                f"Invalid input shape: expected 2D (vertices × time), got {fmri_data.shape}"
            )

        # Ensure the first dimension corresponds to ROIs if possible
        if fmri_data.shape[0] != self.num_rois and fmri_data.shape[1] == self.num_rois:
            fmri_data = fmri_data.T

        if fmri_data.shape[0] != self.num_rois:
            raise ValueError(
                f"fmri_data should have shape (num_rois, time); got {fmri_data.shape}"
            )

        # Initial Pearson correlation matrix
        correlation_matrix = self.correlation_measure.fit_transform([fmri_data.T])[0]
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        # Convert to tensor and apply the distilled MLP (row-wise)
        correlation_tensor = torch.tensor(
            correlation_matrix, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            enhanced = self.model(correlation_tensor)
            enhanced = enhanced.detach().cpu().numpy()

        return enhanced

    def compute_network_from_atlas(self, fmri_data, atlas):
        """Compute functional brain network using a specific brain atlas.

        This first aggregates vertex-level fMRI signals into ROI-level time series
        using the atlas, then applies the distilled PFBN MLP estimator.
        """
        import numpy as np

        # Extract atlas labels
        atlas_data = atlas.get_fdata()

        # Support both volumetric atlases (3D/4D) and CIFTI dlabel-style vectors (1 x V or V x 1)
        if atlas_data.ndim in (3, 4):
            labels = np.argmax(atlas_data, axis=-1) if atlas_data.ndim == 4 else atlas_data
            atlas_labels = labels.flatten()
        elif atlas_data.ndim == 2 and 1 in atlas_data.shape:
            # CIFTI dlabel: e.g. (1, 64984) or (64984, 1)
            atlas_labels = atlas_data.reshape(-1)
        else:
            raise ValueError(
                f"Invalid atlas shape: expected 3D/4D volume or 1D CIFTI vector, got {atlas_data.shape}"
            )

        # Remove background and invalid labels
        valid_mask = atlas_labels > 0
        valid_labels = atlas_labels[valid_mask]
        valid_fmri = fmri_data[valid_mask]

        # Compute mean time series for each ROI
        unique_labels = np.unique(valid_labels)
        roi_time_series = []

        for label in unique_labels:
            roi_mask = valid_labels == label
            roi_data = valid_fmri[roi_mask]
            roi_mean = np.mean(roi_data, axis=0)
            roi_time_series.append(roi_mean)

        roi_time_series = np.array(roi_time_series)

        return self.estimate_correlation(roi_time_series)

    def compute_batch_correlation(self, fmri_dataset):
        """Compute correlation matrices for a batch of fMRI datasets."""
        results = []
        for fmri_data in fmri_dataset:
            try:
                results.append(self.estimate_correlation(fmri_data))
            except Exception as e:
                print(f"Error processing data: {e}")
                results.append(None)
        return results
