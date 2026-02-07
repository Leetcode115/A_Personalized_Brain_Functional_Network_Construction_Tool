#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain Atlas Manager

This module manages the pre-built brain atlases and provides access to them
based on different categories (disorders, acquisition duration, linguistic contexts,
tasks, and age groups).
"""

import os
import nibabel as nib
import numpy as np


class BrainAtlasManager:
    """
    Manages access to pre-built brain atlases for various scenarios.

    The atlases are organized into categories:
    - Disorders: AD, ASD, MDD, ADHD, PD
    - Acquisition duration: long acquisition, short acquisition
    - Linguistic contexts: Chinese, English
    - Tasks: Movie, Retinotopy
    - Age groups: children, adolescents, the elderly
    """

    def __init__(self):
        """Initialize the atlas manager with predefined atlas paths."""
        self.atlas_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'atlases')
        self.atlases = self._define_atlases()

    def _define_atlases(self):
        """Define the atlas paths for different categories."""
        return {
            "AD": os.path.join(self.atlas_dir, "AD", "SdTemplate.dlabel.nii"),
            "ASD": os.path.join(self.atlas_dir, "ASD", "SdTemplate.dlabel.nii"),
            "MDD": os.path.join(self.atlas_dir, "MDD", "SdTemplate.dlabel.nii"),
            "ADHD": os.path.join(self.atlas_dir, "ADHD", "SdTemplate.dlabel.nii"),
            "PD": os.path.join(self.atlas_dir, "PD", "SdTemplate.dlabel.nii"),
            "long acquisition": os.path.join(self.atlas_dir, "long_acquisition", "SdTemplate.dlabel.nii"),
            "short acquisition": os.path.join(self.atlas_dir, "short_acquisition", "SdTemplate.dlabel.nii"),
            "Chinese": os.path.join(self.atlas_dir, "Chinese", "SdTemplate.dlabel.nii"),
            "English": os.path.join(self.atlas_dir, "English", "SdTemplate.dlabel.nii"),
            "Movie": os.path.join(self.atlas_dir, "Movie", "SdTemplate.dlabel.nii"),
            "Retinotopy": os.path.join(self.atlas_dir, "Retinotopy", "SdTemplate.dlabel.nii"),
            "children": os.path.join(self.atlas_dir, "children", "SdTemplate.dlabel.nii"),
            "adolescents": os.path.join(self.atlas_dir, "adolescents", "SdTemplate.dlabel.nii"),
            "the elderly": os.path.join(self.atlas_dir, "the_elderly", "SdTemplate.dlabel.nii")
        }

    def get_available_atlases(self):
        """
        Get the list of available atlas categories.

        Returns:
            list: Sorted list of available atlas categories
        """
        return sorted(self.atlases.keys())

    def load_atlas(self, category):
        """
        Load a brain atlas from the specified category.

        Args:
            category (str): Atlas category

        Returns:
            nibabel.nifti1.Nifti1Image: Loaded atlas image

        Raises:
            ValueError: If category is not recognized
            FileNotFoundError: If atlas file does not exist
        """
        if category not in self.atlases:
            raise ValueError(f"Unknown atlas category: {category}. Available categories: {self.get_available_atlases()}")

        atlas_path = self.atlases[category]

        if not os.path.exists(atlas_path):
            raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

        try:
            return nib.load(atlas_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load atlas: {e}")

    def atlas_exists(self, category):
        """
        Check if an atlas exists for the specified category.

        Args:
            category (str): Atlas category

        Returns:
            bool: True if atlas exists, False otherwise
        """
        if category not in self.atlases:
            return False

        return os.path.exists(self.atlases[category])

    def get_atlas_path(self, category):
        """
        Get the file path for the specified atlas category.

        Args:
            category (str): Atlas category

        Returns:
            str: Atlas file path

        Raises:
            ValueError: If category is not recognized
        """
        if category not in self.atlases:
            raise ValueError(f"Unknown atlas category: {category}. Available categories: {self.get_available_atlases()}")

        return self.atlases[category]

    def list_available_atlases_with_details(self):
        """
        List available atlases with detailed information.

        Returns:
            dict: Dictionary containing atlas details
        """
        details = {}
        for category in self.get_available_atlases():
            path = self.get_atlas_path(category)
            exists = self.atlas_exists(category)
            details[category] = {
                "path": path,
                "exists": exists,
                "size": os.path.getsize(path) if exists else 0
            }
        return details