#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for the Personalized Brain Functional Network Construction Tool.

This script allows the tool to be installed as a Python package.
"""

from setuptools import setup, find_packages
import os


def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements


def read_long_description():
    """Read long description from README.md if available, otherwise use short description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """
A Personalized Brain Functional Network Construction Tool

This tool provides access to pre-built brain atlases covering diverse heterogeneous scenarios, including:
- Brain parcellations targeting five major disorders (AD, ASD, MDD, ADHD, and PD)
- Brain parcellations adapted for both long and short scan duration strategies
- Brain parcellations oriented towards Chinese and English linguistic contexts
- Brain parcellations derived from Movie and Retinotopy tasks
- Age-specific brain atlases for children, adolescents, and the elderly

The neural activity correlation estimation model has been distilled into a lightweight two-layer Multi-Layer Perceptron (MLP).
Basic visualization methods for brain parcellations and personalized brain functional networks are also integrated.
        """.strip()


setup(
    name='PersonalizedBrainNetworkTool',
    version='1.0.0',
    description='A Personalized Brain Functional Network Construction Tool',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    url='https://github.com/yourusername/PersonalizedBrainNetworkTool',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'brain-network-tool = A_Personalized_Brain_Functional_BrainNetwork_Construction_Tool.__init__:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.7',
    include_package_data=True,
)