# A_Personalized_Brain_Functional_Network_Construction_Tool
An open tool for constructing personalized brain functional networks has been developed. This tool is designed to establish a flexible ecosystem wherein users may either act as contributors by sharing custom brain parcellations and neural activity correlation estimation methods with the community, or perform customizations exclusively within a local environment to preserve data privacy.

## The overall construction framework
Currently, pre-built brain parcellations covering diverse heterogeneous scenarios are provided, including: brain parcellations targeting five major disorders (AD, ASD, MDD, ADHD, and PD); brain parcellations adapted for both long and short scan duration strategies; brain parcellations oriented towards Chinese and English linguistic contexts; brain parcellations derived from Movie and Retinotopy tasks; and age-specific brain parcellations for children, adolescents, and the elderly. To facilitate local deployment, the neural activity correlation estimation model has been distilled into a lightweight two-layer Multi-Layer Perceptron (MLP). Basic visualization methods for brain parcellations and personalized brain functional networks are also integrated.

## The personalized brain parcellation

### Different age gourps
The tool provides age-specific brain parcellationes optimized for different developmental stages:
- **Children**: parcellation designed for pediatric populations (typically 6-12 years old)
- **Adolescents**: parcellation optimized for teenage populations (typically 13-18 years old)
- **The elderly**: parcellation tailored for older adults (typically 60+ years old)

### Different brain disorder types
Disease-specific parcellationes are available for five major neurological and psychiatric disorders:
- **Alzheimer's Disease (AD)**: parcellation optimized for patients with Alzheimer's disease
- **Attention Deficit Hyperactivity Disorder (ADHD)**: parcellation designed for individuals with ADHD
- **Autism Spectrum Disorder (ASD)**: parcellation tailored for patients with autism spectrum disorder
- **Major Depressive Disorder (MDD)**: parcellation optimized for individuals with major depressive disorder
- **Parkinson's Disease (PD)**: parcellation designed for patients with Parkinson's disease

### Different data acquision stratigies
The tool supports both long and short scan duration strategies:
- **Long acquisition**: parcellation optimized for high-quality fMRI data with long scan durations (typically 50+ minutes)
- **Short acquisition**: parcellation designed for rapid fMRI acquisitions (typically <50 minutes)
  
### Different Linguistics
Linguistic-specific parcellations are available for both Chinese and English linguistic contexts:
- **Chinese**: parcellations optimized for individuals whose linguistic background is Chinese
- **English**: parcellations designed for individuals whose linguistic background is English

### 

## The correlation estimation between neural activities
The tool uses a lightweight two-layer Multi-Layer Perceptron (MLP) model to estimate brain functional networks from fMRI time series data. This model is a distilled version of our personalized brain functional network construction, providing fast and accurate correlation estimation.
Key features of the correlation estimator:
- **Lightweight architecture**: Two-layer MLP with 256 hidden units, optimized for fast computation
- **GPU acceleration**: Supports GPU computation for improved performance
- **Input format**: Accepts fMRI time series data in vertices × time format
- **Output format**: Returns a brain functional network matrix in vertices × vertices format
  
## Visualization
### Brain parcellation visualization
The tool provides visualization capabilities for brain parcellations in CIFTI format:
- **Vertex-level visualization**: Displays the brain parcellations on a cortical surface
- **ROI highlighting**: Highlights specific regions of interest (ROIs)
- **Color mapping**: Supports various color maps for clear visualization
- **Surface rendering**: Uses standard brain templates (fs_LR_32k) for visualization
- **Hemisphere separation**: Supports separate visualization of left and right hemispheres

### Functional brain network visualization
The tool supports various visualization methods for functional brain networks:
- **Heatmap**: Displays the functional connectivity matrix as a heatmap
- **Node strength plot**: Shows the strength of connections for each node (ROI)
- **Connectivity histogram**: Plots the distribution of connection strengths
- **Network statistics**: Visualizes network metrics such as degree distribution and clustering coefficient
- **Thresholding options**: Supports thresholding of connectivity matrices to focus on significant connections

## Useage
pip install git+https://github.com/jianghongjie328/A_Personalized_Brain_Functional_Network_Construction_Tool.git

### Basic Usage

```python
import sys
sys.path.insert(0, '/path/to/A_Personalized_Brain_Functional_BrainNetwork_Construction_Tool')
from __init__ import PersonalizedBrainNetworkTool

# Initialize the tool
tool = PersonalizedBrainNetworkTool()

# List all available atlases
print("Available brain atlases:")
for atlas in tool.list_atlases():
    print(f"  - {atlas}")

# Load AD atlas
ad_atlas = tool.load_atlas('AD')
print(f"AD atlas shape: {ad_atlas.shape}")

# Generate test data
import numpy as np
fmri_data = np.random.randn(200, 100)

# Calculate functional connectivity matrix
correlation_matrix = tool.estimate_correlation(fmri_data)
print(f"Correlation matrix shape: {correlation_matrix.shape}")

# Visualization
tool.visualize_functional_network(correlation_matrix, "AD Functional Network", "ad_functional_network.png")
```

### Batch Processing

```python
# Batch process multiple tasks
for task in ['AD', 'ASD', 'MDD']:
    atlas = tool.load_atlas(task)
    fmri_data = np.random.randn(200, 100)
    correlation_matrix = tool.estimate_correlation(fmri_data)
    output_path = f"{task.lower()}_functional_network.png"
    tool.visualize_functional_network(correlation_matrix, f"{task} Functional Network", output_path)
```

### Advanced Visualization

```python
# Visualize brain atlas
ad_atlas = tool.load_atlas('AD')
tool.visualize_atlas(ad_atlas, "AD Brain Atlas", "ad_atlas_visualization.png")

# Visualize network statistics
correlation_matrix = tool.estimate_correlation(fmri_data)
tool.plot_network_statistics(correlation_matrix, "Network Statistics", "network_statistics.png")

# Visualize connectivity histogram
tool.plot_connectivity_histogram(correlation_matrix, "Connectivity Histogram", "connectivity_histogram.png")
```

## Dependencies
The tool requires the following dependencies:
- **Core libraries**: numpy, pandas, scipy, scikit-learn
- **Deep learning**: torch, torchvision
- **Neuroimaging**: nibabel, nilearn
- **Visualization**: matplotlib, seaborn
- **Model related**: timm, tensorboardX
- **Other**: PyYAML, tqdm, Pillow
All dependencies are listed in `requirements.txt` and will be installed automatically when using pip.
