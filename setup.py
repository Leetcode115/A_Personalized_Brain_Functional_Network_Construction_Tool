from setuptools import setup, find_packages

setup(
    name="personalized_brain_functional_network_construction_toolbox",  # 包名
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "nilearn",
        "nibabel",
        "pytorch"
    ],
)
