"""
Setup configuration for Neuro CXR Analysis Engine.
Author: Debashis Mohapatra <debashismohapatra8260@gmail.com>
Year: 2026
"""

from setuptools import setup, find_packages

setup(
    name="neuro-cxr",
    version="2.0.0",
    author="Debashis Mohapatra",
    author_email="debashismohapatra8260@gmail.com",
    description="Multimodal Chest X-Ray Analysis with Explainable AI, Disease Localization, and Risk Scoring",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn>=0.29.0",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "ultralytics>=8.4.0",
        "opencv-python-headless>=4.9.0,<4.11.0",
        "Pillow>=10.3.0",
        "numpy>=1.26.0,<2.0.0",
        "python-multipart>=0.0.9",
        "matplotlib>=3.8.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Healthcare Industry",
    ],
)
