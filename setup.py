"""
Setup configuration for In-Context Representation Influence package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="icl-structural-influence",
    version="0.1.0",
    author="Research Team",
    description="Research on how in-context examples influence geometric structures in LLM representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/icl-structural-influence",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0,<3.0.0",
        "transformers>=4.36.0,<5.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        "networkx>=3.0,<4.0",
        "matplotlib>=3.7.0,<4.0.0",
        "tqdm>=4.65.0,<5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
        ],
        "full": [
            "wandb>=0.15.0,<1.0.0",
            "scikit-learn>=1.3.0,<2.0.0",
            "imageio>=2.9.0,<3.0.0",
            "imageio-ffmpeg",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="llm, in-context-learning, mechanistic-interpretability, transformers",
)
