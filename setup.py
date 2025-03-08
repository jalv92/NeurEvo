from setuptools import setup, find_packages

setup(
    name="neurevo",
    version="0.1.0",
    description="Framework de aprendizaje por refuerzo con elementos cognitivos y evolutivos",
    author="NeurEvo Team",
    author_email="info@neurevo.org",
    url="https://github.com/neurevo/neurevo",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "psutil>=5.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 