from setuptools import setup, find_packages

setup(
    name="flux",
    version="1.0.0",
    description="FLUX: File-based Lightweight Universal Xplainable Dataset Versioning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["pandas", "numpy"],
    entry_points={
        "console_scripts": [
            "flux=flux.cli.main:main",
        ],
    },
)
