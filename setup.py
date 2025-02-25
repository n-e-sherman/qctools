from setuptools import setup, find_packages

setup(
    name="qctools",               # Package name
    version="0.1.0",              # Initial version
    packages=find_packages(),     # Automatically find all packages
    install_requires=[            # Dependencies
        "quimb",
        "torch",
        "qiskit",
        "numpy",
        "uuid",
        "mlflow"
    ],
    author="Nick Sherman",
    description="A collection of tools for studying quantum circuits with Quimb.",
    python_requires=">=3.11",      # Adjust Python version as needed
)
