from setuptools import setup, find_packages

setup(
    name="rossmann_sales_forecasting",  # Name of your package
    version="0.1.0",                    # Version number
    author="Olana Kenea",                 # Your name or team name
    author_email="olanakenea6@gmail.com",  # Your email
    description="A machine learning project to forecast Rossmann store sales.",  # Short description
    long_description=open("README.md").read(),  # Long description from README
    long_description_content_type="text/markdown",  # Specify Markdown format
    url="https://github.com/olanak/rossmann-sales-forecasting",  # Project URL (optional)
    packages=find_packages(),           # Automatically find all packages in `src/`
    install_requires=[                  # List of dependencies
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "joblib>=1.0.0",
        "flask>=2.0.0",
        "tensorflow>=2.6.0",  # For deep learning tasks
        "mlflow>=1.20.0",     # For experiment tracking
        "pytest>=6.0.0",      # For testing
    ],
    extras_require={                    # Optional dependencies
        "dev": [
            "black",                    # Code formatting
            "flake8",                   # Linting
            "pre-commit",               # Pre-commit hooks
        ],
    },
    classifiers=[                       # Metadata for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",            # Minimum Python version
)