from setuptools import setup, find_packages

setup(
    name="nlpbench",
    version="1.0.0",
    description="Hugging Face Dataset Quality Measurement Tool using Great Expectations",
    author="NLPBench Team",
    packages=find_packages(),
    install_requires=[
        "datasets>=2.14.0",
        "great-expectations>=0.18.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "jinja2>=3.1.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "pyarrow>=12.0.0",
        "PyYAML>=6.0.0",
        "pytest>=7.0.0",
    ],
    entry_points={
        "console_scripts": [
            "nlpbench=src.main:cli",
        ],
    },
    python_requires=">=3.8",
)