from setuptools import setup, find_packages
import os

VERSION = '1.0.4'
DESCRIPTION = 'Statistical Methods for Data Science Toolkit'

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Setting up
setup(
    name="statstoolkit",
    version=VERSION,
    author="Eng. Marco Schivo and Eng. Alberto Biscalchin",
    author_email="<biscalchin.mau.se@gmail.com>, <marcoschivo1@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'pingouin',
    ],
    keywords=[
        "statistics", "data science", "machine learning", "data analysis",
        "descriptive statistics", "probability distributions", "visualizations"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ],
)
