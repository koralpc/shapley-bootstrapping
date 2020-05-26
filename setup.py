import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shap_bootstrap",  # Replace with your own username
    version="0.0.6",
    author="Koralp Catalsakal",
    author_email="mrkoralp@gmail.com",
    description="Software package for implementing shap-bootstrapping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/koralpc/Shapley-Clustering",
    packages=setuptools.find_packages(),
    install_requires=[
        "scipy",
        "pandas",
        "openml",
        "xgboost",
        "scikit-learn",
        "seaborn",
        "shap",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
