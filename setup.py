import setuptools

setuptools.setup(
    name="shap_bootstrap",  # Replace with your own username
    version="0.0.5",
    author="Koralp Catalsakal",
    author_email="mrkoralp@gmail.com",
    description="Software package for implementing shap-bootstrapping",
    long_description="This package consists of the software implementation of my thesis. \
    When imported, the top module name is `Framework`",
    long_description_content_type="text/markdown",
    url="https://github.com/koralpc/Shapley-Clustering",
    packages=setuptools.find_packages(),
    install_requires=['scipy','pandas','openml','xgboost','scikit-learn','seaborn','shap','matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
