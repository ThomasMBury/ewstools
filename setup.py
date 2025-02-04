import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "pandas>=0.23.0",
    "numpy>=1.14.0",
    "plotly>=2.3.0",
    "lmfit>=0.9.0",
    "arch>=4.4",
    "statsmodels>=0.9.0",
    "scipy>=1.0.1",
    "deprecation>=2.0",
    "entropyhub>=2.0",
]

setuptools.setup(
    name="ewstools",
    version="2.1.2",
    author="Thomas M Bury",
    author_email="tombury182@gmail.com",
    description="""Python package to compute early warning signals (EWS) 
    from time series data""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasMBury/ewstools",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
