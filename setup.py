import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['pandas>=1.2.0',
                'numpy>=1.20.0',
                'plotly>=5.3.0',
                'tensorflow>=2.0.0',
                'lmfit>=0.9', 
                'arch>=4.7',
                'statsmodels>=0.12.0',
                'scipy>=1.5.0',
                'sphinx>=5.0.0',
                ]

setuptools.setup(
    name="ewstools",
    version="1.0.1",
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