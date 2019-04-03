import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["lmfit>=0.9", "arch>=4.7"]

setuptools.setup(
    name="ewstools",
    version="0.0.3",
    author="Thomas M Bury",
    author_email="tombury182@gmail.com",
    description="""A package with tools to compute early warning signals (EWS) 
    from time-series data""",
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