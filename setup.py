import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ewstools",
    version="0.0.1",
    author="Thomas M Bury",
    author_email="tbury@uwaterloo.ca",
    description="Compute early warning signals from time-series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasMBury/ewstools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)