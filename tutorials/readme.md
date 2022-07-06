# Tutorials
**A collection of iPython notebooks to demonstrate various applications of *ewstools*.**

To run and interact with these tutorials and demos, Jupyter notebook must be installed on your system, which can be found [here](https://jupyter.org/install).

I recommend starting with **tutorial_intro.ipynb**. Tutorials for other features are in progress.

### tutorial_intro.ipynb

- Introduction to the object-oriented version of *ewstools* and the TimeSeries class.
- Detrending using Gaussian and Lowess filters
- Computing CSD-based early warning signals
- Applying deep learning classifiers to predict a bifurcation and its type
- Visualsing output from *ewstools*.

### ews_fold.ipynb - Deprecated (uses deprecated functions in ewstools)
- Simulates a single stochastic trajectory of the Ricker model going through a Fold bifurcation
- Shows how to use *ewstools* to compute early warning signals
- Visualises the output of *ewstools* graphically
- Run time < 1 min


### ews_bootstrap.ipynb - Deprecated (uses deprecated functions ewstools)
- Simulates single stochastic trajectories of the Ricker model going through a Flip bifurcation
- Uses *ewstools* to compute bootstrapped time-series and the corresponding EWS.
- Visualises the output, comparing the two bifurcations.
- Run time < 3 mins

