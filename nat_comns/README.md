## Submission to Nature communications : "Detecting and Distinguishing Tipping Points using Spectral Early Warning Signals"

This repository provides instructions and code to reproduce the quantitative results in the paper submitted to the journal *Nature Communications* titled "Detecting and Distinguishing Tipping Points using Spectral Early Warning Signals".

### script_ricker_fold.py
This script simulates a trajectory of the Ricker model going through the *Fold* bifurcation and computes early warning signals, as in Figure 2 of the paper. Run time is long (approximately 30 minutes), since in the paper we use the bootstrapping method and nonlinear optimisation to fit the power spectra and determine the AIC weights. For a quick-run demo of the same model and bifurcation, please refer to the [demos section](../demos).

### script_ricker_flip.py
This scipt simulates a trajectory of the Ricker model going throught the *Flip* bifurcation and computes early warning signals, as in Figure 3 of the paper. Run time is again long, although there is again a quick-run demo of the same model and bifurcation in the [demos section](../demos).

### script_data_analysis.py
This script computes early warning signals within the empirical dataset, as shown in Figure 4 of the paper. This data is not public however, so we only provide the code for reference.


