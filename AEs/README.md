# Auxiliary Experiments

This folder contains the auxiliary experiments. To reference these experiments in the thesis, refer to the appendix section in the thesis document.

- **ADs**: Contains the distribution of samples per class per region for each of the auxiliary datasets.
- **AE-1**: Contains a notebook used to explore thin parcels and informed the decision to remove parcels with an area-to-perimeter ratio below 6. It also includes the summary results of AD-1.
- **AE-2**: Contains the summary results of AD-2, AD-3, and AD-4. Used to inform sample caps for the creation of experimental datasets.
- **AE-3**: Contains the train log of 1-baseline, a re-run to confirm that stochasticity was controlled using a seed in the experiments.
- **AE-4**: Contains the notebook used to inform the selection of parcel area as the sensitive attribute for targeted fairness intervention. Note that it uses the result of 0-baseline, and this dataset is referred to as AD-5 in the thesis.
- **AE-5**: Contains the notebook used to run the piecewise linear regression to determine a threshold for differentiating small and large parcels.
