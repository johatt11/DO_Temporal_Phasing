# DO_Temporal_Phasing
Code published in association with Slattery et al. (2023)

The code for the original implementation of the Bayesian ramp fitting method by Erhardt et al (2019, https://doi.org/10.5194/cp-15-811-2019)
is contained in the folder Original_Method.

The code for the extended implementation of this method, described in Slattery et al. (2024), is contained in the folder Extended_Method.

The notebook file Example_Synthetic_Transition.ipynb contains an example of how this code can be used, in this case for synthetic data.

The python file Figures.py allows one to reproduce the main figures in the article.

The python file Supplementary_Figures.py allows one to reproduce the additional figures in appendices of the article.

The python file Generating_Synthetic_Data.py gives examples of how we generate the synthetic transitions used 
to test for and estimate bias in the article, although we sadly cannot provide all of the >200,000 individual
synthetic transitions used for that analysis.

The folder Data contains the data required to produce all of these figures.

The full NGRIP data used in this study can be found at https://doi.org/10.1594/PANGAEA.896743 and were published alongside Erhardt et al. (2019)

The full CCSM4 data used in this study can be found at https://sid.erda.dk/cgi-sid/ls.py?share_id=Fo2F7YWBmv and were published alongside
Vettoretti et al. (2022, https://doi.org/10.1038/s41561-022-00920-7)

If using either of these data-sets, please cite the researchers who produced them.
If using the original implementation of the ramp fitting method, please cite Erhardt et al.
If using the extended implementation, please cite both Erhardt et al. and Slattery et al.

Required packages: numpy, scipy, pandas, proplot, emcee
