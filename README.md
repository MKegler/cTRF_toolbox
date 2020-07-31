## cTRF_toolbox v0.2

Mikolaj Kegler

Contact: mikolaj.kegler16@imperial.ac.uk

Imperial College London, 31st July 2020

Example use of python implementation of the **c**omplex **TRF** modelling (**cTRF**) toolbox developed and applied in [Etard, et al., 2019](https://doi.org/10.1016/j.neuroimage.2019.06.029).

To download the sample data that can be processed with this script please follow [this link](https://imperialcollegelondon.box.com/s/k3vt564g56tulf6aty55yeqlqvazvgfl). The sample EEG and fundamental waveform data are the same as used in the paper and were obtained as described in the methods section.

Code extracting fundamental waveforms from coninuous speech (originally introduced in [Forte, et al., 2017](https://doi.org/10.7554/eLife.27203)) is available [here](https://github.com/MKegler/fundamental_waveforms_extraction). This is the most up-to-date version with a several minor bugs fixed.

To run Demo.ipynb with example use of the package you will need:
- MNE https://mne.tools/stable/index.html
- NumPy http://www.numpy.org/
- SciPy https://www.scipy.org/
- Matplotlib https://matplotlib.org/
- cTRF custom package (cTRF.py attached)

NOTE: The sample data are high-sampled and therefore fitting the complex backward models might require extensive amounts of RAM and might take a long time to compute. It is highly recommended to run this code on a high-performance machine. On a machine with ~8 GB RAM, we recommend running the complex forward models that are significantly 'lighter' in terms of the number of parameters and required computational power.

References:
- Etard, O., Kegler, M., Braiman, C., Forte, A. E., & Reichenbach, T. (2019). Decoding of selective attention to continuous speech from the human auditory brainstem response. NeuroImage. https://doi.org/10.1016/j.neuroimage.2019.06.029
- Forte, A. E., Etard, O., & Reichenbach, T. (2017). The human auditory brainstem response to running speech reveals a subcortical mechanism for selective attention. Elife. https://doi.org/10.7554/eLife.27203


