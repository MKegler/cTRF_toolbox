## cTRF_toolbox v0.2

Mikolaj Kegler

Contact: mikolaj.kegler16@imperial.ac.uk

Imperial College London, 27.10.2019

Example use of python implementation of the complex TRF modelling (cTRF) toolbox developed and applied in [Etard, et. al., 2019].

To download the sample data that can be processed with this script please use the following link: https://imperiallondon-my.sharepoint.com/:f:/g/personal/mak616_ic_ac_uk/En-V7CKltGdPqYKzd5xJ6a0BljqepRaHF-m9ta__LdtYhA?e=NgMXUv. The sample EEG and fundamental waveform data are the same as used in the paper and were obtained as described in the methods section.

Required packages to run this demo:
- MNE https://martinos.org/mne/dev/index.html
- NumPy http://www.numpy.org/
- SciPy https://www.scipy.org/
- Matplotlib https://matplotlib.org/
- cTRF custom package (cTRF.py attached)

NOTE: The sample data are high-sampled and therefore fitting the complex backward models might require extensive amounts of RAM and might take a long time to compute. It is highly recommended to run this code on a high-performance machine. On a machine with ~8 GB RAM, we recommend running the complex forward models that are significantly 'lighter' in terms of the number of parameters and required computational power.

Reference: Etard, O., Kegler, M., Braiman, C., Forte, A. E., & Reichenbach, T. (2019). Decoding of selective attention to continuous speech from the human auditory brainstem response. NeuroImage. https://doi.org/10.1016/j.neuroimage.2019.06.029
