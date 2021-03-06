# Spectra analysis and simulation
Algorithms ans data being used for meteoric UV and visible spectra analysis and simulation are collected.
* The script `mm_tester.py` covers the drafts of ablation spectra analysis. The computations used allow us to get the approximation of the free electron temperature, the electron density and their relation to the spectral lines.
Moreover, a simpe model calculation of the plasma distribution is given. The parameters were set for a hypothetic meteoric impact characterized by the same physical properties reaching the atmosphere. The file of `input_one.txt`, the only input data insted of the ablation spectra, needs to be downloaded with the script.

## Extra data processing and documentation files
The extra data mining files with own databases, as well as the README files and documentation written elsewhere is stored here.
* The documentation `mm_tester_readme.pdf` describing the calculations used in ablation spectra physical description draft have been uploaded.
* The file of `mm_spectras.py` covers an extended version of the previous computations and script drafts. Except the basic plasma parameters mentioned, it allows us to approach the concentration or relative amount of concrete elements. The statistical database is also invoved. Except the file `https://github.com/Spektras/Spektras/blob/master/Spectra%20analysis%20and%20simulation/input_one.txt.`, the script must be run with the database `data_200_800.txt` downloaded. In both database and the resultant data files, the concentrations and their statistical relations are processed in the orther of C - Na- Mg - Al - Si - P - S - Cl - K - Ca - Ti - V - Cr - Mn - Fe - Co - Ni - Cu - Zn - Se - Sr - Mo - Cd - Pb - U - Be. 
