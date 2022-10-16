# RESCVE
This is the repo for our manuscript "Representation of missense variants for predicting modes of action"

## Environment
We recommend using a conda environment to run the code. The environment can be created using the following command:
```
conda env create -f RESCVE.yml
```
then activate the environment using:
```
conda activate RESCVE
```

## Data
Please download AlphaFold predicted structures to `data/Protein/` directory and change the files in `data/Protein/uniprot.ID/` to your path.

The other data except HGMD data that we used in training process are provided under the `data/` folder.

For MSA, we provided the files here: https://drive.google.com/file/d/1iu32tVQZ_N9WYJ0a8Xeh25uLACZVsBI7/view?usp=sharing. Please download the file and extract it to data/MSA/

## Run
To run the code, please use the following command:
```
python RESCVE.py --mode Required --device Optional --seed Optional
```
Please check the comments in the file `RESCVE.py` for more details about which mode to use.
