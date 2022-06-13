IFCNet
==============================
**Update 13/06/2022:** The code for our follow-up paper [SpaRSE-BIM: Classification of IFC-based geometry via sparse convolutional neural networks](https://doi.org/10.1016/j.aei.2022.101641) will be added soon.

Neural Network Models for the [IFCNet Dataset](https://ifcnet.e3d.rwth-aachen.de/). The paper can be found [here](https://arxiv.org/abs/2106.09712).

## Installation and Training

Make sure you have PyTorch installed. The code in this repository was developed and tested with:

* Python 3.8.5
* PyTorch 1.7.1+cu110
* Ubuntu 20.04

```bash
git clone https://github.com/cemunds/ifcnet-models.git
cd ifcnet-models
```

It is recommended to create a new virtual environment before installing the dependencies with the following command:

```bash
pip install -r requirements.txt
```

Download the data for the model you want to train and place them in the corresponding data folder:

* [MVCNN](https://ifcnet.e3d.rwth-aachen.de/static/IFCNetCorePng.7z)
* [DGCNN](https://ifcnet.e3d.rwth-aachen.de/static/IFCNetCorePly.7z)
* [MeshNet](https://ifcnet.e3d.rwth-aachen.de/static/IFCNetCoreNpz.7z)

For example, for MVCNN:
```bash
mkdir -p data/processed/MVCNN
cd data/processed/MVCNN
wget https://ifcnet.e3d.rwth-aachen.de/static/IFCNetCorePng.7z
7z x IFCNetCorePng.7z
```

Navigate back to the root directory of the project and execute the training script.
```bash
cd ../../..
python src/models/train_model.py MVCNN
```
Running the training script like this will perform a hyperparameter search using Ray Tune and Optuna. If you want to use a fixed set of hyperparameters, you can specify the path to a configuration JSON file with the `--config_file` flag. For the format of the configuration file, please refer to the files included with the pre-trained models.

*Warning:* Performing the hyperparameter search for MVCNN consumes a lot of disk space, as currently the model is saved to disk after every epoch and some of the backbone models (e.g. vgg11) are very large. I intend to change the saving behavior in the future.

## Evaluation

The pre-trained models can be downloaded here:
* [MVCNN](https://ifcnet.e3d.rwth-aachen.de/static/mvcnn_model.7z)
* [DGCNN](https://ifcnet.e3d.rwth-aachen.de/static/dgcnn_model.7z)
* [MeshNet](https://ifcnet.e3d.rwth-aachen.de/static/meshnet_model.7z)

Download and unzip the models into the `models` directory:

```bash
mkdir models
cd models
wget https://ifcnet.e3d.rwth-aachen.de/static/mvcnn_model.7z
7z x mvcnn_model.7z
```
Now you can start a Jupyter Notebook server and run the notebook for the corresponding model.

## Citation
If you use the IFCNet dataset or code please cite:
```
@inproceedings{emunds2021ifcnet,
  title={IFCNet: A Benchmark Dataset for IFC Entity Classification},
  author={Emunds, Christoph and Pauen, Nicolas and Richter, Veronika and Frisch, Jérôme and van Treeck, Christoph},
  booktitle = {Proceedings of the 28th International Workshop on Intelligent Computing in Engineering (EG-ICE)},
  year={2021},
  month={June},
  day={30}
}
```

## Acknowledgements
The code for the neural networks is based on the implementations of the original publications, but had to be updated for recent PyTorch versions:
* [WangYueFt](https://github.com/WangYueFt/dgcnn) for DGCNN
* [jongchyisu](https://github.com/jongchyisu/mvcnn_pytorch) for MVCNN
* [iMoonLab](https://github.com/iMoonLab/MeshNet) for MeshNet

Big thanks goes to [xeolabs](https://github.com/xeolabs) for the [xeokit SDK](https://github.com/xeokit/xeokit-sdk), which is used in the [viewer](https://ifcnet.e3d.rwth-aachen.de/).

The structure of this repository is loosely based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/)
