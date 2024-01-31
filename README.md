# ipu_unet_ts
Using the ipu-based camelyon 16 data set, we reproduced the event in which loss did not decrease when training the unet model.   

## Environment setup

NOTE: You must install PopTorch to your environment (as in 2. below) before installing requirements (as in 5. below), else you may encounter compatibility issues with Horovod

To prepare your environment, follow these steps:

1. Create and activate a fresh Python3 virtual environment:
```bash
python3 -m venv <venv name> --clear
source ./<venv name>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the PopTorch (PyTorch) wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch...x86_64.whl
```   

4. Install requirement package
```bash
pip install -r requirements.txt
```   

## dataset
The dataset is used by downloading files stored in a shared Google Cloud.   
```bash
./dataset_download.sh
```

## train
Set the downloaded dataset path in the ```ipu_train.py``` code.
```bash
python ipu_train.py
```

