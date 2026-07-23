# RaShipWakeDet
The required environment for running can refer to “requirements. txt”, and the installation of the required environment can be completed by entering the following command in the terminal: “pip install -r requirements.txt”

The data is located in the “aug10” folder. After completing the environment configuration, you can test it by running get_map.py.

Setting `USE_KFOLD = True` in `get_map.py` yields the results of 5-fold cross-validation, while setting `USE_KFOLD = False` yields the results for fold 1.

The `logs` folder can be obtained via either of the following links:
Download Link 1: https://pan.baidu.com/s/1L0uPELZhNwejuKBUQdyIRA?pwd=wrh2 (Extraction code: wrh2)
Download Link 2: https://huggingface.co/wwjwork/RashipwakeDet_logs

The path and folder structure are shown below：
RaShipWakeDet/
    ├── detector.py
    ├── get_map1.py
    ├── val_fold1.txt
    ├── val_fold2.txt
    ├── val_fold3.txt
    ├── val_fold4.txt
    ├── val_fold5.txt
    ├── logs/
    │   ├── fold_1/
    │   ├── fold_2/
    │   ├── fold_3/
    │   ├── fold_4/
    │   └── fold_5/
    ├── aug10/
    ├── model_data/
    ├── nets/
    └── utils/
