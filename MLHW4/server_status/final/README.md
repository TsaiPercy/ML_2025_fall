# ML HW4

## Environment setting
- Python 3.12
- CUDA 11.8

```
1. conda create -n MLHW4 python=3.12 -y
2. conda activate MLHW4

3. pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
4. pip install -r requirements.txt

```
## Structure
```
ML_HW4/
├── data/
│   ├── test/
│   ├── train/
│   └── unlabeled/
│
├── model_weight/
│   ├── b5_pseudo.pth
│   ├── b5.pth
│   ├── b6.pth
│   └── best_single_b6.pth
│
├── src/
│   ├── func.py
│   ├── inference_ensemble.py
│   ├── inference.py
│   └── training.py
│
├── submission/
│   ├── ensemble_result.csv
│   └── single_result.csv
│
├── 112550183_weight.txt
├── README.md
└── requirements.txt

```
- please ensure data and model weight exist, and put on the correct position

## Training

```
python src/training.py
```

## Inference

* single best model
```
python src/inference.py
```

* ensemble best model
```
python src/inference_ensemble.py
```



 