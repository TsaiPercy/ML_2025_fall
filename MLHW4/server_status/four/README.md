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
│   └── best_model.pth
│
├── src/
│   ├── func.py
│   ├── inference.py
│   └── training.py
│
├── submission/
│   └── result.csv
│
├── requirements.txt
└── README.md
```
## Training
```
python src/training.py
```

## Inference

* single best model
```
python src/inference.py
```



 