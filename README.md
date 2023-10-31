# WaferCap
# VTS2024 : WaferCap: Open Classification of Wafer Map Patterns using Deep Capsule Network

## Envrionment

* Python 3.10.12
* Numpy 1.23.5
* Pandas 1.5.3
* Torch 2.1.0+cu118
* Sklearn 1.2.2
* scipy 1.11.3
* tqdm 4.66.1
* opencv-python 4.8.0
* GPU : Tesla T4


## To run


### To install the necessary modules

```
pip install -r requirements.txt
```

### To download WM-811k dataset, [click here][1]

### To generate the results

### CLOSED-WORLD SETTING

#### Split ratio = 8:1:1, Data Augmentation set to True, with 100% seen classes
```
python main.py --epoch 10 --augment 1 --percentSeenCls 100 --splitRatio '8:1:1'
```

#### Split ratio = 8:1:1, Data Augmentation set to False, with 100% seen classes
```
python main.py --epoch 7 --augment 0 --percentSeenCls 100 --splitRatio '8:1:1'
```

#### Split ratio = 7:3, Data Augmentation set to True, with 100% seen classes
```
python main.py --epoch 14 --augment 1 --percentSeenCls 100 --splitRatio '7:3'
```

#### Split ratio = 7:3, Data Augmentation set to False, with 100% seen classes
```
python main.py --epoch 8 --augment 0 --percentSeenCls 100 --splitRatio '7:3'
```

#### Split ratio = 8:2, Data Augmentation set to True, with 100% seen classes
```
python main.py --epoch 10 --augment 1 --percentSeenCls 100 --splitRatio '8:2'
```

#### Split ratio = 8:2, Data Augmentation set to False, with 100% seen classes
```
python main.py --epoch 9 --augment 0 --percentSeenCls 100 --splitRatio '8:2'
```



### OPEN-WORLD SETTING


#### Split ratio = 8:2, Data Augmentation set to True, with 25% seen classes
```
python main.py --epoch 4 --augment 1 --percentSeenCls 25 --splitRatio '8:2' 
```

#### Split ratio = 8:2, Data Augmentation set to True, with 50% seen classes
```
python main.py --epoch 5 --augment 1 --percentSeenCls 50 --splitRatio '8:2'
```

#### Split ratio = 8:2, Data Augmentation set to True, with 75% seen classes
```
python main.py --epoch 15 --augment 1 --percentSeenCls 75 --splitRatio '8:2'
```





[1]: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
















