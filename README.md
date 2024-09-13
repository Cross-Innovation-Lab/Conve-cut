# Conv-cut: A Visual Facial Expression Signal Feature Processing Network Based on Streamlined Convnext

## Reqirements
```
# create conda env
conda create -n convenv python=3.8
conda activate convenv

# install packages
pip install tensorflow==2.10.0
pip install scikit-learn
pip install opencv-python

# install cuda-nvcc
conda install -c nvidia cuda-nvcc
```
## Preparing Datasets
The file structure of RAF-DB is as follows.
```
---RAF-DB\
    --train\
        --anger 
        --disgust
        --fear
        ...
    --test
        --anger 
        --disgust
        --fear
        ...
```
The file structure of FERPlus is as follows.
```
---FERPlus\
    --train\
        --anger 
        --disgust
        --fear
        ...
    --val\
        --anger 
        --disgust
        --fear
        ...
    --test
        --anger 
        --disgust
        --fear
        ...
```
Run the data.py file to get the data in h5 format
```
python data.py
```
## Train
Run the train.py file for training.
```
python train.py
```
## Evaluate
Run the evaluate.py file to evaluate. You can get the confusion matrix or T-sne
```
python evaluate.py
```
## Acknowledgement
We borrowed the code from [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [PAtt-Lite](https://github.com/JLREx/PAtt-Lite). Thanks for their wonderful works.