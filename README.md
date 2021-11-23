# MS-SincResNet


Link: https://dl.acm.org/doi/abs/10.1145/3460426.3463619 </br>

### About
This repo inherits the implementation of the MGC model proposed in a recent paper - MS-SincResNet - https://arxiv.org/abs/2109.08910. We are able to acheive a better performance of 91.50% than the original paper. We propose various augmentation strategies and combination of center and label smoothing losses to achieve better test accuracy. Additionally, we have also extended our model on Extended Ballroom Dataset and FMA dataset and are having competitive results with other SOTA methods. </br>  

### Setup 

1. Make sure you have python 3.6 and above
2. Run below command to install dependencies </br>
   pip install -r requirements.txt

Running on single GPU:
```bash
python Main.py
```

Running on two GPUs:
```bash
bash run.sh
```

### Currently support GTZAN and Extended Ballroom and FMA-small.
- GTZAN will be auto downloaded
- To get Extended Ballroom, please first run 
    ```bash
    python preprocess/script/getEBallroom.py
    ```
- To get FMA_small please run
    ```bash
    cd _data/FMA_small
    wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
    wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
    python make_label_FMA_small.py
    ```
### Known issue
Console logging is messy when running multiple GPUs, temporarily unavailable.
