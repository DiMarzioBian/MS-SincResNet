# MS-SincResNet


Link: https://dl.acm.org/doi/abs/10.1145/3460426.3463619

### This is a pytorch replicate of MS-SincResNet

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