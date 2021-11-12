# MS-SincResNet


Link: https://dl.acm.org/doi/abs/10.1145/3460426.3463619

## This is a replicate of MS-SincResNet

Running on single GPU:
```bash
python Main_one_gpu.py
```

Running on two GPUs:
```bash
bash run.sh
```

Currently support GTZAN and Extended Ballroom, FMA-small will added soon.

## Known issue
Console logging is messy when running multiple GPUs, I am optimizing the code.