# ActiveLearning with UncertaintyIndicators in PyTorch

### Prerequisites:
- Linux or macOS
- Python 3.8
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.

### Experiments and Visualization
The code can simply be run using 
```
python3 main.py
```
If you want to use GPU
```
python3 main.py --cuda
```
When using the model with different datasets, the main hyperparameters to tune are
```
--dataset cifar100 --batch_size 68
```

The results will be saved in `results/accuracies.log`. 
