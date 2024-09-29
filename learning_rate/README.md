# Whats Learning Rate?

Learning Rate is a crtical hyperpameter when it comes training deep learning models. Adaptive learning rates can help the parameters to converge faster to local minima optimal values.

##### PyTorch's optim.lr_scheduler
In PyTorch, optim.lr_scheduler is a module that provides various strategies for adjusting the learning rate during training. These learning rate schedulers allow you to change the learning rate of the optimizer at specific intervals, which can help the model converge better and faster, especially during the later stages of training.

> Note: Why Use lr_scheduler?

	•	Avoid Overshooting: High learning rates may cause the optimizer to overshoot the minimum, making it harder to converge. Reducing the learning rate can help the optimizer stabilize.
	•	Speed Up Convergence: Start with a higher learning rate to converge faster in the early stages and then reduce it as the model approaches the optimal solution.
	•	Fine-tuning: In some cases, after initial training, it’s beneficial to lower the learning rate to fine-tune the model’s performance.
 
#### Commonly Used Learning Rate Schedulers in PyTorch
1.	StepLR: Decreases the learning rate by a factor (gamma) every few epochs (based on a step size).
   
 ```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

 	•	step_size: Number of epochs after which the learning rate is decayed.
	•	gamma: Multiplicative factor for learning rate decay.

2.	MultiStepLR: Decreases the learning rate at specific epochs (milestones).
   
```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
```

3.	ExponentialLR: Decays the learning rate exponentially by a factor of gamma at every epoch.

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

 	•	gamma: Multiplicative factor for learning rate decay every epoch.

4.	CosineAnnealingLR: Decreases the learning rate following a cosine curve.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```
	•	T_max: Maximum number of iterations for which the learning rate decays following the cosine schedule.

5.	ReduceLROnPlateau: Reduces the learning rate when a metric has stopped improving (e.g., validation loss).
   
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
```
	•	mode: Defines whether the metric is minimized (‘min’) or maximized (‘max’).
	•	patience: Number of epochs with no improvement after which learning rate is reduced.
	•	factor: Factor by which the learning rate is reduced.
	•	threshold: Minimum change in the metric to qualify as an improvement.
 
6.	LambdaLR: Custom learning rate scheduler that applies a lambda function to adjust the learning rate.


```python
lambda1 = lambda epoch: 0.95 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
```
	•	lr_lambda: A function that defines how the learning rate changes with each epoch.

### Examples with Simple Linear Regression Model

```python
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

#### Generate some data

```python
n=1000
p=5

x=torch.randn(n, p).to(device)
w_true=torch.tensor([i/0.5 for i in range(p)], dtype=torch.float, device=device)

noise_mean=0
noise_std=0.2

y=x@w_true+ noise_std*torch.randn(n, dtype=torch.float, device=device) + noise_mean
y=y.reshape(-1, 1).to(device)

x.shape, y.shape
```
#### Train Model 

```python
def train_model(scheduler=False):
    
    torch.manual_seed(42)
    lr=0.25
    model = nn.Sequential(nn.Linear(p, 1)).to(device)
    loss_fn=nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)
    if scheduler=='StepLR':
        schdler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    if scheduler=='CosineAnnealingLR':
        schdler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    if scheduler=='ExponentialLR':
        schdler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if scheduler=='ReduceLROnPlateau':
        schdler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    
    n_epoch=100

    for epoch in range(n_epoch):
        model.train()
        loss=loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        optimizer.step() 
        if scheduler=='ReduceLROnPlateau':
            schdler.step(loss)
        elif scheduler:
            schdler.step()
        else:
            continue
    print(f"\n\nEpoch :: {epoch+1} ")
    if scheduler:
        print(f"\nLearning rate is {schdler.get_last_lr()[0]}")
    print(f"\nWeights: {model.state_dict().values().mapping['0.weight'][0]}")
    print(f"\nBias:    {model.state_dict().values().mapping['0.bias'][0]}")
    print(f"\nLoss: {loss.item()}\n")
```

#### No Scheduler
```python
train_model(scheduler=False)
```
> Epoch :: 100 

Weights: tensor([4.0653e-03, 2.0031e+00, 3.9963e+00, 6.0017e+00, 7.9960e+00],
       device='mps:0')

Bias:    -0.0032501553650945425

Loss: 0.040374353528022766

#### StepLR
```python
train_model(scheduler=False)
```
Epoch :: 100 

Learning rate is 6.640349721896886e-06

Weights: tensor([2.6342e-04, 1.9987e+00, 3.9895e+00, 5.9875e+00, 7.9746e+00],
       device='mps:0')

Bias:    0.0056349434889853

Loss: 0.04118192568421364

#### CosineAnnealingLR
```python
train_model(scheduler=False)
```
Epoch :: 100 

Learning rate is 0.0

Weights: tensor([4.0653e-03, 2.0031e+00, 3.9963e+00, 6.0017e+00, 7.9960e+00],
       device='mps:0')

Bias:    -0.00325010041706264

Loss: 0.040374353528022766


#### ExponentialLR
```python
train_model(scheduler=False)
```
Epoch :: 100 

Learning rate is 0.0014801323050834992

Weights: tensor([4.0360e-03, 2.0031e+00, 3.9963e+00, 6.0017e+00, 7.9959e+00],
       device='mps:0')

Bias:    -0.0031858838628977537

Loss: 0.04037437215447426


#### #### ReduceLROnPlateau

```python
train_model(scheduler=False)
```
Epoch :: 100 

Learning rate is 2.5000000000000012e-08

Weights: tensor([4.0652e-03, 2.0031e+00, 3.9963e+00, 6.0017e+00, 7.9960e+00],
       device='mps:0')

Bias:    -0.0032497388310730457

Loss: 0.04037436097860336



