# NCSAC

## Requirements
To run the experiment, you need to set up the environment. The requirements are as follows:

1. Python version: Ensure that you are using Python 3.7 or later.
2. Create a virtual environment (optional but recommended): To avoid conflicts with other packages, it's a good practice to create a virtual environment for the project. To do so:
```
conda create -n your_env_name python=3.7
```
3. Then activate the environment:
```
source activate your_env_name
```
4. Install **Pytorch** with version 1.13.0 or later, and install **torch-geometric** with version 2.2.0.

## Run
Execute the `main.py` file
```
python main.py
```

## Modify hyperparameters
In `main.py`, we have provided an example using the **Cora** dataset. You can directly modify the parameters within the code or adjust them via command-line arguments, like this:
```
python main.py --dataset 'cora'  --model_dir 'ckpts'  --train_size 7  --valid_size 0  --seed 42  --with_feature True  --device "cuda:0"  --alpha 0.2  --k 3  --encoder_lr 0.001  --encoder_epochs 100  --triplet_samples 1000  --encoder_batch_size 64  --refiner_epochs 10  --refiner_episode 2000 --gamma 0.99  --lmbda 0.95 --eps 0.2 --policy_lr 0.0001  --value_lr 0.001  --max_step 15  --max_append_step 10  --max_remove_step 10  --test False
```

It is important to note that, in certain communities, the action space may be relatively small, which could result in the complete exploration of all possible actions. This can lead to the occurrence of gradient explosion during training, causing the output to become NaN. In such cases, it is recommended to reduce the learning rate or decrease the number of training episodes to mitigate these issues.

## Train example
```
= = = = = = = = = = = = = = = = = = = = 
Start loading CORA dataset...
#Nodes 2708, #Edges 10556, #Communities 7, #Avg_comms 386.86, #Avg_degree 3.8980797636632203
#Attributes 1434

Start spliting communities...
#Train 7, #Validation 0, #Test 0

Start extracting preliminary community...
time= 0.12613487243652344

Start training gnn encoder...
Epoch 1/100: 100%|██████████████████████████████████████| 15/15 [00:00<00:00, 19.08it/s, loss=0.351]
...
Epoch 100/100: 100%|██████████████████████████████████| 15/15 [00:00<00:00, 174.68it/s, loss=0.0618]

Start training appendNet...
Iteration 1: 100%|█| 100/100 [03:26<00:00,  2.07s/it, episode=100, return=-0.656
Save net in ckpts/cora/com_append_episode100.pt
...
Iteration 20: 100%|█| 100/100 [03:08<00:00,  1.88s/it, episode=2000, return=2.55
Save net in ckpts/cora/com_append.pt

Iteration 1: 100%|█| 100/100 [03:01<00:00,  1.81s/it, episode=100, return=-1.126
...
Iteration 20: 100%|█| 100/100 [02:49<00:00,  1.70s/it, episode=2000, return=2.80
Save net in ckpts/cora/com_remove.pt
```

## Test example
```
...
Load net from ckpts/cora/com_append.pt
Load net from ckpts/cora/com_remove.pt
f1-score:0.48273929318123382, nmi:0.23844738189382337, jac:0.35293847282377328
```

You can modify the testing model within the `refine` function in `refiner.py`, such as:
```
...
Load net from ckpts/cora/com_append_episode800.pt
Load net from ckpts/cora/com_remove_episode1300.pt
f1-score:0.44448373984461076, nmi:0.17241426357349204, jac:0.29133076241133626
```

