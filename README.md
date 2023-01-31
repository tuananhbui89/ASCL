# Adversarial-Supervised-Contrastive-Learning (ASCL)

This is Pytorch version for the paper ["Understanding and Achieving Efficient Robustness with Adversarial Supervised Contrastive Learning"](https://arxiv.org/abs/2101.10027). 

In this implementation, we also provide baseline methods
- (1) PGD-AT: PGD adversarial training, Madry et al. 2017. <br>
- (2) TRADES: Theoretically principled trade-off between robustness and accuracy. Zhang et al. 2019. <br>
- (3) ADR: Adversarial Divergence Reduction, Bui et al. 2020.

## Requirements 
- Python == 3.7
- Auto-Attack == 0.1
- Foolbox == 3.2.1
- Numba == 0.52.0
- Pytorch >= 1.7.0

## Robustness Evaluation 
We use several attackers to challenge the baselines and our method. 

(1) PGD Attack. We use the pytorch version of the Cifar10 Challenge with norm Linf, from the [implementation](https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py). Attack setting for the Cifar10 dataset: `epsilon`=8/255, `step_size`=2/255, `num_steps`=200, `random_init`=True 

(2) Auto-Attack. The official implementation in the [link](https://github.com/fra31/auto-attack). We test with the standard version with Linf

(3) Brendel & Bethge Attack. The B&B attack is adapted from [the implementation](https://github.com/wielandbrendel/adaptive_attacks_paper/tree/master/07_ensemble_diversity) of the paper ["On Adaptive Attacks to Adversarial Example Defenses"](https://arxiv.org/abs/2002.08347). It has been initialized with PGD attack (20 steps, `eta`=`epsilon`/2) to increase the success rate.  

We use the full test-set (10k images) for the attack (1) and 1000 first test images for the attacks (2-3).

## Training and Evaluation 

We provide the default setting for each corresponding dataset (MNIST, CIFAR10) which is used in our paper.

To reproduce the baselines, run the following script. 
```shell
python run_baseline.py
```

To reproduce our results, run the following script. 
```shell
python run_cifar10.py
```

Please refer to the file `mysetting.py` for custom running. The pretrained model will be published soon. 

### Architectures 
We use two popular architectures in the literature, which are (1) ResNet18 and (2) WideResNet-34-10. The model architecture can be chosen by parameter `model` in `mysetting.py`. 

We modified `forward` function in each architecture class to output latent vectors (intermediate layer) as long with logits vectors. The second last layer (before the last dense layer) has been chosen as the intermediate layer to apply the regularization. It is worth noting that, in [FeatureScattering](https://arxiv.org/abs/1907.10764) paper, the last layer has been chosen (logits vector), which might lead to a better result.

### Training setting 
We choose the setting as in Pang et al. 2020. More specifically: 
- optimizer: SGD with momentum 0.9, weight decay 5e-4
- learning rate scheduler: init with 0.1, learning rate decay with rate 0.1 at epoch {100, 105}
- training length: 110 epochs (we extend to 120 epochs)

Other important settings to obtain the result: 
- We init PGD attack with uniform noise (-epsilon, epsilon)
- In the original implementation, the model has been changed to evaluation stage (to change BN to evaluation) when crafting adversarial examples. We follow the advice from Pang et al. 2020 to use training stage when crafting adverasarial examples (in either pgd-at or trades-loss) 
- In PGD-AT, we minimize the robust_loss (cross_entropy(adv_output,y)) (and omit the natural_loss) when training model. It helps to increase the robust accuracy but decrease the natural accuracy 

| PGD-AT                   | Arch             | Nat   | PGD200 | AA   |
|--------------------------|------------------|-------|--------|------|
| robus_loss               | ResNet18         | 82.75 | 52.93  | 48.8 |
| robust_loss+natural_loss | ResNet18         | 87.22 | 48.67  | 44.1 |
| robust_loss              | WideResNet-34-10 | xx.xx | xx.xx  | xx.x |
| robust_loss+natural_loss | WideResNet-34-10 | xx.xx | xx.xx  | xx.x |

### Dataset preprocessing 
There are two common preprocessing methods in literature. 
- the standard normalization where the input has been normalized to the range [0, 1] by dividing (255.) as in Madry et al. 2017 and TRADES paper. 
- the input has been normalized with mean and sigma of the entire training set. Default setting for CIFAR10 are mean=(0.4914, 0.4822, 0.4465) and std=(0.2471, 0.2435, 0.2616), while these for CIFAR100 are mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343) and std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404). 

However, it should be very careful when applying normalization rather than standard one (i.g., normalizing to (0,1)) and be aware of the clamp function with the lower_limit=0.0 and upper_limit=1.0. The incorrect implementation can lead to an improper performance.

The correct pipeline should be: 
- train_loader and test_loader output range (0, 1)
- normalize(clamp(X, lower_limit, upper_limit), mu, std)

In this case, attackers (e.g., pgd attack) should have the normalize function before feeding to model (refer to AWP's implementation in [link] (https://github.com/csdongxian/AWP/blob/main/AT_AWP/train_cifar10.py)) i.e., model(normalize(x))

Another approach is using adding normalization layer in the beginning of the model. (refer to [link]()), i.e., model = nn.Sequential(Normalize(mean=mean, std=std), net). In this case, we no need to worry about normalizing input in attackers. 

<!-- We have tried with two normalizing methods and got similar performances.  -->
In this implementation, we use the second simple preprocessing. 
Detail can be found in `dataset.py` and `02a_adversarial_training.py` 

### Projection head
We provide two options with the projection head: 
- if `feat_dim` less than 0, then we do not use any projection head. The regularization now is applied directly to the intermediate layer. 
- if `feat_dim` greater than 0, then we use a projection head with two MLP layers (dim_in, dim_in) --> (dim_in, feat_dim). The regularization now is applied to the projection layer. 

Intuitively, while using projection head can help to reduce the dimensionality (to apply the regurlaization on), thus improve the natural performance. However, in AML, the robustness is weaker when projecting back from projection layer to intermediate layer. 

We also do not use normalization function inside the projection head (as in [SupContrast implementation](https://github.com/HobbitLong/SupContrast)). We instead manually normalize the latent vector with the parameter `hidden_norm` in the `soft_lcscl` function. This helps us have more freedom in choosing (i) projection head or not (ii) using normalization or not. 

### Latent distance 
We provide several distance functions for two latent vectors (z1, z2): 
- L1: d = norm(z1-z2, l1) = sum(abs(z1-z2)). Requires `hidden_norm=True`
- L2: d = norm(z1-z2, l2) = sum(square(z1-z2)). Requires `hidden_norm=True`
- Linf: d = norm(z1-z2, inf) = max(abs(z1-z2))
- Cosine: d = 1 - cosine_similarity(z1, z2). Requires `hidden_norm=False`
- Matmul: d = -matmul(z1, z2.T). Requires `hidden_norm=True`

### Data Augmentation and Multiple Adversarial examples 

### Representation Learning 
In this version, we provide two options to apply the ASCL. First, we apply the ASCL directly to the Supervised Adversarial Training. In this case, ASCL can be understand as a regurlarization. Second, we provide a separate pre-training process which learn a robust representation with ASCL. This follow with the Supervised Adversarial Training phase on (i) entire model (encoder + linear classifier) (ii) or just fine-tune with linear classifier.

For the pre-training phase, our implementation is based on the SupConTrast implementation.  

## References
- The B&B attack is adapted from [the implementation](https://github.com/wielandbrendel/adaptive_attacks_paper/tree/master/07_ensemble_diversity) of the paper ["On Adaptive Attacks to Adversarial Example Defenses"](https://arxiv.org/abs/2002.08347). 
- The Auto-attack is adapted from [the implementation](https://github.com/fra31/auto-attack) of the paper ["Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks", Francesco Croce, Matthias Hein, ICML 2020](https://arxiv.org/abs/2003.01690).

## Citation 
If you find this implementation useful for your research, please consider to cite our paper 

```
    @article{bui2021understanding,
    title={Understanding and achieving efficient robustness with adversarial supervised contrastive learning},
    author={Bui, Anh and Le, Trung and Zhao, He and Montague, Paul and Camtepe, Seyit and Phung, Dinh},
    journal={arXiv preprint arXiv:2101.10027},
    year={2021}
    }
```