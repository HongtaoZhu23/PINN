# PINN4Bioprocess
This code is for our paper: [Physics-Constrained Latent States Reconciliation PINN Framework for Complex Bio-process Soft Sensing Problems]

![https://github.com/HongtaoZhu23/PINN/blob/main/Bioprocess.png](https://github.com/HongtaoZhu23/PINN/blob/main/Bioprocess.png)
![https://github.com/HongtaoZhu23/PINN/blob/main/Results.png](https://github.com/HongtaoZhu23/PINN/blob/main/Results.png)



Physical modeling is critical for many modern science and engineering applications. From a data science or machine learning perspective, where more domain-agnostic, data-driven models are pervasive, physical knowledge — often expressed as differential equations — is valuable in that it is complementary to data, and it can potentially help overcome issues such as data sparsity, noise, and inaccuracy. In this work, we propose a simple, yet powerful and general framework — AutoIP, for Automatically Incorporating Physics — that can integrate all kinds of differential equations into Gaussian Processes (GPs) to enhance prediction accuracy and uncertainty quantification. These equations can be linear or nonlinear, spatial, temporal, or spatio-temporal, complete or incomplete with unknown source terms, and so on. Based on kernel differentiation, we construct a GP prior to sample the values of the target function, equation related derivatives, and latent source functions, which are all jointly from a multivariate Gaussian distribution. The sampled values are fed to two likelihoods: one to fit the observations, and the other to conform to the equation. We use the whitening method to evade the strong dependency between the sample.



#  System requirements
python version: 3.10.14

|    Package     | Version  |
|:--------------:|:--------:|
|     torch      |  2.3.1   |
|    sklearn     |  1.5.1  |
|     numpy      |  1.24.3  |
|     pandas     |  2.0.3   |
|   matplotlib   |  3.7.2   |



# Installation guide
If you are not familiar with Python and Pytorch framework, 
you can install Anaconda first and use Anaconda to quickly configure the environment.
## Create environment
```angular2html
conda create -n new_environment python=3.10.14
```



## Activate environment
```angular2html
conda activate new_environment
```

## Install dependencies
```angular2html
conda install pytorch=2.3.1
conda install scikit-learn=1.5.1 numpy=1.24.3 pandas=2.0.3 matplotlib=3.7.2
```

# Run examples

### Yeast Glycolysis

* Without physical constrain:&nbsp;&nbsp;&nbsp;    
    ./YG/YG_LSTM.py 
* With physical constrain:&nbsp;&nbsp;&nbsp;   
    ./YG/YG_PINN.py
* PINN With parameter deviation:&nbsp;&nbsp;&nbsp;   
    ./YG/YG_PINN_deviation.py
  
To run noisy pendulum, add "_noise" to the end. For example, to run noisy yeast glycolysis: use ./YG/YG_LSTM_noise.py.

  
**Note: As we all know, the training process of neural network models is random, 
and the volatility of regression models is often greater than that of classification models. 
Therefore, the results obtained from the above process are not expected to be exactly identical to those mentioned in our manuscript. 
However, it is evident that the results obtained from our method are superior to those of MLP and LSTM.**

In addition, we also provide the results of our training, 
which are saved in the `results` folder and `results analysis` folder. 
These results correspond exactly to the data in our manuscript.


# Citation
If you find it useful, please cite our paper:
```bibtex
@article{zhu2025physics,
  title={Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis},
  author={Wang, Fujin and Zhai, Zhi and Zhao, Zhibin and Di, Yi and Chen, Xuefeng},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={4332},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
