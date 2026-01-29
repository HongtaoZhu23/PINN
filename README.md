# PINN4Bioprocess
This code is for our paper: [Calibration-Free Physics-Informed Neural Networks for Adaptive State Inference in Bioprocesses]

![https://github.com/HongtaoZhu23/PINN/blob/main/Bioprocess.png](https://github.com/HongtaoZhu23/PINN/blob/main/Bioprocess.png)
![https://github.com/HongtaoZhu23/PINN/blob/main/Results.png](https://github.com/HongtaoZhu23/PINN/blob/main/Results.png)



Accurate real-time prediction of key quality indicators remains a major challenge in industrial bioprocessing, where complex, time-varying kinetics and unmeasurable metabolic states hinder process optimization. Existing physics-informed neural networks (PINNs) offer a promising hybrid paradigm by integrating data and mechanistic knowledge, yet their reliance on fully calibrated models limits robustness under parameter drift and unmeasurable state variables. To address these limitations, this study proposes a calibration-free PINN framework that jointly estimates key quality indicators and critical, unmeasurable internal system states, enabling robust extrapolation under parameter drift and limited data. Validated by industrial penicillin fermentation (cross-strain/process), simulations, and chemical cases, the framework demonstrates robust extrapolation under noise and parameter variability, accurately predicting key quality indicators. Empirically, the accuracy of unmeasurable state variable estimation is bounded by that of the physical model parameters. This affirms a high-precision, strongly generalizable modeling paradigm for complex bioprocesses.



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


