# PINN4Bioprocess
This code is for our paper: [Physics-Constrained Latent States Reconciliation PINN Framework for Complex Bio-process Soft Sensing Problems]

![https://github.com/HongtaoZhu23/PINN/blob/main/Bioprocess.png](https://github.com/HongtaoZhu23/PINN/blob/main/Bioprocess.png)
![https://github.com/HongtaoZhu23/PINN/blob/main/Results.png](https://github.com/HongtaoZhu23/PINN/blob/main/Results.png)



Accurate online predicting of key quality indicators in bioprocesses is critical for operation safety and optimal control. However, many bioprocesses are often transient in nature and have complex kinetics, resulting in significant challenges for developing online monitoring models. Data-driven models exhibit poor generalizability due to delays in data transmission and existing of unmeasurable crucial state variables, while physics-based models are often hard to be well-calibrated for an industrial process due to data availability and batch variations in reaction materials and operating conditions. The Physics-Informed Neural Network (PINN) endeavors to overcome the above challenges, but usually requires well-established mechanistic models, and meets challenges facing drifting model parameters and unmeasurable latent variables. To address these limitations, this study proposes an innovative PINN framework that leverages the kinetics models of parameter drifting issue with limited measured data for real-time estimating latent variables for soft sensing. The proposed approach is thoroughly tested across an industrial penicillin fermentation study (cross-strain/cross-process extrapolation), theoretic simulation experiments, and three representative chemical reaction cases. It is found the proposed modeling framework has excellent extrapolation capabilities and the accuracy of the estimated latent variables is empirically bounded by the accuracy of the physical model parameters, demonstrating a high-precision, strongly generalizable modeling paradigm for complex biochemical processes.



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
