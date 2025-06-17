# Model Evolution Summary

This document tracks the evolution of DenseNet architectures developed during this research, including performance metrics, architectural changes, and observed behaviors.

## Performance Comparison

### Overall Metrics Comparison
This metrics are collected from train and validation datasets logs.

#### F1 Score
F1 Score is metric that shows balance between precision and recall, in this case it is used to measure how well the model is able to identify positive cases (diseases) in the validation and training datasets. This is a main metric of a project and can be calculated using following algorithm

```latex
F1 = 2 * (precision * recall) / (precision + recall)
```

| Model                | Best Val F1 | Avg Val F1 | Best Training F1 | Epochs to Best |
|----------------------|-------------|------------|------------------|----------------|
|SimplifiedDenseNetV1_1|   0.3969    | 0.3447     | 0.9221           | 33             |
|SimplifiedDenseNetV1_2|   0.4119    | 0.3614     | 0.6581           | 54             |
|SimplifiedDenseNetV1_3|   0.1285    | 0.0529     | 0.1331           | 28             |
|DenseNetV1_1          |   0.4244	   | 0.3491	    | 0.7167	         | 32             |
|DenseNetV1_2          |   0.1338	   | 0.0917	    | 0.1483	         | 98             |
|DenseNetV1_3          |   0.4091	   | 0.3152	    | 0.7662	         | 81             |
|DenseNetV2_1          |   0.2841	   | 0.1797	    | 0.7605	         | 29             |
|DenseNetV2_2          |   0.2675	   | 0.1916	    | 0.6611	         | 59             |
|DenseNetV2_3          |   0.3648	   | 0.2771	    | 0.6886	         | 30             |
|DenseNetV3_1          |   0.2716	   | 0.1816	    | 0.8130	         | 35             |
|DenseNetV3_2          |   0.2545	   | 0.1556	    | 0.6106	         | 27             |
|DenseNetV3_3          |   0.4314	   | 0.3571	    | 0.8122	         | 21             |
|DenseNetV4_1          |   0.2755	   | 0.1979	    | 0.7202	         | 82             |
|DenseNetV4_2          |   0.2594	   | 0.1732	    | 0.6788	         | 59             |
|DenseNetV5_1          |   0.2270	   | 0.1656	    | 0.5883	         | 83             |
|DenseNetV5_2          |   0.2194	   | 0.1478	    | 0.5192	         | 47             |
|DenseNetV5_3          |   0.2458	   | 0.1855	    | 0.6353	         | 87             |
|DenseNetV5_4          |   0.2590	   | 0.1928	    | 0.6648	         | 99             |

#### AUC ROC Score

AUC ROC is a metrics that shows how well the model is able to separate positive and negative cases. This is a secondary metric.

| Model                | Best Val AUC | Avg Val AUC | Training AUC | Epochs to Best |
|----------------------|--------------|-------------|--------------|----------------|
|SimplifiedDenseNetV1_1| 0.7612       | 0.7009      | 0.9871       | 11             |
|SimplifiedDenseNetV1_2| 0.7793       | 0.7411      | 0.9226       | 25             |
|SimplifiedDenseNetV1_3| 0.7477       | 0.7114      | 0.7920       | 30             |
|DenseNetV1_1          | 0.7814       | 0.7438      | 0.9369       | 18             |
|DenseNetV1_2          | 0.7501       | 0.7375      | 0.8020       | 37             |
|DenseNetV1_3          | 0.7551       | 0.7162      | 0.9713       | 11             |
|DenseNetV2_1          | 0.7599       | 0.7226      | 0.9713       | 14             |
|DenseNetV2_2          | 0.7634       | 0.7434      | 0.9592       | 16             |
|DenseNetV2_3          | 0.7572       | 0.6964      | 0.9732       | 10             |
|DenseNetV3_1          | 0.7682       | 0.7306      | 0.9789       | 14             |
|DenseNetV3_2          | 0.7599       | 0.7386      | 0.9488       | 12             |
|DenseNetV3_3          | 0.7764       | 0.7318      | 0.9633       | 11             |
|DenseNetV4_1          | 0.7645       | 0.7417      | 0.9703       | 16             |
|DenseNetV4_2          | 0.7716       | 0.7489      | 0.9647       | 20             |
|DenseNetV5_1          | 0.7751       | 0.7459      | 0.9462       | 25             |
|DenseNetV5_2          | 0.7769       | 0.7485      | 0.9310       | 23             |
|DenseNetV5_3          | 0.7721       | 0.7525      | 0.9552       | 25             |
|DenseNetV5_4          | 0.7685       | 0.7488      | 0.9610       | 18             |

#### Precision

Precision is a metrics that shows how many of the predicted positive cases are actually positive. This is a secondary metric that can be calculated using following algorithm

```latex
Precision = True Positives / (True Positives + False Positives)
```

| Model                | Best Val Precision | Avg Val Precision | Training Precision | Epochs to Best |
|----------------------|--------------------|-------------------|--------------------|----------------|
|SimplifiedDenseNetV1_1| 0.7902             | 0.5435            | 0.9647             | 4              |
|SimplifiedDenseNetV1_2| 0.7427             | 0.6048            | 0.8321             | 5              |
|SimplifiedDenseNetV1_3| 0.4483             | 0.3341            | 0.4968             | 26             |
|DenseNetV1_1          | 0.7880             | 0.6161            | 0.8561             | 9              |
|DenseNetV1_2          | 0.5104             | 0.4121            | 0.5094             | 9              |
|DenseNetV1_3          | 0.5749             | 0.4244            | 0.8474             | 11             |
|DenseNetV2_1          | 0.5420             | 0.2976            | 0.8631             | 4              |
|DenseNetV2_2          | 0.4044             | 0.3123            | 0.7984             | 10             |
|DenseNetV2_3          | 0.8889             | 0.6542            | 0.9271             | 2              |
|DenseNetV3_1          | 0.3900             | 0.2977            | 0.8977             | 14             |
|DenseNetV3_2          | 0.5844             | 0.3046            | 0.7760             | 8              |
|DenseNetV3_3          | 0.7705             | 0.5599            | 0.8964             | 4              |
|DenseNetV4_1          | 0.4824             | 0.3144            | 0.8324             | 8              |
|DenseNetV4_2          | 0.7000             | 0.3234            | 0.8075             | 5              |
|DenseNetV5_1          | 0.4429             | 0.3113            | 0.7744             | 9              |
|DenseNetV5_2          | 0.5290             | 0.3170            | 0.7444             | 9              |
|DenseNetV5_3          | 0.5886             | 0.3230            | 0.7932             | 6              |
|DenseNetV5_4          | 0.5000             | 0.3278            | 0.8065             | 2              |

#### Recall

Recall is a metrics that shows how many of the actual positive cases are predicted as positive. This is a secondary metric that can be calculated using following algorithm

```latex
Recall = True Positives / (True Positives + False Negatives)
```

| Model                | Best Val Recall | Avg Val Recall | Training Recall | Epochs to Best |
|----------------------|-----------------|----------------|-----------------|----------------|
|SimplifiedDenseNetV1_1| 0.3824          | 0.3214         | 0.8993          | 43             |
|SimplifiedDenseNetV1_2| 0.3933          | 0.3376         | 0.5970          | 39             |
|SimplifiedDenseNetV1_3| 0.0851          | 0.0378         | 0.0945          | 28             |
|DenseNetV1_1          | 0.4047          | 0.3233         | 0.6574          | 33             |
|DenseNetV1_2          | 0.0898          | 0.0631         | 0.1070          | 9              |
|DenseNetV1_3          | 0.3771          | 0.2707         | 0.7063          | 78             |
|DenseNetV2_1          | 0.2826          | 0.1626         | 0.6898          | 29             |
|DenseNetV2_2          | 0.2573          | 0.1628         | 0.5792          | 59             |
|DenseNetV2_3          | 0.3381          | 0.2375         | 0.5665          | 37             |
|DenseNetV3_1          | 0.2721          | 0.1595         | 0.7497          | 37             |
|DenseNetV3_2          | 0.2403          | 0.1283         | 0.5247          | 27             |
|DenseNetV3_3          | 0.4062          | 0.3191         | 0.7507          | 21             |
|DenseNetV4_1          | 0.2796          | 0.1710         | 0.6482          | 82             |
|DenseNetV4_2          | 0.2469          | 0.1485         | 0.6033          | 67             |
|DenseNetV5_1          | 0.4750          | 0.1437         | 0.5030          | 13             |
|DenseNetV5_2          | 0.1872          | 0.1233         | 0.4352          | 47             |
|DenseNetV5_3          | 0.2150          | 0.1558         | 0.5521          | 97             |
|DenseNetV5_4          | 0.2542          | 0.1637         | 0.5854          | 99             |

### Detailed Class Performance Comparison
This metrics are collected from test dataset logs.

#### F1 Score per Class

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 | DV1_3 | DV2_1 | DV2_2 | DV2_3 | DV3_1 | DV3_2 | DV3_3 | DV4_1 | DV4_2 | DV5_1 |
|--------------------|--------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Atelectasis        | 0.207  | 0.216  | 0.074  | 0.281 | 0.049 | 0.278 | 0.328 | 0.310 | 0.180 | 0.290 | 0.234 | 0.257 | 0.246 | 0.287 | 0.240 |
| Cardiomegaly       | 0.238  | 0.315  | 0.309  | 0.255 | 0.315 | 0.336 | 0.311 | 0.338 | 0.151 | 0.351 | 0.301 | 0.285 | 0.330 | 0.348 | 0.338 |
| Consolidation      | 0.120  | 0.022  | 0.000  | 0.037 | 0.000 | 0.155 | 0.061 | 0.084 | 0.040 | 0.094 | 0.167 | 0.087 | 0.108 | 0.145 | 0.115 |
| Edema              | 0.108  | 0.154  | 0.217  | 0.097 | 0.222 | 0.160 | 0.206 | 0.158 | 0.068 | 0.166 | 0.196 | 0.154 | 0.195 | 0.186 | 0.167 |
| Effusion           | 0.428  | 0.473  | 0.324  | 0.487 | 0.380 | 0.435 | 0.478 | 0.470 | 0.346 | 0.425 | 0.464 | 0.412 | 0.468 | 0.465 | 0.455 |
| Emphysema          | 0.160  | 0.193  | 0.144  | 0.185 | 0.192 | 0.245 | 0.250 | 0.225 | 0.168 | 0.216 | 0.131 | 0.197 | 0.177 | 0.218 | 0.246 |
| Fibrosis           | 0.021  | 0.000  | 0.000  | 0.008 | 0.077 | 0.075 | 0.044 | 0.051 | 0.000 | 0.053 | 0.054 | 0.039 | 0.073 | 0.038 | 0.064 |
| Hernia             | 0.000  | 0.057  | 0.208  | 0.000 | 0.250 | 0.273 | 0.423 | 0.360 | 0.111 | 0.375 | 0.275 | 0.273 | 0.400 | 0.353 | 0.360 |
| Infiltration       | 0.278  | 0.294  | 0.102  | 0.296 | 0.071 | 0.269 | 0.319 | 0.276 | 0.136 | 0.291 | 0.243 | 0.308 | 0.174 | 0.221 | 0.136 |
| Mass               | 0.181  | 0.227  | 0.000  | 0.264 | 0.021 | 0.259 | 0.237 | 0.204 | 0.195 | 0.208 | 0.251 | 0.245 | 0.197 | 0.217 | 0.260 |
| No Finding         | 0.649  | 0.658  | 0.151  | 0.665 | 0.142 | 0.637 |  N/A  |  N/A  | 0.664 |  N/A  |  N/A  | 0.662 |  N/A  |  N/A  |  N/A  |
| Nodule             | 0.064  | 0.057  | 0.000  | 0.052 | 0.000 | 0.096 | 0.106 | 0.143 | 0.088 | 0.117 | 0.131 | 0.130 | 0.026 | 0.101 | 0.128 |
| Pleural_Thickening | 0.079  | 0.025  | 0.000  | 0.030 | 0.000 | 0.098 | 0.040 | 0.084 | 0.089 | 0.073 | 0.056 | 0.077 | 0.019 | 0.103 | 0.125 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 | 0.028 | 0.034 | 0.015 | 0.009 | 0.009 | 0.000 | 0.009 | 0.023 | 0.023 | 0.022 |
| Pneumothorax       | 0.283  | 0.345  | 0.112  | 0.367 | 0.172 | 0.340 | 0.324 | 0.332 | 0.240 | 0.351 | 0.310 | 0.353 | 0.307 | 0.343 | 0.330 |

| Class              | DV5_2 | DV5_3 | DV5_4 |
|--------------------|-------|-------|-------|
| Atelectasis        | 0.199 | 0.277 | 0.275 |
| Cardiomegaly       | 0.354 | 0.348 | 0.298 |
| Consolidation      | 0.162 | 0.118 | 0.173 |
| Edema              | 0.204 | 0.178 | 0.150 |
| Effusion           | 0.463 | 0.455 | 0.452 |
| Emphysema          | 0.215 | 0.232 | 0.224 |
| Fibrosis           | 0.053 | 0.079 | 0.040 |
| Hernia             | 0.250 | 0.314 | 0.431 |
| Infiltration       | 0.096 | 0.193 | 0.277 |
| Mass               | 0.251 | 0.229 | 0.140 |
| Nodule             | 0.068 | 0.110 | 0.032 |
| Pleural_Thickening | 0.088 | 0.098 | 0.058 |
| Pneumonia          | 0.028 | 0.051 | 0.058 |
| Pneumothorax       | 0.340 | 0.358 | 0.324 |

#### Precision per Class

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 | DV1_3 | DV2_1 | DV2_2 | DV2_3 | DV3_1 | DV3_2 | DV3_3 | DV4_1 | DV4_2 | DV5_1 |
|--------------------|--------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Atelectasis        | 0.312  | 0.343  | 0.439  | 0.316 | 0.377 | 0.291 | 0.263 | 0.271 | 0.317 | 0.268 | 0.308 | 0.298 | 0.337 | 0.286 | 0.320 |
| Cardiomegaly       | 0.365  | 0.381  | 0.234  | 0.443 | 0.277 | 0.368 | 0.324 | 0.313 | 0.487 | 0.348 | 0.359 | 0.398 | 0.320 | 0.380 | 0.330 |
| Consolidation      | 0.136  | 0.320  | 0.000  | 0.209 | 0.000 | 0.171 | 0.188 | 0.175 | 0.107 | 0.160 | 0.139 | 0.144 | 0.144 | 0.176 | 0.159 |
| Edema              | 0.205  | 0.226  | 0.163  | 0.286 | 0.186 | 0.205 | 0.183 | 0.204 | 0.206 | 0.272 | 0.172 | 0.248 | 0.148 | 0.246 | 0.181 |
| Effusion           | 0.430  | 0.493  | 0.588  | 0.507 | 0.550 | 0.458 | 0.473 | 0.444 | 0.504 | 0.485 | 0.414 | 0.509 | 0.485 | 0.453 | 0.488 |
| Emphysema          | 0.230  | 0.368  | 0.253  | 0.359 | 0.290 | 0.318 | 0.275 | 0.286 | 0.494 | 0.375 | 0.439 | 0.344 | 0.274 | 0.313 | 0.335 |
| Fibrosis           | 0.088  | 0.000  | 0.000  | 0.333 | 0.083 | 0.137 | 0.111 | 0.069 | 0.000 | 0.157 | 0.110 | 0.113 | 0.160 | 0.092 | 0.098 |
| Hernia             | 0.000  | 1.000  | 0.186  | 0.000 | 0.318 | 0.600 | 0.611 | 0.562 | 1.000 | 0.643 | 0.412 | 0.600 | 0.625 | 0.529 | 0.562 |
| Infiltration       | 0.295  | 0.339  | 0.426  | 0.360 | 0.415 | 0.321 | 0.334 | 0.371 | 0.365 | 0.358 | 0.361 | 0.334 | 0.392 | 0.363 | 0.412 |
| Mass               | 0.248  | 0.378  | 0.000  | 0.436 | 0.529 | 0.310 | 0.435 | 0.338 | 0.356 | 0.386 | 0.219 | 0.353 | 0.311 | 0.292 | 0.375 |
| No Finding         | 0.646  | 0.685  | 0.797  | 0.703 | 0.822 | 0.693 |  N/A  |  N/A  | 0.675 |  N/A  |  N/A  | 0.661 |  N/A  |  N/A  |  N/A  |
| Nodule             | 0.159  | 0.265  | 0.000  | 0.388 | 0.000 | 0.180 | 0.160 | 0.153 | 0.217 | 0.138 | 0.128 | 0.179 | 0.232 | 0.158 | 0.158 |
| Pleural_Thickening | 0.127  | 0.117  | 0.000  | 0.250 | 0.000 | 0.138 | 0.220 | 0.158 | 0.265 | 0.160 | 0.130 | 0.195 | 0.156 | 0.146 | 0.165 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 | 0.056 | 0.030 | 0.034 | 0.071 | 0.043 | 0.000 | 0.111 | 0.028 | 0.055 | 0.045 |
| Pneumothorax       | 0.420  | 0.407  | 0.485  | 0.357 | 0.444 | 0.393 | 0.452 | 0.359 | 0.492 | 0.385 | 0.421 | 0.452 | 0.383 | 0.381 | 0.373 |

| Class              | DV5_2 | DV5_3 | DV5_4 |
|--------------------|-------|-------|-------|
| Atelectasis        | 0.332 | 0.302 | 0.245 |
| Cardiomegaly       | 0.351 | 0.365 | 0.353 |
| Consolidation      | 0.161 | 0.133 | 0.125 |
| Edema              | 0.187 | 0.220 | 0.229 |
| Effusion           | 0.483 | 0.477 | 0.426 |
| Emphysema          | 0.307 | 0.277 | 0.327 |
| Fibrosis           | 0.106 | 0.105 | 0.125 |
| Hernia             | 0.429 | 0.471 | 0.647 |
| Infiltration       | 0.375 | 0.386 | 0.349 |
| Mass               | 0.333 | 0.345 | 0.493 |
| Nodule             | 0.188 | 0.154 | 0.271 |
| Pleural_Thickening | 0.136 | 0.155 | 0.157 |
| Pneumonia          | 0.051 | 0.079 | 0.059 |
| Pneumothorax       | 0.359 | 0.362 | 0.366 |

#### Recall per Class

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 | DV1_3 | DV2_1 | DV2_2 | DV2_3 | DV3_1 | DV3_2 | DV3_3 | DV4_1 | DV4_2 | DV5_1 |
|--------------------|--------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Atelectasis        | 0.155  | 0.158  | 0.040  | 0.253 | 0.026 | 0.266 | 0.433 | 0.361 | 0.126 | 0.315 | 0.188 | 0.226 | 0.194 | 0.287 | 0.193 |
| Cardiomegaly       | 0.176  | 0.268  | 0.454  | 0.179 | 0.365 | 0.309 | 0.300 | 0.367 | 0.089 | 0.355 | 0.258 | 0.222 | 0.341 | 0.321 | 0.348 |
| Consolidation      | 0.108  | 0.012  | 0.000  | 0.020 | 0.000 | 0.143 | 0.036 | 0.055 | 0.025 | 0.067 | 0.210 | 0.063 | 0.086 | 0.124 | 0.090 |
| Edema              | 0.073  | 0.117  | 0.322  | 0.058 | 0.275 | 0.132 | 0.237 | 0.129 | 0.041 | 0.120 | 0.228 | 0.111 | 0.287 | 0.149 | 0.155 |
| Effusion           | 0.427  | 0.454  | 0.223  | 0.468 | 0.290 | 0.415 | 0.482 | 0.498 | 0.263 | 0.378 | 0.529 | 0.346 | 0.452 | 0.477 | 0.426 |
| Emphysema          | 0.122  | 0.130  | 0.101  | 0.125 | 0.144 | 0.199 | 0.229 | 0.186 | 0.101 | 0.152 | 0.077 | 0.138 | 0.130 | 0.168 | 0.194 |
| Fibrosis           | 0.012  | 0.000  | 0.000  | 0.004 | 0.071 | 0.052 | 0.028 | 0.040 | 0.000 | 0.032 | 0.036 | 0.024 | 0.048 | 0.024 | 0.048 |
| Hernia             | 0.000  | 0.029  | 0.235  | 0.000 | 0.206 | 0.176 | 0.324 | 0.265 | 0.059 | 0.265 | 0.206 | 0.176 | 0.294 | 0.265 | 0.265 | 
| Infiltration       | 0.262  | 0.259  | 0.058  | 0.251 | 0.039 | 0.231 | 0.305 | 0.220 | 0.084 | 0.246 | 0.183 | 0.286 | 0.112 | 0.159 | 0.081 |
| Mass               | 0.143  | 0.163  | 0.000  | 0.189 | 0.010 | 0.222 | 0.163 | 0.146 | 0.135 | 0.143 | 0.294 | 0.188 | 0.144 | 0.173 | 0.199 |
| No Finding         | 0.652  | 0.634  | 0.084  | 0.632 | 0.078 | 0.589 |  N/A  |  N/A  | 0.653 |  N/A  |  N/A  | 0.664 |  N/A  |  N/A  |  N/A  |
| Nodule             | 0.040  | 0.032  | 0.000  | 0.028 | 0.000 | 0.066 | 0.080 | 0.135 | 0.055 | 0.102 | 0.134 | 0.102 | 0.014 | 0.074 | 0.107 |
| Pleural_Thickening | 0.058  | 0.014  | 0.000  | 0.016 | 0.000 | 0.075 | 0.022 | 0.058 | 0.054 | 0.048 | 0.036 | 0.048 | 0.010 | 0.079 | 0.101 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 | 0.019 | 0.038 | 0.009 | 0.005 | 0.005 | 0.000 | 0.005 | 0.019 | 0.014 | 0.014 |
| Pneumothorax       | 0.213  | 0.300  | 0.063  | 0.378 | 0.106 | 0.299 | 0.252 | 0.308 | 0.158 | 0.322 | 0.246 | 0.290 | 0.256 | 0.312 | 0.295 |

| Class              | DV5_2 | DV5_3 | DV5_4 |
|--------------------|-------|-------|-------|
| Atelectasis        | 0.143 | 0.257 | 0.314 |
| Cardiomegaly       | 0.357 | 0.333 | 0.258 |
| Consolidation      | 0.164 | 0.106 | 0.284 |
| Edema              | 0.225 | 0.149 | 0.111 |
| Effusion           | 0.445 | 0.434 | 0.480 |
| Emphysema          | 0.165 | 0.199 | 0.170 |
| Fibrosis           | 0.036 | 0.063 | 0.024 |
| Hernia             | 0.176 | 0.235 | 0.324 |
| Infiltration       | 0.055 | 0.129 | 0.230 |
| Mass               | 0.201 | 0.172 | 0.081 |
| Nodule             | 0.041 | 0.085 | 0.017 |
| Pleural_Thickening | 0.065 | 0.071 | 0.036 |
| Pneumonia          | 0.019 | 0.038 | 0.057 |
| Pneumothorax       | 0.323 | 0.354 | 0.290 |

### Loss Metrics Evolution

**Loss** metric allows to track how well our model is learning over time. We can see how many mistakes, our model is doing over time. In summary, we can see that our model learned something based on diff between initial and final loss. And we aim to have a low initial loss and a low final loss.

- **Initial Loss** is a metric that shows the initial loss of the model. In summary it shows how well our model was learning at the beginning. 
- **Final Loss** is a metric that shows the final loss of the model. In summary its a loss at last epoch. In summary it shows how well our model was learning at the end.
- **Rate of Convergence** is a metric that shows the rate of convergence of the model. Convergence shows us how quickly model moves to end state. Bigger value, in theory means model is learning faster.
- **Loss Volatility** is a metric that shows the volatility of the loss of the model. Volatility is a how much the loss changes over time. Lower value means more stable learning process, bigger value means more unstable learning process.

| Model                | Initial Loss | Final Loss | Rate of Convergence | Loss Volatility |
|----------------------|--------------|------------|---------------------|-----------------|
|SimplifiedDenseNetV1_1| 0.7782       | 0.0784     | 0.0149              | 0.1459          |
|SimplifiedDenseNetV1_2| 0.7842       | 0.1581     | 0.0092              | 0.1073          |
|SimplifiedDenseNetV1_3| 0.7668       | 0.2010     | 0.0135              | 0.1218          |
|DenseNetV1_1          | 2.1351       | 0.1413     | 0.0433              | 0.3716          |
|DenseNetV1_2          | 2.1788       | 0.4187     | 0.0178              | 0.2473          |
|DenseNetV1_3          | 2.0327       | 0.106      | 0.0238              | 0.0916          |
|DenseNetV2_1          | 2.0690       | 0.1120     | 0.0455              | 0.3564          |
|DenseNetV2_2          | 1.8808       | 0.0428     | 0.0252              | 0.2447          |
|DenseNetV2_3          | 2.0104       | 0.184      | 0.0415              | 0.302           |
|DenseNetV3_1          | 2.0813       | 0.1121     | 0.0402              | 0.3375          |
|DenseNetV3_2          | 1.8681       | 0.0454     | 0.0445              | 0.3143          |
|DenseNetV3_3          | 2.0247       | 0.2256     | 0.0391              | 0.2880          |
|DenseNetV4_1          | 1.8661       | 0.0338     | 0.0191              | 0.212           |
|DenseNetV4_2          | 1.8597       | 0.026      | 0.0251              | 0.2421          |
|DenseNetV5_1          | 1.7754       | 0.0265     | 0.0213              | 0.2094          |
|DenseNetV5_2          | 1.7698       | 0.0299     | 0.0285              | 0.2388          |
|DenseNetV5_3          | 1.7926       | 0.0270     | 0.0178              | 0.1934          |
|DenseNetV5_4          | 1.7854       | 0.0280     | 0.0178              | 0.1926          |

### Training Dynamics Comparison

#### Learning Convergence Patterns

This section shows the learning convergence patterns of the models. It shows how well the models have converged and how they have stabilized.

- **Converged** is a metric that shows if the model has converged. In summary it shows if the model has reached a stable state.
- **Epochs to Stabilize** is a metric that shows the number of epochs it took for the model to stabilize. In summary it shows how many epochs it took for the model to reach a stable state.
- **Oscillation After Convergence** is a metric that shows the oscillation of the model after convergence. In summary it shows how much the model oscillates after it has reached a stable state.
- **Final vs. Best Epoch** is a metric that shows the final vs. best epoch of the model. In summary it shows how much the model has improved over time.

| Model                | Converged | Epochs to Stabilize | Oscillation After Convergence | Final vs. Best Epoch                     |
|----------------------|-----------|---------------------|-------------------------------|------------------------------------------|
|SimplifiedDenseNetV1_1| Yes       | 5                   | Medium                        | 99.5% (Close to Best (Best Near End))    |
|SimplifiedDenseNetV1_2| Yes       | 4                   | Medium                        | 99.2% (Close to Best (Best Near End))    |
|SimplifiedDenseNetV1_3| Yes       | 19                  | High                          | 84.8% (Moderate Drop)                    |
|DenseNetV1_1          | Yes       | 4                   | Medium                        | 98.4% (Close to Best (Best Near End))    |
|DenseNetV1_2          | Yes       | 19                  | Medium                        | 88.6% (Moderate Drop (Best Near End))    |
|DenseNetV1_3          | Yes       | 24                  | Low                           | 100.0% (Close to Best (Best at Final))   |
|DenseNetV2_1          | Yes       | 10                  | High                          | 94.1% (Moderate Drop)                    |
|DenseNetV2_2          | Yes       | 20                  | Medium                        | 94.5% (Moderate Drop)                    |
|DenseNetV2_3          | Yes       | 8                   | High                          | 98.1% (Close to Best)                    |
|DenseNetV3_1          | Yes       | 24                  | Medium                        | 95.3% (Close to Best)                    |
|DenseNetV3_2          | Yes       | 14                  | High                          | 91.3% (Moderate Drop)                    |
|DenseNetV3_3          | Yes       | 14                  | Low                           | 96.9% (Close to Best)                    |
|DenseNetV4_1          | Yes       | 15                  | Medium                        | 76.0% (Significant Drop (Best Near End)) |
|DenseNetV4_2          | Yes       | 21                  | Medium                        | 86.1% (Moderate Drop)                    |
|DenseNetV5_1          | Yes       | 19                  | Medium                        | 100.0% (Close to Best (Best at Final))   |
|DenseNetV5_2          | Yes       | 12                  | High                          | 98.0% (Close to Best)                    |
|DenseNetV5_3          | Yes       | 16                  | Medium                        | 98.8% (Close to Best (Best Near End))    |
|DenseNetV5_4          | Yes       | 14                  | Medium                        | 83.7% (Moderate Drop (Best Near End))    |

### Test Samples Rate Evolution

#### Test True Positive Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 | DV1_3 | DV2_1 | DV2_2 | DV2_3 | DV3_1 | DV3_2 | DV3_3 | DV4_1 | DV4_2 | DV5_1 |
|--------------------|--------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Atelectasis        | 0.155  | 0.158  | 0.040  | 0.253 | 0.026 | 0.266 | 0.433 | 0.361 | 0.126 | 0.315 | 0.188 | 0.226 | 0.194 | 0.287 | 0.193 |
| Cardiomegaly       | 0.176  | 0.268  | 0.454  | 0.179 | 0.365 | 0.309 | 0.300 | 0.367 | 0.089 | 0.355 | 0.258 | 0.222 | 0.341 | 0.321 | 0.348 |
| Consolidation      | 0.108  | 0.012  | 0.000  | 0.020 | 0.000 | 0.143 | 0.036 | 0.055 | 0.025 | 0.067 | 0.210 | 0.063 | 0.086 | 0.124 | 0.090 |
| Edema              | 0.073  | 0.117  | 0.322  | 0.058 | 0.275 | 0.132 | 0.237 | 0.129 | 0.041 | 0.120 | 0.228 | 0.111 | 0.287 | 0.149 | 0.155 |
| Effusion           | 0.427  | 0.454  | 0.223  | 0.468 | 0.290 | 0.415 | 0.482 | 0.498 | 0.263 | 0.378 | 0.529 | 0.346 | 0.452 | 0.477 | 0.426 |
| Emphysema          | 0.122  | 0.130  | 0.101  | 0.125 | 0.144 | 0.199 | 0.229 | 0.186 | 0.101 | 0.152 | 0.077 | 0.138 | 0.130 | 0.168 | 0.194 |
| Fibrosis           | 0.012  | 0.000  | 0.000  | 0.004 | 0.071 | 0.052 | 0.028 | 0.040 | 0.000 | 0.032 | 0.036 | 0.024 | 0.048 | 0.024 | 0.048 |
| Hernia             | 0.000  | 0.029  | 0.235  | 0.000 | 0.206 | 0.176 | 0.324 | 0.265 | 0.059 | 0.265 | 0.206 | 0.176 | 0.294 | 0.265 | 0.265 | 
| Infiltration       | 0.262  | 0.259  | 0.058  | 0.251 | 0.039 | 0.231 | 0.305 | 0.220 | 0.084 | 0.246 | 0.183 | 0.286 | 0.112 | 0.159 | 0.081 |
| Mass               | 0.143  | 0.163  | 0.000  | 0.189 | 0.010 | 0.222 | 0.163 | 0.146 | 0.135 | 0.143 | 0.294 | 0.188 | 0.144 | 0.173 | 0.199 |
| No Finding         | 0.652  | 0.634  | 0.084  | 0.632 | 0.078 | 0.589 |  N/A  |  N/A  | 0.653 |  N/A  |  N/A  | 0.664 |  N/A  |  N/A  |  N/A  |
| Nodule             | 0.040  | 0.032  | 0.000  | 0.028 | 0.000 | 0.066 | 0.080 | 0.135 | 0.055 | 0.102 | 0.134 | 0.102 | 0.014 | 0.074 | 0.107 |
| Pleural_Thickening | 0.058  | 0.014  | 0.000  | 0.016 | 0.000 | 0.075 | 0.022 | 0.058 | 0.054 | 0.048 | 0.036 | 0.048 | 0.010 | 0.079 | 0.101 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 | 0.019 | 0.038 | 0.009 | 0.005 | 0.005 | 0.000 | 0.005 | 0.019 | 0.014 | 0.014 |
| Pneumothorax       | 0.213  | 0.300  | 0.063  | 0.378 | 0.106 | 0.299 | 0.252 | 0.308 | 0.158 | 0.322 | 0.246 | 0.290 | 0.256 | 0.312 | 0.295 |

| Class              | DV5_2 | DV5_3 | DV5_4 |
|--------------------|-------|-------|-------|
| Atelectasis        | 0.143 | 0.257 | 0.314 |
| Cardiomegaly       | 0.357 | 0.333 | 0.258 |
| Consolidation      | 0.164 | 0.106 | 0.284 |
| Edema              | 0.225 | 0.149 | 0.111 |
| Effusion           | 0.445 | 0.434 | 0.480 |
| Emphysema          | 0.165 | 0.199 | 0.170 |
| Fibrosis           | 0.036 | 0.063 | 0.024 |
| Hernia             | 0.176 | 0.235 | 0.324 |
| Infiltration       | 0.055 | 0.129 | 0.230 |
| Mass               | 0.201 | 0.172 | 0.081 |
| Nodule             | 0.041 | 0.085 | 0.017 |
| Pleural_Thickening | 0.065 | 0.071 | 0.036 |
| Pneumonia          | 0.019 | 0.038 | 0.057 |
| Pneumothorax       | 0.323 | 0.354 | 0.290 |

#### Test False Positive Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 | DV1_3 | DV2_1 | DV2_2 | DV2_3 | DV3_1 | DV3_2 | DV3_3 | DV4_1 | DV4_2 | DV5_1 |
|--------------------|--------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Atelectasis        | 0.043  | 0.038  | 0.006  | 0.069 | 0.005 | 0.082 | 0.152 | 0.122 | 0.034 | 0.108 | 0.053 | 0.067 | 0.048 | 0.090 | 0.051 |
| Cardiomegaly       | 0.008  | 0.012  | 0.041  | 0.006 | 0.026 | 0.015 | 0.017 | 0.022 | 0.003 | 0.018 | 0.013 | 0.009 | 0.020 | 0.014 | 0.020 |
| Consolidation      | 0.032  | 0.001  | 0.000  | 0.004 | 0.000 | 0.032 | 0.007 | 0.012 | 0.010 | 0.016 | 0.061 | 0.017 | 0.024 | 0.027 | 0.022 |
| Edema              | 0.006  | 0.009  | 0.037  | 0.003 | 0.027 | 0.012 | 0.024 | 0.011 | 0.004 | 0.007 | 0.025 | 0.008 | 0.037 | 0.010 | 0.016 |
| Effusion           | 0.084  | 0.069  | 0.023  | 0.067 | 0.035 | 0.073 | 0.079 | 0.092 | 0.038 | 0.059 | 0.111 | 0.049 | 0.071 | 0.085 | 0.066 |
| Emphysema          | 0.010  | 0.006  | 0.007  | 0.006 | 0.009 | 0.011 | 0.015 | 0.012 | 0.003 | 0.006 | 0.002 | 0.007 | 0.009 | 0.009 | 0.010 |
| Fibrosis           | 0.002  | 0.000  | 0.000  | 0.004 | 0.013 | 0.005 | 0.004 | 0.009 | 0.001 | 0.003 | 0.005 | 0.003 | 0.004 | 0.004 | 0.007 |
| Hernia             | 0.000  | 0.000  | 0.002  | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 0.000 |
| Infiltration       | 0.148  | 0.119  | 0.018  | 0.105 | 0.013 | 0.116 | 0.144 | 0.088 | 0.035 | 0.105 | 0.077 | 0.135 | 0.041 | 0.066 | 0.027 |
| Mass               | 0.026  | 0.016  | 0.000  | 0.015 | 0.001 | 0.029 | 0.013 | 0.017 | 0.014 | 0.013 | 0.062 | 0.020 | 0.019 | 0.025 | 0.020 |
| No Finding         | 0.359  | 0.292  | 0.021  | 0.268 | 0.017 | 0.262 |  N/A  |  N/A  | 0.316 |  N/A  |  N/A  | 0.343 |  N/A  |  N/A  |  N/A  |
| Nodule             | 0.014  | 0.006  | 0.000  | 0.003 | 0.000 | 0.000 | 0.027 | 0.049 | 0.013 | 0.041 | 0.060 | 0.030 | 0.003 | 0.026 | 0.037 |
| Pleural_Thickening | 0.013  | 0.004  | 0.000  | 0.002 | 0.000 | 0.000 | 0.003 | 0.010 | 0.005 | 0.008 | 0.008 | 0.007 | 0.002 | 0.016 | 0.017 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 | 0.000 | 0.017 | 0.004 | 0.001 | 0.001 | 0.001 | 0.001 | 0.009 | 0.003 | 0.004 |
| Pneumothorax       | 0.016  | 0.024  | 0.004  | 0.037 | 0.007 | 0.025 | 0.017 | 0.030 | 0.009 | 0.028 | 0.018 | 0.019 | 0.022 | 0.027 | 0.027 |

| Class              | DV5_2 | DV5_3 | DV5_4 |
|--------------------|-------|-------|-------|
| Atelectasis        | 0.036 | 0.075 | 0.122 |
| Cardiomegaly       | 0.018 | 0.016 | 0.013 |
| Consolidation      | 0.040 | 0.032 | 0.093 |
| Edema              | 0.022 | 0.012 | 0.009 |
| Effusion           | 0.070 | 0.070 | 0.096 |
| Emphysema          | 0.009 | 0.013 | 0.009 |
| Fibrosis           | 0.005 | 0.009 | 0.003 |
| Hernia             | 0.001 | 0.001 | 0.000 |
| Infiltration       | 0.022 | 0.049 | 0.102 |
| Mass               | 0.024 | 0.019 | 0.005 |
| Nodule             | 0.012 | 0.030 | 0.003 |
| Pleural_Thickening | 0.014 | 0.013 | 0.007 |
| Pneumonia          | 0.005 | 0.006 | 0.013 |
| Pneumothorax       | 0.031 | 0.034 | 0.027 |

#### Test False Negative Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 | DV1_3 | DV2_1 | DV2_2 | DV2_3 | DV3_1 | DV3_2 | DV3_3 | DV4_1 | DV4_2 | DV5_1 |
|--------------------|--------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Atelectasis        | 0.845  | 0.842  | 0.960  | 0.747 | 0.974 | 0.734 | 0.567 | 0.639 | 0.874 | 0.892 | 0.812 | 0.774 | 0.806 | 0.713 | 0.807 |
| Cardiomegaly       | 0.824  | 0.732  | 0.546  | 0.821 | 0.635 | 0.691 | 0.700 | 0.633 | 0.911 | 0.982 | 0.742 | 0.778 | 0.659 | 0.679 | 0.652 |
| Consolidation      | 0.892  | 0.988  | 1.000  | 0.980 | 1.000 | 0.857 | 0.964 | 0.945 | 0.975 | 0.984 | 0.790 | 0.937 | 0.914 | 0.876 | 0.910 |
| Edema              | 0.927  | 0.883  | 0.678  | 0.942 | 0.725 | 0.868 | 0.763 | 0.871 | 0.959 | 0.993 | 0.772 | 0.889 | 0.713 | 0.851 | 0.845 |
| Effusion           | 0.573  | 0.546  | 0.777  | 0.532 | 0.710 | 0.585 | 0.518 | 0.502 | 0.737 | 0.941 | 0.471 | 0.654 | 0.548 | 0.523 | 0.574 |
| Emphysema          | 0.878  | 0.870  | 0.899  | 0.875 | 0.856 | 0.801 | 0.771 | 0.814 | 0.899 | 0.994 | 0.923 | 0.862 | 0.870 | 0.832 | 0.806 |
| Fibrosis           | 0.988  | 1.000  | 1.000  | 0.996 | 0.929 | 0.948 | 0.972 | 0.960 | 1.000 | 0.997 | 0.964 | 0.976 | 0.952 | 0.976 | 0.952 |
| Hernia             | 1.000  | 0.971  | 0.765  | 1.000 | 0.794 | 0.824 | 0.676 | 0.735 | 0.941 | 1.000 | 0.794 | 0.824 | 0.706 | 0.735 | 0.735 |
| Infiltration       | 0.738  | 0.741  | 0.942  | 0.749 | 0.961 | 0.769 | 0.695 | 0.780 | 0.916 | 0.895 | 0.817 | 0.714 | 0.888 | 0.841 | 0.919 |
| Mass               | 0.857  | 0.837  | 1.000  | 0.811 | 0.990 | 0.778 | 0.837 | 0.854 | 0.865 | 0.987 | 0.706 | 0.812 | 0.856 | 0.827 | 0.801 |
| No Finding         | 0.348  | 0.366  | 0.916  | 0.368 | 0.922 | 0.411 |  N/A  |  N/A  | 0.347 |  N/A  |  N/A  | 0.336 |  N/A  |  N/A  |  N/A  |
| Nodule             | 0.960  | 0.968  | 1.000  | 0.972 | 1.000 | 0.934 | 0.920 | 0.865 | 0.945 | 0.959 | 0.866 | 0.898 | 0.990 | 0.926 | 0.893 |
| Pleural_Thickening | 0.942  | 0.986  | 1.000  | 0.984 | 1.000 | 0.925 | 0.978 | 0.942 | 0.946 | 0.992 | 0.946 | 0.952 | 0.990 | 0.921 | 0.899 |
| Pneumonia          | 1.000  | 1.000  | 1.000  | 1.000 | 1.000 | 0.981 | 0.962 | 0.991 | 0.995 | 0.999 | 1.000 | 0.995 | 0.981 | 0.986 | 0.986 |
| Pneumothorax       | 0.787  | 0.700  | 0.937  | 0.622 | 0.894 | 0.701 | 0.748 | 0.692 | 0.842 | 0.972 | 0.754 | 0.710 | 0.744 | 0.688 | 0.705 |

| Class              | DV5_2 | DV5_3 | DV5_4 |
|--------------------|-------|-------|-------|
| Atelectasis        | 0.857 | 0.743 | 0.686 |
| Cardiomegaly       | 0.643 | 0.667 | 0.742 |
| Consolidation      | 0.836 | 0.894 | 0.716 |
| Edema              | 0.775 | 0.851 | 0.889 |
| Effusion           | 0.555 | 0.566 | 0.520 |
| Emphysema          | 0.835 | 0.801 | 0.830 |
| Fibrosis           | 0.964 | 0.937 | 0.976 |
| Hernia             | 0.824 | 0.765 | 0.676 |
| Infiltration       | 0.945 | 0.871 | 0.770 |
| Mass               | 0.799 | 0.828 | 0.919 |
| Nodule             | 0.959 | 0.915 | 0.983 |
| Pleural_Thickening | 0.935 | 0.929 | 0.964 |
| Pneumonia          | 0.981 | 0.962 | 0.943 |
| Pneumothorax       | 0.677 | 0.646 | 0.710 |

#### Test True Negative Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 | DV1_3 | DV2_1 | DV2_2 | DV2_3 | DV3_1 | DV3_2 | DV3_3 | DV4_1 | DV4_2 | DV5_1 |
|--------------------|--------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Atelectasis        | 0.957  | 0.962  | 0.994  | 0.931 | 0.995 | 0.918 | 0.848 | 0.878 | 0.966 | 0.892 | 0.947 | 0.933 | 0.952 | 0.910 | 0.949 |
| Cardiomegaly       | 0.992  | 0.988  | 0.959  | 0.994 | 0.974 | 0.985 | 0.983 | 0.978 | 0.997 | 0.982 | 0.987 | 0.991 | 0.980 | 0.986 | 0.980 |
| Consolidation      | 0.968  | 0.999  | 1.000  | 0.996 | 1.000 | 0.968 | 0.993 | 0.988 | 0.990 | 0.984 | 0.939 | 0.983 | 0.976 | 0.973 | 0.978 |
| Edema              | 0.994  | 0.991  | 0.963  | 0.997 | 0.973 | 0.988 | 0.976 | 0.989 | 0.996 | 0.993 | 0.975 | 0.992 | 0.963 | 0.990 | 0.984 |
| Effusion           | 0.916  | 0.931  | 0.977  | 0.933 | 0.965 | 0.927 | 0.921 | 0.908 | 0.962 | 0.941 | 0.889 | 0.951 | 0.929 | 0.915 | 0.934 |
| Emphysema          | 0.990  | 0.994  | 0.993  | 0.994 | 0.991 | 0.989 | 0.985 | 0.988 | 0.997 | 0.994 | 0.998 | 0.993 | 0.991 | 0.991 | 0.990 |
| Fibrosis           | 0.998  | 1.000  | 1.000  | 0.999 | 0.987 | 0.995 | 0.996 | 0.991 | 0.999 | 0.997 | 0.995 | 0.997 | 0.996 | 0.996 | 0.993 |
| Hernia             | 1.000  | 1.000  | 0.998  | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 | 1.000 |
| Infiltration       | 0.852  | 0.881  | 0.982  | 0.895 | 0.987 | 0.884 | 0.856 | 0.912 | 0.965 | 0.895 | 0.932 | 0.865 | 0.959 | 0.934 | 0.973 |
| Mass               | 0.974  | 0.984  | 1.000  | 0.985 | 0.999 | 0.971 | 0.987 | 0.983 | 0.986 | 0.987 | 0.938 | 0.980 | 0.981 | 0.975 | 0.980 |
| No Finding         | 0.641  | 0.708  | 0.979  | 0.732 | 0.983 | 0.738 |  N/A  |  N/A  | 0.684 |  N/A  |  N/A  | 0.657 |  N/A  |  N/A  |  N/A  |
| Nodule             | 0.986  | 0.994  | 1.000  | 0.997 | 1.000 | 0.980 | 0.973 | 0.951 | 0.987 | 0.959 | 0.940 | 0.970 | 0.997 | 0.974 | 0.963 |
| Pleural_Thickening | 0.987  | 0.996  | 1.000  | 0.998 | 1.000 | 0.984 | 0.997 | 0.990 | 0.995 | 0.992 | 0.992 | 0.993 | 0.998 | 0.984 | 0.983 |
| Pneumonia          | 1.000  | 1.000  | 1.000  | 1.000 | 1.000 | 0.996 | 0.983 | 0.996 | 0.999 | 0.999 | 0.999 | 0.999 | 0.991 | 0.997 | 0.996 | 
| Pneumothorax       | 0.984  | 0.976  | 0.996  | 0.963 | 0.993 | 0.975 | 0.983 | 0.970 | 0.991 | 0.972 | 0.982 | 0.981 | 0.978 | 0.973 | 0.937 |

| Class              | DV5_2 | DV5_3 | DV5_4 |
|--------------------|-------|-------|-------|
| Atelectasis        | 0.964 | 0.925 | 0.878 |
| Cardiomegaly       | 0.982 | 0.984 | 0.987 |
| Consolidation      | 0.960 | 0.968 | 0.907 |
| Edema              | 0.978 | 0.988 | 0.991 |
| Effusion           | 0.930 | 0.930 | 0.904 |
| Emphysema          | 0.991 | 0.987 | 0.991 |
| Fibrosis           | 0.995 | 0.991 | 0.997 |
| Hernia             | 0.999 | 0.999 | 1.000 |
| Infiltration       | 0.978 | 0.951 | 0.898 |
| Mass               | 0.976 | 0.981 | 0.995 |
| Nodule             | 0.988 | 0.970 | 0.997 |
| Pleural_Thickening | 0.986 | 0.987 | 0.993 |
| Pneumonia          | 0.995 | 0.994 | 0.987 |
| Pneumothorax       | 0.969 | 0.966 | 0.973 |

## Architectural Changes and Impacts

### SimplifiedDenseNetV1_1 (Baseline)
Baseline model with default configuration, no changes applied only Early Stopping was enabled to reduce training time.

#### Configuration
- Epochs: 33(100)
- Batch size: 128
- Optimizer: Adam(0.0001)
- Loss: Binary Crossentropy
- Image Augmentation: None
- Architecture: SimplifiedDenseNet
- Learning rate schedule: ReduceLROnPlateau
- Class weights: None

### SimplifiedDenseNetV1_2

#### Configuration
- **Changes from v1:** 
  - Added Basic Image Augmentation
- **Hypothesis:** 
  - Image augmentation will improve model generalization by 
    exposing it to more varied training samples and reducing overfitting
- **Impact:** 
  - Improved model robustness to variations in input images
  - Better generalization on test data
  - Reduced overfitting compared to v1

### SimplifiedDenseNetV1_3

#### Configuration
- **Changes from v2:** 
  - Added class weights to model training
- **Hypothesis:** 
  - Class weights will help address class imbalance 
    issues and improve performance on minority classes
- **Impact:**
  - Better performance on minority classes
  - Most classes showed decreased performance 
  - Significantly decreased overall F1 score (from 0.4119 to 0.1285)

### DenseNetV1_1

#### Configuration
- **Changes from v3:**
  - Full DenseNet121 architecture with bottleneck layers
  - Basic Image Augmentation from previous versions
  - Removed class weights
- **Hypothesis:**
  - Full DenseNet121 architecture with all its layers and features
    - Overall model performance
    - Feature extraction capabilities
- **Impact:**
  - Improved overall metrics compared to SimplifiedDenseNetV1_2

### DenseNetV1_2

#### Configuration
- **Changes from v1:**
  - Added class weights to model training
- **Hypothesis:**
  - Class weights will help address class imbalance 
    issues and improve performance on minority classes
- **Impact:**
  - Better performance on minority classes
  - Most classes showed decreased performance

### DenseNgved recall on minority classes through oversampling
  - Slightly lower overall precision but better F1-scores
  - Better handling of class imbalance without sacrificing model stability

### DenseNetV2_1

#### Configuration
- **Changes from v1_3:**
  - Filtered dataset to remove No Finding class
- **Hypothesis:**
  - Removing No Finding class will help address class imbalance 
    issues and improve performance on minority classes
- **Impact:**
  - Successfully addressed class imbalance by removing "No Finding"
  - Improved focus on actual pathology detection
  - High loss volatility during training
  - Inconsistent performance across different pathologies

### DenseNetV2_2

#### Configuration
- **Changes from v2_1:**
  - Switched to use Binary Focal Cross-Entropy Loss
- **Hypothesis:**
  - Binary Focal Cross-Entropy Loss will help address class imbalance 
    issues and improve performance on minority classes
- **Impact:**
  - Better performance on minority classes
  - Most classes showed decreased performance

### DenseNetV2_3

#### Configuration
- **Changes from v2_2:**
  - Added No Finding class back to dataset
  - Switched to custom loss function that addresses No Finding class specifics
- **Hypothesis:**
  - Custom loss function will punish more when No Finding class is predicted with other diseases
- **Impact:**
  - Better performance on No Finding class
  - Most classes showed decreased performance

### DenseNetV3_1
This is model DenseNetV2_1 with added label smoothing

#### Configuration
- **Changes from v2_1:**
  - Added label smoothing 
- **Hypothesis:**
  - Label smoothing will help reduce model overconfidence and improve generalization
  - Small value (0.01) chosen to maintain model confidence while providing minimal regularization
  - Should help stabilize training without significantly impacting model's ability to make confident predictions
- **Impact:**
  - Similar slightly lower test performance to previous versions.
  - Higher training performance, 
  - Model is even more overfitting than previous versions

### DenseNetV3_2
This is model DenseNetV2_2 with added label smoothing

#### Configuration
- **Changes from v2_2:**
  - Added label smoothing
- **Hypothesis:**
  - Label smoothing will help reduce model overconfidence and improve generalization
  - Small value (0.01) chosen to maintain model confidence while providing minimal regularization
  - Should help stabilize training without significantly impacting model's ability to make confident predictions
- **Impact:**
  - Better generalization than V2_2
  - Improved performance on minority classes
  - Slightly lower overall F1 score compared to V2_2 and V3_1
  - More balanced learning with still visible overfitting

### DenseNetV3_3
This is model DenseNetV2_3 with added label smoothing

#### Configuration
- **Changes from v2_3:**
  - Added label smoothing
- **Hypothesis:**
  - Label smoothing will help reduce model overconfidence and improve generalization
  - Small value (0.01) chosen to maintain model confidence while providing minimal regularization
  - Should help stabilize training without significantly impacting model's ability to make confident predictions
- **Impact:**
  - Better generalization than V2_3
  - Improved performance on minority classes
  - Slightly lower overall F1 score compared to V2_2 and V3_1
  - More balanced learning with still visible overfitting

## DenseNetV2 & V3 Summary

### Filtered or No Filtered No Finding Class
- **Filtered:** 
  - Models without No Finding class generally show:
    - Better performance on minority classes
    - Visibly lower overall F1 score
- **No Filtered:**
  - Models with No Finding class generally show:
    - Better overall performance yet mostly visible in No Finding class

### Focal Loss vs Plain Binary Loss

#### Training Dynamics
- **BinaryCrossentropy (V2_1, V3_1)**
  - V2_1: More stable validation performance
  - V3_1: Shows higher but volatile validation scores
  - Training F1 reaches ~0.8
  - Larger train-test gap (especially in V3_1: ~0.54)

- **FocalLoss (V2_2, V3_2, V3_3)**
  - V2_2: More controlled training curve (max F1 ~0.65)
  - V3_2: Shows early performance decline
  - Better generalization (smaller train-test gap ~0.35)
  - More consistent epoch-to-epoch performance

#### Performance Characteristics
- **BinaryCrossentropy**
  - Higher training ceiling
  - More prone to overfitting
  - Better recall on minority classes

- **FocalLoss**
  - More controlled learning
  - Better generalization
  - More stable long-term performance

### DenseNetV4_1

#### Configuration
- **Changes from v3_2:**
  - Set label smoothing to 0.05
- **Hypothesis:**
  - Increasing label smoothing from 0.01 to 0.05 would further reduce model overconfidence
  - Higher smoothing value would provide stronger regularization against overfitting
- **Impact:**
  - Still shows evidence of overfitting but to a lesser degree than V3_2
  - Improved F1 score (from 0.2545 to 0.2755) representing approximately 8% relative improvement

### DenseNetV4_2

#### Configuration
- **Changes from v4_1:**
  - Set label smoothing to 0.1
- **Hypothesis:**
  - More aggressive smoothing would further reduce overfitting and improve generalization
  - Could potentially lead to more balanced performance across all pathology classes
- **Impact:**
  - Slightly decreased F1 score (from 0.2755 to 0.2594) suggesting that smoothing value may be too aggressive
  - Higher smoothing appears to reach the point of diminishing returns in this architecture
  
### DenseNetV5_1

#### Configuration
- **Changes from v4_1:**
  - Add Squeeze-and-Excitation (SE) blocks, ratio = 16
- **Hypothesis:**
  - Better feature refinement across channels
  - Improved ability to identify relevant features for specific pathologies
- **Impact:**
  - Decreased F1 score (from 0.2755 to 0.2270) suggesting SE blocks negatively affected overall performance
  - Significantly improved recall (from 0.2796 to 0.4750) showing better detection of positive cases
  - Higher AUC ROC (0.7751 vs 0.7645) indicating better discrimination ability
  - Reduced training F1 (0.5883 vs 0.7202) with smaller train-validation gap, suggesting less overfitting
  - Overall trade-off: better generalization and improved recall at the cost of precision and F1 score

### DenseNetV5_2

#### Configuration
- **Changes from v4_1:**
  - Add Squeeze-and-Excitation (SE) blocks, ratio = 8
- **Hypothesis:**
  - More aggressive channel attention (lower ratio) would produce stronger feature refinement
  - Enhanced modeling of interdependencies between channels should improve classification performance
  - Better feature recalibration might help with challenging pathology detection
- **Impact:**
  - Further decrease in F1 score compared to both V4_1 (0.2194 vs 0.2755) and V5_1 (0.2194 vs 0.2270)
  - Slight improvement in AUC ROC over V5_1 (0.7769 vs 0.7751) indicating marginally better discrimination
  - Reduced training F1 (0.5192 vs 0.7202) with still substantial overfitting
  - Lower SE ratio appears counterproductive, suggesting more aggressive channel attention negatively impacts model performance

### DenseNetV5_3

#### Configuration
- **Changes from v5_2:**
  - Replace Squeeze-and-Excitation (SE) blocks with ECA blocks
- **Hypothesis:**
  - ECA blocks will provide similar channel attention benefits
  - ECA blocks should help improve feature recalibration and model performance
- **Impact:**
  - Improved F1 score compared to V5_2 (0.2458 vs 0.2194) but still lower than V4_1 (0.2755)
  - Slightly lower AUC ROC than V5_2 (0.7721 vs 0.7769) but remains competitive with previous versions
  - ECA blocks appear more effective than SE blocks with ratio=8, providing better balance between feature refinement and overall classification performance
  - Still shows significant overfitting but performs better than previous channel attention mechanisms

### DenseNetV5_4

#### Configuration
- **Changes from v5_3:**
  - Add ECA blocks only to Dense Block 2 & 3
- **Hypothesis:**
  - ECA blocks will provide similar channel attention benefits
  - ECA blocks should help improve feature recalibration and model performance
- **Impact:**
  - Further improved F1 score (0.2590) compared to V5_3 (0.2458), showing continued progress with channel attention refinement
  - More balanced performance across pathology classes with improved recall for several conditions
  - Selective application of ECA blocks appears more effective than applying them throughout the network
  - Best performing version in the V5 series, suggesting that targeted channel attention at mid-network layers provides optimal feature refinement