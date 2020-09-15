## Overview

Solution for Kaggle [Home Credit Default Risk Challenge](https://www.kaggle.com/c/home-credit-default-risk).
This repository contains logistic regression and LightGBM model training on features proposed [here](https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features).

LightGBM model achieves AUC-ROC 0.78819 on Public LB.

The pipeline visualized with `kedro viz`:
![alt text](https://github.com/VladSkripniuk/home_credit/blob/master/screenshots/Screenshot%20from%202020-09-15%2000-45-35.png)
Metrics, models and predictions are logged in Mlflow:
![alt text](https://github.com/VladSkripniuk/home_credit/blob/master/screenshots/Screenshot%20from%202020-09-15%2000-46-39.png)


## Installing dependencies

To install dependencies, run:

```
kedro install
```

## Running Kedro

You can run this Kedro project with:

```
kedro run
```

## Testing Kedro

You can run unit tests with the following command:

```
kedro test
```

## Building API documentation

To build API docs for this project using Sphinx, run:

```
kedro build-docs
```

See documentation by opening `docs/build/html/index.html`.
