# RSPF - Road Safety Performance Function

A Python package for modeling road safety performance using **Negative Binomial regression**. This package is designed to help traffic safety researchers fit, analyze, and visualize crash prediction models with robust diagnostics.

---

## Features

- Fit Negative Binomial models for crash count data.
- Automatic log-transformation of selected independent variables.
- Computes model diagnostics:
  - Deviance
  - Pearson Chi²
  - Dispersion
  - AIC and Log-Likelihood
- Display coefficients and p-values in a user-friendly format.
- Generate **CURE plots** (Cumulative Residuals) for model checking.
- Construct RSPF-style formulas for predicted crash counts.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/rspf.git
cd rspf
```

Make sure you have the required dependencies installed:

```
pip install numpy pandas statsmodels scipy matplotlib
```

## Usage

```
import pandas as pd
from rspf import RSPF

# Load your dataset
data = pd.read_csv("data_exemple/your_data.csv")

# Initialize the model
model = RSPF(
    data,
    y="N_CRASH",              # Dependent variable
    X=["N_LANES", "AADT"],    # Independent variables
    X_log=["AADT"],            # Variables to log-transform
    constant=True
)

# View summary
model.summary()

# Generate CURE plot
cure_data = model.cureplot(data["AADT"])
model.plot_cureplot(data["AADT"])

```

## Methods

### `summary()`

Prints a detailed summary of the fitted model, including coefficients, statistics, and the RSPF formula.

### `metrics()`

Returns a DataFrame with key model metrics:

* Log-likelihood
* Deviance
* Pearson Chi²
* AIC
* Dispersion
* Alpha and Theta

### `coef()`

Returns a DataFrame with model coefficients and p-values.

### `predict(X)`

Predict expected crash counts for new observations.

### `cureplot(covariate, residuals=None)`

Returns cumulative residuals for a covariate.

### `plot_cureplot(covariate, residuals=None)`

Plots the CURE (Cumulative Residuals) plot for a given covariate.


## Referências

- **Hauer, E.** (2015). *The Art of Regression Modeling in Road Safety*. Springer. ISBN: 978-3-319-12528-2. Disponível em: [Springer Link](https://link.springer.com/book/10.1007/978-3-319-12529-9)
- **Hauer, E.** (2004). Statistical Road Safety Modeling. *Transportation Research Record*, 1897(1), 11–16. DOI: [10.3141/1897-11](https://journals.sagepub.com/doi/10.3141/1897-11)
- **Hauer, E.** (2015). Observational Before–After Studies in Road Safety: Estimating the Effect of Highway and Traffic Engineering Measures. *Springer*. ISBN: 978-3-319-35446-0. 

## License

This project is licensed under the MIT License.
