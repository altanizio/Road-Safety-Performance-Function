import numpy as np
import pandas as pd
from statsmodels.api import NegativeBinomial
from statsmodels.tools.tools import add_constant
from scipy.stats import chi2
import matplotlib.pyplot as plt


class RSPF:
    """
    Road Safety Performance Function (RSPF)

    A wrapper for fitting Negative Binomial models in road safety studies,
    with additional utilities for results inspection and diagnostic plots.
    """

    def __init__(self, data, y, X=None, X_log=None, constant=True):
        """
        Initialize the RSPF model.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing dependent and independent variables.
        y : str
            Name of the dependent variable column.
        X : list[str], optional
            List of independent variable columns. If None, all except `y` are used.
        X_log : list[str], optional
            List of independent variables to log-transform.
        constant : bool, default=True
            Whether to add a constant (intercept) to the model.
        """
        self.y = data[y].copy()
        self.data = data.copy()

        # Independent variables
        if X:
            self.X = data[X].copy()
        else:
            self.X = data[data.columns.difference([y])].copy()

        # Log-transform selected variables
        if X_log is not None:
            self.X[X_log] = np.log(self.X.loc[:, X_log])
            if self.X[X_log].isnull().any().any():
                raise ValueError("Log transformation resulted in NaN values.")
            self.X.rename(columns={col: f"{col}_log" for col in X_log}, inplace=True)

        # Add intercept
        if constant:
            self.X = add_constant(self.X)

        # Fit Negative Binomial model
        self.model = NegativeBinomial(self.y, self.X).fit(disp=0)
        self._compute_results()

    def predict(self, X, X_log=None, constant=True):
        """
        Predict values using the fitted model.
        """
        if X_log is not None:
            X[X_log] = np.log(X[X_log])
            if X[X_log].isnull().any().any():
                raise ValueError("Log transformation resulted in NaN values.")
            X.rename(columns={col: f"{col}_log" for col in X_log}, inplace=True)

        # Add intercept
        if constant:
            X = add_constant(X)
        return self.model.predict(X)

    def _compute_results(self):
        """
        Compute and store model diagnostics and summary statistics.
        """
        n = self.model.nobs
        p = self.model.df_model + 1

        fitted = np.exp(self.model.fittedvalues)
        resid = self.model.resid_response

        lnalpha = getattr(self.model, "lnalpha", None)
        alpha = np.exp(lnalpha) if lnalpha is not None else None
        theta = 1 / alpha if alpha is not None else None

        y = self.model.model.endog
        mu = fitted
        if alpha is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                term1 = np.where(y > 0, y * np.log(y / mu), 0.0)
                term2 = (y + theta) * np.log((y + theta) / (mu + theta))
                D_i = 2 * (term1 - term2)
                deviance = np.sum(D_i)
        else:
            deviance = np.nan

        residuals_sq = resid**2
        if alpha is not None:
            X2 = np.sum(residuals_sq / (fitted + fitted**2 * alpha))
        else:
            X2 = np.nan

        dispersion = X2 / (n - p) if n > p else np.nan

        self.results = {
            "AIC": self.model.aic,
            "LogLik": self.model.llf,
            "-2 LogLik": -2 * self.model.llf,
            "N_obs": n,
            "Residual df": self.model.df_resid,
            "Pearson Chi²": X2,
            "Dispersion": dispersion,
            "Deviance": deviance,
            "Alpha": alpha,
            "Theta": theta,
            "Critical Chi² (95%)": chi2.ppf(0.95, df=self.model.df_resid),
            "Coefficients": pd.DataFrame(
                {
                    "Variable": self.model.params.index,
                    "Estimate": self.model.params.values,
                    "P-value": self.model.pvalues.values,
                }
            ),
            "Deviance Residuals": D_i if alpha is not None else None,
        }

    def _build_rsp_formula(self):
        """
        Build RSPF-style formula for the fitted model:
        E(K) = const * X1^b1 * X2^b2 * exp(sum(other_vars * coef))
        """
        const = self.model.params.get("const", 1.0)
        power_terms = []
        exp_terms = []

        for var, coef in zip(self.model.params.index, self.model.params.values):
            if var == "const" or var == "alpha":
                continue
            if var.endswith("_log"):
                original_var = var.replace("_log", "")
                power_terms.append(f"{original_var}^{coef:.6f}")
            else:
                exp_terms.append(f"{coef:.6f}*{var}")

        formula = f"E(K) = {np.exp(const):.6f}"
        if power_terms:
            formula += " * " + " * ".join(power_terms)
        if exp_terms:
            formula += " * exp(" + " + ".join(exp_terms) + ")"
        return formula

    def metrics(self):
        res = self.results

        stats = [
            ["LogLik", f"{res['LogLik']:.3f}"],
            ["-2 LogLik", f"{res['-2 LogLik']:.3f}"],
            ["Residual df", res["Residual df"]],
            ["Pearson Chi²", f"{res['Pearson Chi²']:.3f}"],
            ["Critical Chi² (95%)", f"{res['Critical Chi² (95%)']:.3f}"],
            ["Deviance", f"{res['Deviance']:.3f}"],
            ["AIC", f"{res['AIC']:.3f}"],
            ["Dispersion", f"{res['Dispersion']:.3f}"],
            ["Alpha", f"{res['Alpha']:.5f}" if res["Alpha"] is not None else "—"],
            ["Theta", f"{res['Theta']:.5f}" if res["Theta"] is not None else "—"],
            ["Observations", res["N_obs"]],
        ]

        df_stats = pd.DataFrame(stats, columns=["Metric", "Value"])

        return df_stats

    def coef(self):
        coef_table = self.results["Coefficients"].copy()
        coef_table["Estimate"] = coef_table["Estimate"].apply(lambda x: f"{x:.5f}")
        coef_table["P-value"] = coef_table["P-value"].apply(lambda x: f"{x:.5f}")

        return coef_table

    def summary(self):
        """
        Print a user-friendly summary of the fitted model,
        including the estimated regression formula.
        """

        print("\n" + "=" * 60)
        print(" ROAD SAFETY PERFORMANCE FUNCTION - SUMMARY ".center(60, "="))
        print("=" * 60 + "\n")

        # --- General statistics ---
        print(self.metrics().to_string(index=False))
        print("\n")

        # --- Coefficients ---
        print("Model Coefficients:\n")
        print(self.coef().to_string(index=False))
        print("\n")

        # --- Model formula ---
        print("Model Formula:\n")
        print(self._build_rsp_formula())
        print("\n" + "=" * 60 + "\n")

    def cureplot(self, covariate: pd.Series, residuals: np.ndarray = None):
        """
        Generate a CURE (Cumulative Residuals) plot. If residuals are not provided,
        they will be computed from the model's fitted values.
        """
        if residuals is None:
            resid = self.model.resid_response
        else:
            resid = residuals

        resid_cum = np.cumsum(resid)

        resid_sq = resid**2
        resid_sq_cum = np.cumsum(resid_sq)
        resid_sq_sum = np.sum(resid_sq)
        Qr = np.sqrt(resid_sq_cum) * np.sqrt(1 - resid_sq_cum / resid_sq_sum)
        upper = Qr * 2
        lower = -Qr * 2

        cure_data = pd.DataFrame(
            {
                covariate.name: covariate.values,
                "CumulativeResiduals": resid_cum,
                "UpperBound": upper,
                "LowerBound": lower,
            }
        )

        return cure_data

    def plot_cureplot(
        self, covariate: pd.Series, residuals: np.ndarray = None, figsize=(8, 5)
    ):
        """
        Plot the CURE (Cumulative Residuals) plot for a given covariate.
        """
        cure_data = self.cureplot(covariate, residuals=residuals)

        plt.figure(figsize=figsize)
        plt.plot(
            cure_data[covariate.name],
            cure_data["CumulativeResiduals"],
            label="Cumulative Residuals",
        )
        plt.plot(
            cure_data[covariate.name],
            cure_data["UpperBound"],
            "r--",
            label="Upper Limit",
        )
        plt.plot(
            cure_data[covariate.name],
            cure_data["LowerBound"],
            "r--",
            label="Lower Limit",
        )
        plt.xlabel(covariate.name)
        plt.ylabel("Cumulative Residuals")
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.grid(True, linestyle="--", alpha=0.6)
