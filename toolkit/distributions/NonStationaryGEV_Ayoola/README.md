# Documentation for Automatic Time-Dependent GEV fit: GEVnonstat

The aim of this class is to select and calculate the parameters which minimize the AIC related to time-dependent GEV distribution using the Maximum Likelihood method within an iterative scheme, including one parameter at a time based on a perturbation criteria. The process is repeated until no further improvement in the objective function is achieved. 

The Non-Stationary part is introduced in the three GEV parameters ($\mu_t, \psi_t, \xi_t$) as follows:

$$
\begin{align*}
\mu_t &= \beta_0 + \sum_{i=1}^{P_\mu}[\beta_{2i-1}cos(iwt) + \beta_{2i} sin(iwt)] + \beta_{LT}t + \sum_{k=1}^{Q_\mu}\beta_k^{co}n_{k,t}\\
log(\psi_t) &= \alpha_0 + \sum_{i=1}^{P_\psi}[\alpha_{2i-1}cos(iwt) + \alpha_{2i} sin(iwt)] + \alpha_{LT}t + \sum_{k=1}^{Q_\psi}\alpha_k^{co}n_{k,t}\\

\xi_t &= \gamma_0 + \sum_{i=1}^{P_\xi}[\gamma_{2i-1}cos(iwt) + \gamma_{2i} sin(iwt)] + \sum_{k=1}^{Q_\xi}\gamma_k^{co}n_{k,t}
\end{align*}
$$

The main function of this class is the AutoAdjust.

## Class Definition

### Input variables
- `xt`: Environmental maxima data.
- `t`: Time when maxima occurs in yearly scale.
- `kt`: Frequency parameter to measure the importance of the number of points.
- `covariates`: covariate dictionary where each column corresponds to the values of a covariate at the times when maxima occurs (Dictionary containing the data).
- `trends`: Number of trends to introduced
- `example`: Boolean to see if a plot has to be created.
- `quanval`: Quantile value to be plotted (default 0.95).
- - harm: Boolean if harmonic part should be introduced

#### Input Variables type:

| Variable     | Type          | Default       |
|--------------|---------------|---------------|
| `xt`         | numpy.array   |       -       |
| `t`          | numpy.array   |       -       |
| `kt`         | numpy.array   |      None     |
| `covariates` | dictionary    |      None     |
| `trends`     | Boolean       |      False    |
| `harm`       | Boolean       |      True     |
| `example`    | String        |      None     |
| `quanval`    | 0-1 float     |      0.95     |

### Functions
Functions implemented in the `GEVnonstat2` class.

- `__init__`: Initialization of the data.
- `validate_data`: Validate the data introduced.
- `AutoAdjust`: Perform the algorithm implemented (Most important function). Inputs: 
    - `maxiter`: number of iterations for the algorithm (100 as default).
    - `plot`: boolean value which calls the plot function (False as default).
- `_AIC`: (Static Method) Compute the Akaike Information Criterion given certain loglikelihood value and the number of parameters.
- `_optimize_parameters`: Function to estimate the parameters of the Time-Dependent GEV distribution using the Maximum Likelihood Method.
- `_auxmin_loglikelihood`: Auxiliar function with the loglikelihood function to use in the `_optimize_parameters` minimization step.
- `_auxmin_loglikelihood_grad`: Auxiliar function with the gradient of the loglikelihood function to use in the `_optimize_parameters` minimization step.
- `compute_numerical_hessian`: (NOT USED) Auxiliar function to compute the numerical hessian matrix of the loglikelihood.
- `_loglikelihood`: Function which calculate the loglikelihood, the Gradient and Hessian given certain parameters.
- `_parametro`: Calculates the value of the location, scale and shape parameter, one in each call. 
- `_search`: Auxiliar function used in `_parametro` to search the nearest value of certain time.
- `_evaluate_params`:Function to evaluate the parameters in the corresponding values ($\beta_0,\beta,\dots$).
- `_Dparam`: Derivative of the location, scale and shape functions with respect to harmonic parameters. It corresponds to the rhs in equation (A.11) of the paper.
- `_quantile`: Calculates the quantile q associated with a given parameterization, the main input is quanval introduced in __init__ (default 0.95).
- `plot`: Plot the location, scale and shape parameter and also the PP-plot and QQ-plot. If `return_plot = True` plot the return period plot if no covariates and trends added.
- `QQplot`: QQ plot
- `_Zstandardt`: Calculates the standardized variable corresponding to the given parameters.
- `_Dzweibull`: Calculates the derivatives of the standardized maximum with respect to parameters
- `_Dmupsiepst`: Calculates the derivatives of the standardized maximum with respect to parameters
- `_DQuantile`: Calculates the quantile derivative associated with a given parameterization with respect model parameters
- `PPplot`: PP plot
- `_CDFGEVt`: Calculates the GEV distribution function corresponding to the given parameters.
- `ReturnPeriodPlot`: Function to plot the Aggregated Return period plot for each month and if annualplot, the annual Return period (default True)
- `_aggquantile`: Function to compute the aggregated quantile for certain parameters
- `_fzeroquanint`: Function to solve the quantile.
- `_fzeroderiquanint`: Function to solve the quantile
- `_ConfidInterQuanAggregate`: Auxiliar function to compute the std for the aggregated quantiles