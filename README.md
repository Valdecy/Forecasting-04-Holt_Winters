# Forecasting-04-Holt_Winters

This Function is an Implementation of the Holt-Winters' Method for Time Series with Trend and Seasonality. If Necessary it Can Also Return the Best values for Alpha, Beta and Gama.

* timeseries = The dataset in a Time Series format.

* alpha = Level smoothing parameter. The default value is 0.2

* beta = Trend smoothing parameter. The default value is 0.1

* Gama = Seasonal smoothing parameter. The default value is 0.2

* graph = If True then the original dataset and the moving average curves will be plotted. The default value is True.

* horizon = Calculates the prediction h steps ahead. The default value is 0.

* trend = Indicates the types of trend: "additive", "multiplicative" or "none". The default value is "multiplicative".

* seasonality = Indicates the types of seasonality: "additive", "multiplicative" or "none". The default value is "multiplicative".

Finally a brute force optimization can be done by calling the "optimize_holt" function.
