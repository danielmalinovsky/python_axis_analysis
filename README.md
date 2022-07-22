# python_axis_analysis
Time series model focusing on dynamic time steps

# Motivation
Goal of the script was to test hhypothesis if it is to model time series based on aggregation of short term trends.
Prices are usually moving in momentum. This means that price of an asset can increase two, three or four days in row. Therefore is is unlikely for probability of growth or decline to be 50% each day.
By clustering time series data into random vector (including two infromation: price change in percentage terms, number of days), and assigning this random vector to time series with steps going from one end of trend change to another trend change, we create new time series
This time series then could be modelled like any ordinary random vector time series by estimating each value in vector. 
Result of such model then should be a time period (number of following days in which growth or decline will appear), and price change in percentage that will in given period occur.
This should give us answers to questions "how long" and "how much" will price increase or decrease.

# Findings
Due to the discrete nature of the day values, it is not possible to generate day values from normal distribution. 
All of the autoregressive function models failed jarque berra normality test of residuals, which means that BLUE conditions were not met, thus disvalidating prediction
By separating dataset to positive and negative values proble with modelling percentage price change appeared. This problem has to do that normal distribution assumes symetricity, however within the first deviation values can reach opposite sign (i.e. values from positive dataframe can turn negative and vice versa).
This causes issue with interpreting probability since symetricity of normal distribution is violated.

# Conclusion
This is a long overdue project of personal meaning. Price modelling was greatest motivation of my interest in statistics and financial engineering.
Although this adjustmend of data offers no added value, I am greatful of the lessons I learned along the way.
Therefore with before listed issues, I have to conclude that this approach to time series modeling   S H O U L D   N O T   be used in any financial decisions.

## Functions

### yields_preprocessing
Function that is adding collumns to datafram containing percentage price changes of underlying asset.
Function can be toggled to output clustered dataframe, where percentage price changes are calculated base on first and last day of growing or declining streak
i.e. if price is going up for three days percentage change is calculated from the open price of first day and close price of third days
for increased volatility values are being selected from lows on declining day and highs from growing day

parameters: data, clustering = False
- data: pandas dataframe - data input into function
- clustering: boolean - triggering clustering output

output: pandas dataframe

### AR_prediction
Function is generating prediction with autoregresive function and it's diagnostics.

parameters: data, ar_lags = 1, split_train_ration = 0.8, name = ''
- data: pandas core series - time series data input
- ar_lags: int - number of lags that should autoregresive function work with
- split_train_ration: float - percentage of training data
- name: string - name of the run

output: pandas dataframe including information about 

dataframe includes information about:
- mean
- standard deviation
- upper bound mean + standard deviation
- upper bound mean + 2 * standard deviation
- upper bound mean + 3 * standard deviation
- residual durbin watson (autocorelation)
- p-values of residual white's test (homoskedasticity)
- p-values of residual jarque-berra test (normality)
- akaike information criterion
