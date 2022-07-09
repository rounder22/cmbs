# Hedging

This application analyzes the relationships between different market indices 
to help determine hedge ratios between different securities.

## Usage
Select an Independent(the hedge) series and a Dependent(the security to be hedged) series to chart and run simple
linear regression analysis over different time periods.

Statistics for each series over the entire dataset is shown to the left. If the series have different lengths of history, only the 
time periods where there are values for both series is considered.

After selecting the two data series, select if you want to use the change or percentage change of each series in the analysis. 
If neither percentage change or change is selected the price levels of the indices are used in all analysis except for the 
hedge ratio analysis.

The Rolling Correlations show the correlation coefficients over different time periods and the distribution of correlation 
coefficients over the entire dataset. This shows how stable the relationship between the two series is over different time periods.
There is also a table showing the rolling standard deviations for each series over the time periods entered.

The Rolling Hedge Ratios shows the optimal hedge ratio for the time periods selected over the history of the dataset. The tables with
the statistics and the distributions shows how stable the hedge ratio is for the time period over the history of the dataset.
The hedge ratio can be multiplied by the notional amount of the security you own to determine how much of the hedge to sell.

## Uploading New Series Data
Using the sidebar on the left, you can download an input template. You can see the last available data for any series by zooming into
the graph to the 1m period and looking to the right using the hover tool to see the last date available. 

Copy and paste new data into the input template, Data and Dates go horizontal.

After uploading the data, clear the cache by selecting the three lines in the upper right corner of the screen and re-run the app.
