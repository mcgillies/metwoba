METWOBA is an overarching offensive statistic for Major League Baseball. Computations are inspired by wOBA created by Fangraphs (https://library.fangraphs.com/offense/woba/). See the current leaderboard [here](data/metwOBAlb.csv)

We calculate metwOBA utilizing the formula: (BB_coef*BB + HBP_coef*HBP + 1B_coef*1B*RC + 2B_coef*2B*RC + 3B_coef*3B*RC + HR_coef*HR*RC)/PA, where each of the items in the numerator represents a "successful" outcome in baseball (where the batter reaches base). 

Where BB_coef, HBP_coef, etc. are calculated in the following way: 
- First group each event and find the average change in run expectancy for each event
- Then rescale the average changes in run expectancy so that the average throughout is 1.
  The current coefficients are [here](data/outcome_coefs.csv)

RC represents what I have coined the "run coefficient". This is calculated using a Random Forest machine learning model, where exit velocity and launch angle are used as inputs to predict the change in run expectancy. These are then rescaled using a piecewise function, where if the prediction of run coefficient is less than zero the logistic function is mapped to a value in [0,1], and if the prediction is greater than one an exponential transformation is applied (e^p + 1) to ensure the value is greater than 1. 

Once RC and the outcome coefficients are calculated, these are multiplied together to compute a total score contribution for each successful outcome. Then we simply calculate the sum of each player's score contribution, and divide it by their total plate appearances to compute metwOBA


