# METWOBA: An Advanced Offensive Metric for Major League Baseball

**METWOBA** is a comprehensive offensive statistic developed for Major League Baseball, inspired by the wOBA metric originally created by Fangraphs. For more details on the original wOBA, please visit [Fangraphs' wOBA page](https://library.fangraphs.com/offense/woba/). To see the current METWOBA leaderboard, check [here](data/metwOBAlb.csv).

Data is obtained from [Baseball Savant](https://baseballsavant.mlb.com/), credit to pybaseball for assisting with data scraping. 

## Calculation of METWOBA

METWOBA is calculated using the following formula:

METWOBA = (BB_coef * BB + HBP_coef * HBP + 1B_coef * 1B * RC + 2B_coef * 2B * RC + 3B_coef * 3B * RC + HR_coef * HR * RC) / PA

Each term in the numerator represents a "successful" outcome in baseball where the batter reaches base safely. The coefficients for each outcome (BB_coef, HBP_coef, etc.) are determined as follows:

1. **Grouping by Event**: First, each event type (like walks, hits, etc.) is grouped to find the average change in run expectancy for each event.
2. **Rescaling**: These average changes are then rescaled so that the average across all events is 1.

You can find the current coefficients [here](data/outcome_coefs.csv).

## Run Coefficient (RC)

**RC** or the "run coefficient" is a key component in calculating METWOBA. The desire here is to add an "outcomeless" proponent of the statistic, to attempt to reward hard contact outs over bloop hits. It is derived using a Random Forest machine learning model that takes exit velocity and launch angle as inputs to predict changes in run expectancy. These predictions are rescaled as follows:

- **For predictions less than zero**: A logistic function is applied to map the value to the range [0, 1].
- **For predictions greater than zero**: An exponential transformation is used (`e^p + 1`) to ensure the value is greater than 1.

## Use of Run Expectancy
Baseball Savant's run expectancy is used to evaluate plays as we hope to reflect the value of timing and 'clutchness' in players. A three-run home run produces more value than a solo home run, and through using run expectancy we hope to account for this aspect of baseball.

## Final Computation

After determining the RC and the outcome coefficients, these values are multiplied together to compute a total score contribution for each successful outcome. The sum of each player's score contributions is then divided by their total plate appearances to compute the METWOBA value. Currently the leaderboard filters to plate appearances above the 30% quantile of overall plate appearances. 

## Future Works
Further works are mainly focused on optimizing the outcome and run coefficients, specifically on designing the scaling so that the statistic is easily interpretable. Currently the scale of metwOBA does not follow any other major statistics, and therefore may be difficult to interpret to the viewer. I also hope to develop a baseball
savant data scraper in order to update the data daily automatically. 

