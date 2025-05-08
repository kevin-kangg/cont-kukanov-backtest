# Smart Order Router Backtest (Cont & Kukanov Model)

## Overview
This project implements a static Smart Order Router based on the cost model from *Cont & Kukanov: Optimal Order Placement in Limit Order Markets*. The goal is to minimize the execution cost of a 5,000-share buy order by splitting it intelligently across multiple venues. The model is tuned over three risk parameters:

- `lambda_over`: penalty for overfilling  
- `lambda_under`: penalty for underfilling  
- `theta_queue`: penalty for queue position risk  

Performance is evaluated on a 9-minute stream of mocked Level-1 data and benchmarked against three baselines:
- Best-ask (always routes to lowest price)
- TWAP (time-weighted average price)
- VWAP (volume-weighted average price)

## Project Files
- `backtest.py`: Core script that implements the allocator, simulates execution, performs parameter tuning, and prints a final JSON summary.
- `plot_results.py`: Optional script that generates `results.png`, showing cumulative execution cost over time.
- `results.png`: Visualization of cumulative cost for the best run.
- `README.md`: This file.

## Parameter Search
Grid search was conducted over:
- `lambda_over`: {0.01, 0.02}
- `lambda_under`: {0.01, 0.02}
- `theta_queue`: {0.001, 0.005}

## Suggested Improvement
To better model real-world execution, queue position could be explicitly tracked. Estimating fill probability based on order depth and venue outflow would allow more realistic simulation of partial or delayed fills, especially in high-volume or thin markets.

### Best Result
```json
{ "lambda_over": 0.01, "lambda_under": 0.01, "theta_queue": 0.001 }

