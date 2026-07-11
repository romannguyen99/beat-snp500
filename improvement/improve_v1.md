# Why was K-Mean model beat LightGBM? (and how we improve it)

We have two 2 models that each pick 10 stocks a month, trying to beat the stock market. One model uses a fancier AI model (LightGBM), one uses a simpler grouping technique (K-mean). At first, K-mean model beat LightGBM model by a lot. We investigated why, ruled out the obvious explanation, found a real bug hiding underneath, fixed it, and the simpler program's results got meaningfully better and more stable as a result.

## Some background: what these two programs actually do

The project picks 10 stocks out of the S&P 500 every month, trying to earn more money than you'd get by just owning the whole S&P 500 index. There are two approaches running side by side, so we can compare them:

- **LightGBM**: It's a machine-learning model that studies 14 features about each stock and tries to *predict* which stocks will do best next month.
- **K-mean**: It doesn't try to predict anything. Each month it just groups all the stocks into 4 clusters based on those same 14 features, picks whichever cluster looks the most "trending upward," and buys the top stocks in that group.

You'd expect the prediction-based approach to win. Instead, K-mean was beating LightGBM by a wide margin, and even beating the S&P 500 itself on returns, though not on a risk-adjusted basis. That was the mystery we set out to explain.

## Step 1: Is the smart model just "cramming" instead of learning?

The first suspicion with any model is  **overfitting**. Meaning, it memorizes the training data instead of learning the underlying patterns. If this were the case, a simpler model should have done meaningfully better.

However, it didn't. The simple method landed at the same (poor) result as the complex AI model, doing better in about half the months and worse in the other half like a coin flip. That told us the AI model's complexity wasn't the problem.

## Step 2: Maybe the ingredients themselves aren't useful

The next suspect is the 14 features both were working from.

We checked each of the 14 features on its own: how well does recent price trend, or volatility, or market sensitivity, predict next month's winners? The answer, across the board, was: barely at all. None of the 14 features, by itself, carried a meaningful, reliable signal for picking next month's winning stocks among these 150 large companies.

**Conclusion:** It's not broken or badly built. It's doing a reasonable job with the information it's been given, that information just doesn't contain much predictive power one month at a time, for this group of very large, closely-watched companies. This is a genuinely hard problem (large, well-known companies are heavily scrutinized by professional investors, so easy patterns tend to get traded away quickly). Improving Champion further would mean giving it better features — which is a bigger undertaking for another time.

## Step 3: A real bug turns up in the simple model

While digging into why the simple grouping approach was doing so well, we found something that didn't add up: in some months, its "trending" group of stocks was surprisingly tiny — sometimes just a single stock.

Here's the problem: the program was designed to buy 10 stocks and split the money evenly between them. But there was no safeguard checking that the "trending group" actually contained 10 stocks. When the group shrank to 1 or 2, the program still went ahead and put the **entire portfolio into that handful of stocks** for the month — effectively turning a diversified 10-stock investment into a single, concentrated bet, at random, whenever this happened.

Checking the full history, this happened in roughly **1 out of every 4 months** — including several stretches where the program was 100% invested in one single stock (at different points, a metals company called TIE, and later Netflix, which happened to triple in value in 2013 while the program was sitting in it).

**Why is that a problem, even if some of those bets happened to pay off?** Putting all your money in one stock is far riskier than spreading it across 10 — one bad month in that single stock can hurt a lot more than one bad stock out of ten. It also meant the program was constantly swinging all-in and all-out of these narrow bets, incurring the trading costs of doing so, month after month.

## The fix

We added a simple safety check: **if the "trending group" doesn't have at least 10 stocks in it, don't force a bet — just keep whatever the program was already holding from before.** This mirrors a safeguard the "smart" Champion model already had, which Challenger was missing.

Think of it like a rule for a fund manager: "if you can't find 10 good trending stocks this month, don't dump your whole portfolio into the 2 you did find — just hold what you've got and try again next month."

## The result

We reran the full 16-year history with the fix in place and compared it to the S&P 500 and to the old (buggy) version:

| | Yearly average return | Return per unit of risk taken | Worst peak-to-valley loss |
|---|---|---|---|
| Challenger — **before fix** | 21.9% | 0.76 | −57% |
| Challenger — **after fix** | **25.6%** | **0.92** | **−38%** |
| S&P 500 (the benchmark) | 14.7% | 0.82 | −34% |

*("Return per unit of risk taken" — often called the Sharpe ratio — is a way of asking not just "how much did it make?" but "how much did it make relative to how bumpy and stressful the ride was?" Higher is better. "Worst peak-to-valley loss" is the biggest drop your money would have taken at any point — a measure of how bad the worst moment would have felt.)*

The fix didn't just remove a risky flaw — it made the strategy better on every measure we track: it earned more, it earned that money more smoothly, and its worst-case losses shrank substantially. After the fix, it now beats the S&P 500 both on total return *and* on a risk-adjusted basis, which it hadn't managed before.

## Where things stand now

- **Challenger (the simple, grouping-based strategy):** genuinely improved by a real bug fix — not just a lucky backtest number, but a program that now behaves the way it was always supposed to.
- **Champion (the smart AI model):** still underperforming, but for an understandable reason — the underlying stock characteristics it's working from don't currently carry enough predictive signal one month at a time. Fixing this would mean giving it new, richer information to learn from (for example, company financial fundamentals it doesn't currently see), which is a larger follow-up project rather than a quick fix.
- We also noticed that the "trending" signal seems to work noticeably better when it's judged *within* a group of similar stocks rather than across the entire market at once — an interesting thread for future improvement that hasn't been acted on yet.
