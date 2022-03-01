import requests, json, math, functools, datetime
from scipy import optimize

TIME_AVG = True
FRAC_TO_END = 0.5
TAG = 'RussiaUkraine'
SCORE_FUNC = 'log'
RESOLVE_BY = datetime.datetime.fromisoformat('2022-03-01T00:00:01-08:00')
# or None if you don't want to put a time limit on when markets should have
# resolved

assert SCORE_FUNC in ['log', 'brier']

if TIME_AVG:
    print("Analyzing time-averaged probability of each market.")
else:
    print(f"Analyzing probability of each market {FRAC_TO_END * 100}% of the"
          + " way thru its life.")

print("Analysis will use the " + SCORE_FUNC + " score.")

url = requests.get("https://manifold.markets/api/v0/markets")
text = url.text
data = json.loads(text)
resolved_markets = []
for market in data:
    if TAG in market['tags'] and market['isResolved']:
        resolved_markets.append(market)


def and_func(x, y):
    return x and y


def log_score(prob, outcome):
    prob_outcome = prob if outcome == 'YES' else 1 - prob
    return math.log2(prob_outcome)


def brier_score(prob, outcome):
    prob_outcome = prob if outcome == 'YES' else 1 - prob
    return (1 - prob_outcome)**2

    
scores = []
is_yes = []
for market in resolved_markets:
    market_id = market['id']
    market_url = "https://manifold.markets/api/v0/market/" + market_id
    r = requests.get(market_url)
    text = r.text
    market_data = json.loads(text)
    open_time = market_data['createdTime']
    close_time = (market_data['closeTime']
                  if 'closeTime' in market_data else None)
    resolve_time = (market_data['resolutionTime']
                    if 'resolutionTime' in market_data else None)
    if close_time is not None and resolve_time is not None:
        end_time = min(close_time, resolve_time)
    elif close_time is not None:
        end_time = close_time
    elif resolve_time is not None:
        end_time = resolve_time
    else:
        print(market_data)
        assert False, "WTF no close or resolve time?"
    if RESOLVE_BY is not None:
        resolve_by_timestamp = RESOLVE_BY.timestamp() * 1000
        if end_time > resolve_by_timestamp:
            # only look at markets that have resolved before our set time.
            continue
    market_bets = market_data['bets']
    if TIME_AVG:
        open_length = end_time - open_time
        num_bets = len(market_bets)
        market_prob = 0
        total_bet_weight = 0
        for j, bet in enumerate(market_bets):
            bet_time = bet['createdTime']
            next_bet_time = (market_bets[j+1]['createdTime']
                             if j != num_bets - 1 else end_time)
            bet_weight = (next_bet_time - bet_time) / open_length
            total_bet_weight += bet_weight
            market_prob += bet['probAfter'] * bet_weight
        market_prob /= total_bet_weight
        assert 0.98 < total_bet_weight < 1.02
    else:
        test_point = open_time * (1 - FRAC_TO_END) + end_time * FRAC_TO_END
        early_bets = list(filter(lambda bet: bet['createdTime'] < test_point,
                                 market_bets))
        halfway_bet = early_bets[-1]
        market_prob = halfway_bet['probAfter']
    market_outcome = market['resolution']
    are_bets_binary = map(lambda bet: bet['outcome'] in ['YES', 'NO'],
                          market_bets)
    is_binary = functools.reduce(and_func, are_bets_binary, True)
    defined_outcome = market_outcome in ['YES', 'NO']
    if is_binary and defined_outcome:
        if SCORE_FUNC == 'log':
            scores.append(log_score(market_prob, market_outcome))
        elif SCORE_FUNC == 'brier':
            scores.append(brier_score(market_prob, market_outcome))
        if market_outcome == 'YES':
            is_yes.append(1)
        else:
            is_yes.append(0)

average_score = sum(scores) / len(scores)
frac_yes = sum(is_yes) / len(is_yes)
print("Number of markets analyzed:", len(scores))
print("Average score:", average_score)
print("Proportion of markets that resolved 'yes'", frac_yes)


def oracle_log_score(p):
    return p * math.log2(p) + (1-p) * math.log2(1-p)


def oracle_brier_score(p):
    return p * (1 - p)**2 + (1-p) * p**2

f = oracle_log_score if SCORE_FUNC == 'log' else oracle_brier_score
worst_oracle_val = f(0.5)
fail_string = "On these questions, Manifold was worse than a coin-flip."
if SCORE_FUNC == 'log' and average_score < worst_oracle_val:
    print(fail_string)
elif SCORE_FUNC == 'brier' and average_score > worst_oracle_val:
    print(fail_string)
else:
    oracle_prob_sol = optimize.root_scalar(lambda p: f(p) - average_score,
                                           bracket=[0.501, 0.999],
                                           method='brentq')
    oracle_prob = oracle_prob_sol.root
    oracle_score = f(oracle_prob)
    print("This is as good as an oracle that knows the answer with probability",
          oracle_prob)
    print("Just checking: that oracle would get an average score of",
          oracle_score)
