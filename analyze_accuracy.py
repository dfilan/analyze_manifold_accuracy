import requests, json, math, functools
from scipy import optimize

TIME_AVG = True
FRAC_TO_END = 0.5
TAG = 'RussiaUkraine'
SCORE_FUNC = 'brier'

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
    prob_outcome = market_prob if outcome == 'YES' else 1 - market_prob
    return math.log2(prob_outcome)


def brier_score(prob, outcome):
    prob_outcome = market_prob if outcome == 'YES' else 1 - market_prob
    return (1 - prob_outcome)**2

    
scores = []
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
        assert False, "WTF no close or resolve time?"
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
        assert total_bet_weight > 0.98 and total_bet_weight < 1.02
    else:
        test_point = open_time * (1 - FRAC_TO_END) + end_time * FRAC_TO_END
        early_bets = list(filter(lambda bet: bet['createdTime'] < test_point,
                                 market_bets))
        halfway_bet = early_bets[-1]
        market_prob = halfway_bet['probAfter']
    outcome = market['resolution']
    are_bets_binary = map(lambda bet: bet['outcome'] in ['YES', 'NO'],
                          market_bets)
    is_binary = functools.reduce(and_func, are_bets_binary, True)
    defined_outcome = outcome in ['YES', 'NO']
    if is_binary and defined_outcome:
        if SCORE_FUNC == 'log':
            scores.append(log_score(market_prob, outcome))
        elif SCORE_FUNC == 'brier':
            scores.append(brier_score(market_prob, outcome))

average_score = sum(scores) / len(scores)
print("Number of markets analyzed:", len(scores))
print("Average score:", average_score)

def oracle_log_score(p):
    return p * math.log2(p) + (1-p) * math.log2(1-p)

def oracle_brier_score(p):
    return p * (1 - p)**2 + (1-p) * p**2

def oracle_log_score_deriv(p):
    return math.log2(p) - math.log2(1-p)

def oracle_brier_score_deriv(p):
    return 1 - 2*p

f = oracle_log_score if SCORE_FUNC == 'log' else oracle_brier_score
fprime = (oracle_log_score_deriv
          if SCORE_FUNC == 'log'
          else oracle_brier_score_deriv)

oracle_prob_sol = optimize.root_scalar(lambda p: f(p) - average_score, x0=0.6,
                                       fprime=fprime, method='newton')
oracle_prob = oracle_prob_sol.root
oracle_score = (oracle_log_score(oracle_prob)
                if SCORE_FUNC == 'log'
                else oracle_brier_score(oracle_prob))
print("This is as good as an oracle that knows the answer with probability",
      oracle_prob)
print("Just checking: that oracle would get an average score of",
      oracle_score)
