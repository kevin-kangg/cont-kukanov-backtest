import pandas as pd
import numpy as np
import json
from itertools import product

# Constants
ORDER_SIZE = 5000
STEP = 100
FEE = 0.003  # placeholder fee
REBATE = 0.002  # placeholder rebate

def allocate(order_size, venues, λ_over, λ_under, θ_queue):
    splits = [[]]
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v]['ask_size'])
            for q in range(0, max_v + 1, STEP):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float('inf')
    best_split = []
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, λ_over, λ_under, θ_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    return best_split, best_cost

def compute_cost(split, venues, order_size, λo, λu, θ):
    executed = 0
    cash_spent = 0
    for i in range(len(venues)):
        exe = min(split[i], venues[i]['ask_size'])
        executed += exe
        cash_spent += exe * (venues[i]['ask'] + venues[i]['fee'])
        maker_rebate = max(split[i] - exe, 0) * venues[i]['rebate']
        cash_spent -= maker_rebate
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = θ * (underfill + overfill)
    cost_pen = λu * underfill + λo * overfill
    return cash_spent + risk_pen + cost_pen

def load_data(filename):
    df = pd.read_csv(filename)
    df = df.sort_values('ts_event')
    df = df.drop_duplicates(['ts_event', 'publisher_id'], keep='first')
    return df

def build_snapshots(df):
    grouped = df.groupby('ts_event')
    snapshots = []
    for _, group in grouped:
        venues = []
        for _, row in group.iterrows():
            venues.append({
                'ask': row['ask_px_00'],
                'ask_size': row['ask_sz_00'],
                'fee': FEE,
                'rebate': REBATE,
                'id': row['publisher_id']
            })
        snapshots.append(venues)
    return snapshots

def simulate(snapshots, λ_over, λ_under, θ_queue):
    remaining = ORDER_SIZE
    cash_spent = 0
    for venues in snapshots:
        if remaining <= 0:
            break
        best_split, _ = allocate(remaining, venues, λ_over, λ_under, θ_queue)
        filled = 0
        for i, shares in enumerate(best_split):
            fill = min(shares, venues[i]['ask_size'])
            cash_spent += fill * (venues[i]['ask'] + venues[i]['fee'])
            remaining -= fill
            filled += fill
    avg_price = cash_spent / (ORDER_SIZE - remaining) if ORDER_SIZE - remaining > 0 else 0
    return cash_spent, avg_price

def best_ask_baseline(snapshots):
    remaining = ORDER_SIZE
    cost = 0
    for venues in snapshots:
        if remaining <= 0:
            break
        best = min(venues, key=lambda x: x['ask'])
        fill = min(best['ask_size'], remaining)
        cost += fill * (best['ask'] + best['fee'])
        remaining -= fill
    return cost, cost / ORDER_SIZE

def twap_baseline(snapshots):
    chunks = np.array_split(np.arange(ORDER_SIZE), len(snapshots))
    remaining = ORDER_SIZE
    cost = 0
    for i, venues in enumerate(snapshots):
        qty = len(chunks[i])
        if qty == 0: continue
        best = min(venues, key=lambda x: x['ask'])
        fill = min(best['ask_size'], qty)
        cost += fill * (best['ask'] + best['fee'])
        remaining -= fill
    return cost, cost / ORDER_SIZE

def vwap_baseline(snapshots):
    remaining = ORDER_SIZE
    cost = 0
    for venues in snapshots:
        total_sz = sum(v['ask_size'] for v in venues)
        if total_sz == 0 or remaining <= 0:
            continue
        for v in venues:
            w = v['ask_size'] / total_sz if total_sz > 0 else 0
            qty = int(w * ORDER_SIZE)
            fill = min(qty, v['ask_size'], remaining)
            cost += fill * (v['ask'] + v['fee'])
            remaining -= fill
    return cost, cost / ORDER_SIZE

def main():
    df = load_data('l1_day.csv')
    snapshots = build_snapshots(df)

    param_grid = list(product([0.01, 0.02], [0.01, 0.02], [0.001, 0.005]))
    best = {'cost': float('inf')}

    for λo, λu, θ in param_grid:
        cost, avg_px = simulate(snapshots, λo, λu, θ)
        if cost < best['cost']:
            best.update({
                'cost': cost,
                'avg_px': avg_px,
                'λo': λo,
                'λu': λu,
                'θ': θ
            })

    # Baselines
    best_ask_cost, best_ask_px = best_ask_baseline(snapshots)
    twap_cost, twap_px = twap_baseline(snapshots)
    vwap_cost, vwap_px = vwap_baseline(snapshots)

    def savings(base): return round((base - best['cost']) / base * 10000, 2)

    print(json.dumps({
        "best_params": {"lambda_over": best['λo'], "lambda_under": best['λu'], "theta_queue": best['θ']},
        "optimal_cost": round(best['cost'], 2),
        "optimal_avg_px": round(best['avg_px'], 5),
        "best_ask_cost": round(best_ask_cost, 2),
        "best_ask_avg_px": round(best_ask_px, 5),
        "twap_cost": round(twap_cost, 2),
        "twap_avg_px": round(twap_px, 5),
        "vwap_cost": round(vwap_cost, 2),
        "vwap_avg_px": round(vwap_px, 5),
        "savings_vs_best_ask_bps": savings(best_ask_cost),
        "savings_vs_twap_bps": savings(twap_cost),
        "savings_vs_vwap_bps": savings(vwap_cost)
    }, indent=2))

if __name__ == '__main__':
    main()
