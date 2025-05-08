import pandas as pd
import matplotlib.pyplot as plt
from backtest import allocate, load_data, build_snapshots, ORDER_SIZE

# Set best parameters
λo = 0.01
λu = 0.01
θ = 0.001

def simulate_with_tracking(snapshots, λo, λu, θ):
    remaining = ORDER_SIZE
    cash_spent = 0
    cumulative_cost = []
    for venues in snapshots:
        if remaining <= 0:
            break
        alloc, _ = allocate(remaining, venues, λo, λu, θ)
        for i, shares in enumerate(alloc):
            fill = min(shares, venues[i]['ask_size'])
            cost = fill * (venues[i]['ask'] + venues[i]['fee'])
            cash_spent += cost
            remaining -= fill
        cumulative_cost.append(cash_spent)
    return cumulative_cost

if __name__ == '__main__':
    df = load_data('l1_day.csv')
    snapshots = build_snapshots(df)
    cumulative_cost = simulate_with_tracking(snapshots, λo, λu, θ)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_cost, label='Smart Router (Tuned)', color='blue')
    plt.title('Cumulative Execution Cost Over Time')
    plt.xlabel('Snapshot Index')
    plt.ylabel('Cumulative Cost (USD)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results.png')
    print("Saved results.png")
