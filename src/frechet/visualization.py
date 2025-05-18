import matplotlib.pyplot as plt
from typing import List, Tuple

# Visualization functions
def plot_frechet(curve_a: Curve, curve_b: Curve, path: List[Tuple[int, int]], title: str):
    fig, ax = plt.subplots()
    ax.plot(*zip(*curve_a), label='Curve A', marker='o')
    ax.plot(*zip(*curve_b), label='Curve B', marker='x')

    for (i, j) in path:
        ax.plot([curve_a[i][0], curve_b[j][0]], [curve_a[i][1], curve_b[j][1]], 'k--', alpha=0.3)

    ax.legend()
    ax.set_title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

# Example usage
def main():
    curve_a = [(1, 1), (2, 2), (3, 3), (4, 4)]
    curve_b = [(1, 2), (2, 3), (3, 4), (4, 5)]

    discrete_dist = discrete_frechet_distance(curve_a, curve_b)
    greedy_dist, greedy_path = greedy_frechet_distance(curve_a, curve_b)

    print(f"Discrete Frechet Distance: {discrete_dist}")
    print(f"Greedy Frechet Distance: {greedy_dist}")

    plot_frechet(curve_a, curve_b, greedy_path, f'Greedy Frechet Distance: {greedy_dist:.2f}')

if __name__ == '__main__':
    main()
