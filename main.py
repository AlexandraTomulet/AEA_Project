from src.data.sample_instance import create_sample_instance
from src.strategy1.solver import Strategy1Solver
from src.evaluation.metrics import route_cost
from src.utils.plotting import plot_routes


def main() -> None:
    instance = create_sample_instance()
    solver = Strategy1Solver()
    result = solver.solve(instance)

    print(f"Instanță: {instance.name}")
    print(f"Număr noduri: {instance.num_nodes}")
    print(f"Număr salesmeni: {instance.num_salesmen}")
    print(f"Soluție validă: {result.is_valid}")

    if result.validation_errors:
        print("Erori validare:")
        for error in result.validation_errors:
            print(f" - {error}")

    print()

    for idx, route in enumerate(result.routes, start=1):
        cost = route_cost(route, instance.distance_matrix)
        print(f"Ruta salesman {idx}: {route}")
        print(f"Cost ruta {idx}: {cost:.2f}")
        print()

    if result.is_valid:
        print(f"Cost total: {result.total_cost:.2f}")
        print(f"Cea mai lungă rută: {result.longest_route_cost:.2f}")

        plot_routes(
            instance=instance,
            routes=result.routes,
            title="Strategia 1 - Sample Instance",
            save_path="results/plots/sample_strategy1.png"
        )


if __name__ == "__main__":
    main()