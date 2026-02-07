"""Entry point: load config, build solver, run simulation, plot results."""

import sys

from minipbl.config import load_config
from minipbl.solver import Solver
from minipbl.plotting import plot_results


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/cbl_1d.yaml"
    print(f"Loading config from {config_path}")

    cfg = load_config(config_path)
    solver = Solver(cfg)
    output_file = solver.run()

    print("\nGenerating plots...")
    plot_results(output_file, cfg.output.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
