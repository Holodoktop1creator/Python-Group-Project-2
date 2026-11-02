"""Небесный курьер — симулятор управления беспилотником.

Запуск:  python main.py            (случайный город)
         python main.py --seed 42  (один и тот же город)
"""

import argparse

from drone_sim.game import Game


def main() -> None:
    parser = argparse.ArgumentParser(description="Симулятор дрона-доставщика")
    parser.add_argument("--seed", type=int, default=None, help="зерно генерации города")
    args = parser.parse_args()
    Game(seed=args.seed).run()


if __name__ == "__main__":
    main()
