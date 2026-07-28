import argparse
from cfr_poker.games.kuhn import Kuhn
from cfr_poker.cfr.trainer import CFRTrainer

def main():
    parser = argparse.ArgumentParser(description="Train CFR on Kuhn poker.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000000,
        help="Number of CFR iterations to run.",
    )
    args = parser.parse_args()
    
    trainer = CFRTrainer(Kuhn())
    avg_game_value = trainer.train(args.iterations)
    avg_strategies = trainer.get_average_strategies()

    print(f"Average game value: {avg_game_value}")

    for key in sorted(avg_strategies):
        print(f"{key}: {avg_strategies[key]}")


if __name__ == "__main__":
    main()