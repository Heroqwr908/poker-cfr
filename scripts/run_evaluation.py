from cfr_poker.games.kuhn import Kuhn
from cfr_poker.cfr.trainer import CFRTrainer
from cfr_poker.evaluation.best_response import BestResponse

def main():
    trainer = CFRTrainer(Kuhn())
    trainer.train(1000000)
    strategies = trainer.get_average_strategies()
    br = BestResponse(Kuhn())
    print(br.compute_exploitability(strategies))

if __name__ == "__main__":
    main()