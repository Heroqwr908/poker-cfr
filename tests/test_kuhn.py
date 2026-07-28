"""Verify the trained Kuhn poker strategy against its known analytical equilibrium.

Kuhn poker has a solved Nash equilibrium, which
makes it the standard first target for validating a CFR implementation:
the learned strategy can be checked directly against known-correct values.

These tests train a single seeded model once (module-scoped fixture) and
assert two kinds of property against it:

- The game value to player 0 converges to -1/18.
- The card-by-card betting strategy matches the equilibrium structure --
  e.g. the King value-bets, the Jack bluffs at rate xi and folds to bets,
  the Queen checks and bluff-catches, and the King's opening bet frequency
  is three times the Jack's.

The equilibrium is a family parameterized by xi in [0, 1/3], so xi-dependent
quantities are checked as ranges or relationships rather than fixed points.

Note: this comparison-to-known-solution approach is specific to Kuhn.
Larger games (Leduc) have no analytical solution and are instead validated
by exploitability.
"""

import random
import pytest
from cfr_poker.games.kuhn import Kuhn
from cfr_poker.cfr.trainer import CFRTrainer

@pytest.fixture(scope="module")
def train_model():
    random.seed(42)
    trainer = CFRTrainer(Kuhn())
    avg_game_value = trainer.train(500000)
    return (avg_game_value, trainer.get_average_strategies())

def test_equilibrium_value(train_model):
    avg_game_value = train_model[0]
    margin = 0.005
    assert avg_game_value == pytest.approx(-1/18, abs=margin)
    
def test_jack_open_bluff_within_range(train_model):
    jack_prob = train_model[1]["1d"][1]
    margin = 0.005
    assert -margin <= jack_prob <= 1/3 + margin

def test_queen_never_opens(train_model):
    queen_prob = train_model[1]["2d"][1]
    margin = 0.005
    assert queen_prob == pytest.approx(0, abs=margin)

def test_king_opens_three_times_jack(train_model):
    king_prob = train_model[1]["3d"][1]
    jack_prob = train_model[1]["1d"][1]
    margin = 0.01
    assert king_prob == pytest.approx(jack_prob * 3, abs=margin)

def test_jack_folds_to_bet(train_model):
    jack_prob = train_model[1]["1db"][1]
    margin = 0.005
    assert jack_prob == pytest.approx(0, abs=margin)

def test_queen_calls_bet_one_third(train_model):
    queen_prob = train_model[1]["2db"][1]
    margin = 0.005
    assert queen_prob == pytest.approx(1/3, abs=margin)

def test_king_always_calls_bet(train_model):
    king_prob = train_model[1]["3db"][1]
    margin = 0.005
    assert king_prob == pytest.approx(1, abs=margin)

def test_jack_bluffs_when_checked_to(train_model):
    jack_prob = train_model[1]["1dp"][1]
    margin = 0.005
    assert jack_prob == pytest.approx(1/3, abs=margin)

def test_queen_checks_back(train_model):
    queen_prob = train_model[1]["2dp"][1]
    margin = 0.005
    assert queen_prob == pytest.approx(0, abs=margin)

def test_king_bets_when_checked_to(train_model):
    king_prob = train_model[1]["3dp"][1]
    margin = 0.005
    assert king_prob == pytest.approx(1, abs=margin)

def test_jack_folds_to_bet_after_checking(train_model):
    jack_prob = train_model[1]["1dpb"][1]
    margin = 0.005
    assert jack_prob == pytest.approx(0, abs=margin)

def test_queen_calls_at_jack_bluff_rate_plus_one_third(train_model):
    queen_prob = train_model[1]["2dpb"][1]
    jack_prob = train_model[1]["1d"][1]
    margin = 0.005
    assert queen_prob == pytest.approx(jack_prob + 1/3, abs=margin)

def test_king_always_calls_after_checking(train_model): 
    king_prob = train_model[1]["3dpb"][1]
    margin = 0.005
    assert king_prob == pytest.approx(1, abs=margin)