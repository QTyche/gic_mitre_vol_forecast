import random

import numpy as np

from qtyche_qrc.seed import derive_seed, set_global_seed


def test_seed_initialization_is_deterministic() -> None:
    first_generator = set_global_seed(37)
    first = (random.random(), np.random.random(), first_generator.random())

    second_generator = set_global_seed(37)
    second = (random.random(), np.random.random(), second_generator.random())

    assert first == second
    assert derive_seed(37, "simulator") == derive_seed(37, "simulator")
    assert derive_seed(37, "simulator") != derive_seed(37, "readout")
