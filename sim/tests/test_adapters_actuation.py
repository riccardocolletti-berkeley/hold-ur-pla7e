"""Tests for sim.adapters.actuation."""

import math

from sim.adapters.actuation import alpha_from_tau


def test_alpha_from_tau_matches_first_order_discretization():
    assert math.isclose(alpha_from_tau(0.002, 0.15), 1.0 - math.exp(-0.002 / 0.15))


def test_alpha_from_tau_zero_tau_is_immediate():
    assert alpha_from_tau(0.002, 0.0) == 1.0
