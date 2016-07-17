from nonogram_solver import solver
from nonogram_solver.nonogram import Nonogram


def test_nonogram_init():
    string_box = [
        '.ooo.',
        '.....',
        '.....',
        '.....',
        '.....']

    nonogram = Nonogram()
    nonogram.init_from_string_box(string_box)

    assert len(nonogram.rows_constraints) == 5
    assert len(nonogram.cols_constraints) == 5
    assert nonogram.rows_constraints[0] == [3]
    assert all(nonogram.rows_constraints[1:] == [])
    assert nonogram.cols_constraints[0] == []
    assert nonogram.cols_constraints[1:4] == [1]
    assert nonogram.cols_constraints[4] == []


def test_nonogram_solve():
    string_box = [
        'o..o.',
        '..o.o',
        '..oo.',
        '.o..o',
        'o...o']

    nonogram = Nonogram()
    nonogram.init_from_string_box(string_box)
    puzzle_constraints = \
        solver.generate_puzzle_constraints(nonogram)

    print puzzle_constraints
