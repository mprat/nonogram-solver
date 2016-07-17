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
    nonogram.init_from_solution_string(string_box)

    assert len(nonogram.rows_constraints) == 5
    assert len(nonogram.cols_constraints) == 5
    assert nonogram.rows_constraints[0] == [3]
    assert nonogram.rows_constraints[1:] == [[], [], [], []]
    assert nonogram.cols_constraints[0] == []
    assert nonogram.cols_constraints[1:4] == [[1], [1], [1]]
    assert nonogram.cols_constraints[4] == []
    assert sorted(nonogram.solution_list) == [
        (0, 1), (0, 2), (0, 3)]
    # assert nonogram.solution_list == nonogram.hint_eligible


def test_nonogram_solve():
    string_box = [
        'o..o.',
        '..o.o',
        '..oo.',
        '.o..o',
        'o...o']

    nonogram = Nonogram()
    nonogram.init_from_solution_string(string_box)
    puzzle_constraints = \
        solver.generate_puzzle_constraints(nonogram)

    print puzzle_constraints
