import pytest
from nonogram_solver import solver
from nonogram_solver.nonogram import Nonogram
import numpy as np


@pytest.fixture
def simple_nonogram_from_string_box():
    string_box = [
        'o..o.',
        '..o.o',
        '..oo.',
        '.o..o',
        'o...o']
    
    nonogram = Nonogram()
    nonogram.init_from_solution_string(string_box)
    return nonogram


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


def test_nonogram_solve(simple_nonogram_from_string_box):
    nonogram = simple_nonogram_from_string_box
    solveable, _ = solver.solve(nonogram, add_puzzle_constraints=False)
    assert solveable is False

    solveable, nonogram_solver = solver.solve(
        nonogram, add_puzzle_constraints=True)
    assert solveable is True
    assert np.all(nonogram_solver.puzzle_state == nonogram.solution_state)


def test_nonogram_solver_manual(simple_nonogram_from_string_box):
    nonogram = simple_nonogram_from_string_box
    nonogram_solver = solver.NonogramSolver(nonogram)
    nonogram_solver._generate_solutions()
    assert nonogram_solver._puzzle_is_solved() is False
    assert np.all(nonogram_solver.puzzle_state == np.array(
        [[-1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1],
         [-1, -1, -1, 0, 1],
         [-1, -1, -1, -1, -1]]))

    nonogram_solver._pick_help_square(position=(2, 2))
    assert (2, 2) not in nonogram_solver.filled_positions_hint_eligible
    nonogram_solver._generate_solutions()
    assert nonogram_solver._puzzle_is_solved() is False
    assert np.all(nonogram_solver.puzzle_state == np.array(
        [[-1, -1, 0, 1, 0],
         [-1, -1, -1, 0, 1],
         [0, 0, 1, 1, 0],
         [-1, -1, -1, 0, 1],
         [-1, -1, 0, 0, 1]]))


def svgGen():
    nonogram = Nonogram()
    nonogram.init_from_json("rose_nonogram.json")
    nonogram.display_puzzle_svg()
svgGen()