from nonogram_solver import solver


def test_nonogram_simple():
    braille_box = [
        '.ooo.',
        '.....',
        '.....',
        '.....',
        '.....']

    nonogram = solver.Nonogram(braille_box)
    puzzle_constraints = \
        solver.generate_puzzle_constraints(nonogram)

    print puzzle_constraints


# braille_box = ['.ooo...o', '.o..o..', '..oo....', '......oo', '.o...o..']
braille_box = [
    'o..o.',
    '..o.o',
    '..oo.',
    '.o..o',
    'o...o']

nonogram = solver.Nonogram(braille_box)
print nonogram.rows_constraints
print nonogram.cols_constraints
puzzle_constraints = \
    solver.generate_puzzle_constraints(nonogram)

print puzzle_constraints
