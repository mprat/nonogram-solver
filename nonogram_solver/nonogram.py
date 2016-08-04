"""
Defines the nonogram datatype.
"""
import numpy as np


class Nonogram(object):
    def __init__(self, ordered=True):
        self.puzzle_state = None
        self.n_rows = None
        self.n_cols = None
        self.rows_constraints = None
        self.cols_constraints = None
        self.solution_state = None
        self.solution_list = None
        self.ordered = ordered

    def _init_puzzle(self):
        self.puzzle_state = -1 * np.ones((self.n_rows, self.n_cols))

    def init_from_solution_string(self, string_box):
        self.n_rows = len(string_box)
        self.n_cols = len(string_box[0])

        cols = []
        rows_constraints = []
        cols_constraints = []
        for row in string_box:
            row_constraints = [
                len(x) for x in row.split('.') if len(x) > 0]
            if not self.ordered:
                row_constraints = sorted(row_constraints)
            rows_constraints.append(row_constraints)
            for col_char, col_index in zip(
                    row, range(len(string_box[0]))):
                try:
                    col = cols[col_index]
                except IndexError:
                    col = ''
                    cols.append([])
                col += col_char
                cols[col_index] = col
        for col in cols:
            col_constraints = [
                len(x) for x in col.split('.') if len(x) > 0]
            if not self.ordered:
                col_constraints = sorted(col_constraints)
            cols_constraints.append(col_constraints)

        filled_positions = []
        for row_index, row in zip(range(self.n_rows), string_box):
            for col_index, col_char in zip(range(self.n_cols), row):
                if col_char == 'o':
                    filled_positions.append((row_index, col_index))

        self.solution_list = filled_positions
        self.rows_constraints = rows_constraints
        self.cols_constraints = cols_constraints
        self._init_puzzle()
        self.solution_state = np.zeros((self.n_rows, self.n_cols))
        for (row, col) in filled_positions:
            self.solution_state[row, col] = 1

    def init_from_matrix(self, matrix):
        """
        Args:
            matrix (numpy array): array of arrays representing the solution
        """
        self.solution_state = matrix
        self.solution_list = zip(np.where(matrix == 1))
        # TODO: finish this function

    def display_puzzle_svg(self):
        pass

    def display_solution_svg(self):
        pass
