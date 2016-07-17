"""
Given a solution for a nonogram puzzle:
1. prove it can be solveable
2. if it cannot be solveable, add constraints until the solution is unique
"""
import numpy as np
import random
random.seed(50)

debug = True


def possibilities_generator(
        prior, min_pos, max_start_pos, constraint_len, total_filled):
    """
    Given a row prior, a min_pos, max_start_pos, and constraint length,
    yield each potential row

    prior is an array of:
        -1 (unknown),
        0 (definitely empty),
        1 (definitely filled)
    """
    prior_filled = np.zeros(len(prior)).astype(bool)
    prior_filled[prior == 1] = True
    prior_empty = np.zeros(len(prior)).astype(bool)
    prior_empty[prior == 0] = True
    for start_pos in range(min_pos, max_start_pos + 1):
        possible = -1 * np.ones(len(prior))
        possible[start_pos:start_pos + constraint_len] = 1
        if start_pos + constraint_len < len(possible):
            possible[start_pos + constraint_len] = 0
        if start_pos > 0:
            possible[start_pos - 1] = 0

        # add in the prior
        possible[np.logical_and(possible == -1, prior == 0)] = 0
        possible[np.logical_and(possible == -1, prior == 1)] = 1

        # if contradiction with prior, continue
        # 1. possible changes prior = 1 to something else
        # 2. possible changes prior = 0 to something else
        # 3. everything is assigned in possible but there are not
        #    enough filled in
        # 4. possible changes nothing about the prior
        if np.any(possible[np.where(prior == 1)[0]] != 1) or \
                np.any(possible[np.where(prior == 0)[0]] != 0) or \
                np.sum(possible == 1) > total_filled or \
                (np.all(possible >= 0) and np.sum(possible == 1) <
                    total_filled) or \
                np.all(prior == possible):
            continue
        yield possible


def generate_constraints_line(constraints, prior, size):
    """
    example:
    row length = 4, row constraint = 2 (it is the only constraint)
    min_pos = 0, max_pos = 3, max_start_pos = 2

    row length = 5, row constraints 1, 2
    constraint 1:
        min_pos = 0, max_pos = 1, max_start_pos = 1
    constraint 2:
        min_pos = 2, max_pos = 4, max_start_pos = 3
    """
    if len(constraints) == 0:
        return (prior == 1, np.logical_not(prior == 0))

    total_filled = np.sum(constraints)

    # if the number of filled in squares is equal to the total
    # constraints, fill in the rest with empties and call it a day
    if np.sum(prior == 1) == total_filled:
        for index in range(len(prior)):
            if prior[index] == -1:
                prior[index] = 0
        return (prior == 1, prior == 0)

    # if there is only one square left un-solved, solve it
    index_unknown = np.where(prior == -1)[0]
    if len(index_unknown) == 1:
        currently_filled_in_prior = np.sum(prior == 1)
        if total_filled == currently_filled_in_prior:
            prior[index_unknown[0]] = 0
        else:
            prior[index_unknown[0]] = 1

        return (prior == 1, prior == 0)

    # if any constraints are solved, don't
    # iterate over them
    indices_to_remove = []
    constraint_indices = np.ones(len(constraints))

    start_blocks = []
    start_would_be_zero_index = []
    block = 0
    for index, item in zip(range(len(prior)), prior):
        if item != -1 and item == 1:
            block += 1
        else:
            if block > 0:
                start_would_be_zero_index.append(index)
                start_blocks.append(block)
                block = 0
            if item == -1:
                break
    if block > 0:
        start_would_be_zero_index.append(index)
        start_blocks.append(block)

    index = 0
    for block, constraint in zip(
            start_blocks, constraints[:len(start_blocks)]):
        if block == constraint:
            # remove from iteration
            indices_to_remove.append(index)
            # if the block is correct, update the prior appropriately
            prior[start_would_be_zero_index[index]] = 0
            before_index = start_would_be_zero_index[index] - constraint - 1
            if before_index > 0:
                prior[before_index] = 0
        else:
            break
        index += 1

    end_blocks = []
    end_would_be_zero_index = []
    reverse_index_range = range(len(prior))
    reverse_index_range.reverse()
    block = 0
    for index, item in zip(reverse_index_range, prior[::-1]):
        if item != -1 and item == 1:
            block += 1
        else:
            if block > 0:
                end_would_be_zero_index.append(index)
                end_blocks.append(block)
                block = 0
            if item == -1:
                break
    if block > 0:
        end_would_be_zero_index.append(index)
        end_blocks.append(block)

    index = len(constraints) - 1
    for block, constraint in zip(
            end_blocks[::-1],
            constraints[-1 * len(end_blocks):]):
        if block == constraint:
            # remove from iteration
            indices_to_remove.append(index)
            # if the block is correct, update the prior appropriately
            prior[end_would_be_zero_index[len(constraints) - 1 - index]] = 0
            before_index = end_would_be_zero_index[
                len(constraints) - 1 - index] + constraint + 1
            if before_index > 0 and before_index < len(prior):
                prior[before_index] = 0
        else:
            break
        index -= 1

    for index in indices_to_remove:
        constraint_indices[index] = 0

    # if all the constraints are met, make sure the
    # -1s are 0
    if np.all(constraint_indices == 0):
        for index in range(len(prior)):
            if prior[index] == -1:
                prior[index] = 0
        return (prior == 1, prior == 0)

    possibilities_filled = prior == 1
    possibilities_empty = prior == 0

    for constraint_index, contraint in zip(
            range(len(constraints)), constraints):
        if constraint_indices[constraint_index] == 0:
            continue

        possibilities_filled_temp = np.ones(size).astype(bool)
        possibilities_empty_temp = np.ones(size).astype(bool)
        min_pos = sum(constraints[:constraint_index]) + constraint_index
        max_pos = size - 1 - (
            sum(constraints[constraint_index + 1:]) +
            len(constraints[constraint_index + 1:]))

        # if we already have everything in that window
        # just continue
        if np.all(prior[min_pos:max_pos + 1] != -1):
            possibilities_filled[min_pos:max_pos + 1] = \
                prior[min_pos:max_pos + 1] == 1
            possibilities_empty[min_pos:max_pos + 1] = \
                prior[min_pos:max_pos + 1] == 0
            continue

        # if you have some contiguous thing from the prior in the range of
        # min_pos and max_pos, fix it
        potential_indices = np.where(prior[min_pos:max_pos + 1] != 0)[0]
        if len(potential_indices) > 0:
            potential_indices += min_pos
            min_pos = min(potential_indices)
            max_pos = max(potential_indices)

        max_start_pos = max_pos + 1 - contraint

        # the smart logic about the constraints is generated
        # in the possibilities_generator
        i = 0
        for possible_nums in possibilities_generator(
                prior, min_pos, max_start_pos, contraint, total_filled):
            possible_filled = possible_nums == 1
            possible_empty = possible_nums != 1
            possibilities_filled_temp = np.logical_and(
                possibilities_filled_temp, possible_filled)
            possibilities_empty_temp = np.logical_and(
                possibilities_empty_temp, possible_empty)
            i += 1

        if i > 0:
            possibilities_filled[min_pos:max_pos + 1] = \
                possibilities_filled_temp[min_pos:max_pos + 1]
            possibilities_empty[min_pos:max_pos + 1] = \
                possibilities_empty_temp[min_pos:max_pos + 1]

        # TODO: you can probably end early here. figure out the constraint
        # if np.all(possibilities_filled == False) and \
        #         np.all(possibilities_empty == False):
        #     break
    return (possibilities_filled, possibilities_empty)


def generate_solutions(
        n_rows, n_cols, rows_constraints, cols_constraints, puzzle):
    """
    Given the total size of the grid, and the row
    and column constraints for the grid, use depth-first
    search to find all the possible solutions to the puzzle.

    Step 1:
    For each row / for each column, take the set of constraints
    (and you have to know whether it is ordered or unordered)
    generate all possible combinations of answers. Take a
    logical and across all of them to get the squares that have
    to be constrained. Continue iterating through the rows
    and columns until no more squares have been added.
    Then start the DFS approach to building the solution tree.

    Step 2:
    Start in the top left corner of the grid and move right,
    then down.
    """
    # TODO: start with the shorter one
    # Step 1
    # rows
    n_changed = 1
    iters = 0
    while n_changed > 0:
        n_changed = 0
        for row_index, row_constraints in zip(range(n_rows), rows_constraints):
            # if the row has no empty spaces left,
            # just skip that set of constraints
            if np.sum(puzzle[row_index, :] == -1) == 0:
                continue

            (possibilities_filled, possibilities_empty) = \
                generate_constraints_line(
                    row_constraints,
                    puzzle[row_index, :],
                    n_cols)

            for index in range(len(possibilities_filled)):
                if possibilities_filled[index] and \
                        puzzle[row_index, index] == 1:
                    possibilities_filled[index] = False

            for index in range(len(possibilities_empty)):
                if possibilities_empty[index] and \
                        puzzle[row_index, index] == 0:
                    possibilities_empty[index] = False

            if np.all(np.logical_not(possibilities_filled)) and \
                    np.all(np.logical_not(possibilities_empty)):
                continue

            puzzle[row_index, :][possibilities_filled] = 1
            puzzle[row_index, :][possibilities_empty] = 0

            n_changed += sum(possibilities_filled)
            n_changed += sum(possibilities_empty)

        if debug:
            for row_index in range(n_rows):
                print puzzle[row_index, :]
        print "n_changed = %s" % n_changed

        # cols
        for col_index, col_constraints in zip(range(n_cols), cols_constraints):
            # if the row has no empty spaces left,
            # just skip that set of constraints
            if np.sum(puzzle[:, col_index] == -1) == 0:
                continue

            (possibilities_filled, possibilities_empty) = \
                generate_constraints_line(
                    col_constraints,
                    puzzle[:, col_index],
                    n_rows)

            for index in range(len(possibilities_filled)):
                if possibilities_filled[index] and \
                        puzzle[index, col_index] == 1:
                    possibilities_filled[index] = False

            for index in range(len(possibilities_empty)):
                if possibilities_empty[index] and \
                        puzzle[index, col_index] == 0:
                    possibilities_empty[index] = False

            if np.all(np.logical_not(possibilities_filled)) and \
                    np.all(np.logical_not(possibilities_empty)):
                continue

            puzzle[:, col_index][possibilities_filled] = 1
            puzzle[:, col_index][possibilities_empty] = 0

            n_changed += sum(possibilities_filled)
            n_changed += sum(possibilities_empty)

        print "n_changed = %s" % n_changed
        print "finished iter %s" % iters

        iters += 1
    return puzzle


def generate_row_solutions_recursive(
        constraints_list, state, prior,
        possibilities_filled, possibilities_empty,
        total_filled_in_row, state_len):

    # print state, constraints_list

    # print "---"
    # print "state = %s" % state
    # print "possibilities empty %s" % possibilities_empty
    # print "possibilities filled %s" % possibilities_filled
    # print "constraints_list %s" % constraints_list
    # print "---"
    # possibilities_filled_temp = np.ones(len(prior)).astype(bool)
    # possibilities_empty_temp = np.ones(len(prior)).astype(bool)

    # import pudb; pudb.set_trace()

    if np.all(np.logical_not(possibilities_empty)) and \
            np.all(np.logical_not(possibilities_filled)):
        return (possibilities_filled, possibilities_empty)

    if np.sum(state == 1) == total_filled_in_row or \
            len(constraints_list) == 0:
        for index in xrange(state_len):
            if state[index] == -1:
                state[index] = 0

        # if contradiction between state and prior,
        # return None
        if np.any(state[np.where(prior == 1)[0]] != 1) or \
                np.any(state[np.where(prior == 0)[0]] != 0) or \
                np.sum(state == 1) > total_filled_in_row or \
                (np.all(state >= 0) and np.sum(state == 1) !=
                    total_filled_in_row) or \
                np.all(prior == state):
            # print "---"
            return (possibilities_filled, possibilities_empty)

        possibilities_filled = np.logical_and(
            possibilities_filled, state == 1)
        possibilities_empty = np.logical_and(
            possibilities_empty, state == 0)
        return (possibilities_filled, possibilities_empty)

    # for constraint_index, constraint in zip(
    #         range(len(constraints_list)), constraints_list):
    constraint = constraints_list[0]

    unknown_pos = np.where(state != -1)[0]
    min_possible_index = 0
    if len(unknown_pos) > 0:
        # min_possible_index = max(unknown_pos + 1)
        min_possible_index = unknown_pos[-1] + 1
    # min_pos = sum(
    #     constraints_list[:constraint_index]) + constraint_index
    max_pos = state_len - 1 - (
        sum(constraints_list[1:]) +
        len(constraints_list[1:]))

    # min_pos += min_possible_index
    max_pos = min(max_pos + min_possible_index, state_len - 1)
    max_start_pos = max_pos + 1 - constraint

    for start_pos in xrange(min_possible_index, max_start_pos + 1):
        # if you are going to write over the state, don't
        # consider it
        possible = state.copy()

        possible[start_pos:start_pos + constraint] = 1
        if start_pos + constraint < state_len:
            possible[start_pos + constraint] = 0

        for index in xrange(start_pos):
            if possible[index] == -1:
                possible[index] = 0

        # if np.any(prior[np.where(possible == 1)[0]] == 0) or \
        #         np.any(prior[np.where(possible == 0)[0]] == 1) or \
        #         np.sum(possible == 1) > total_filled_in_row or \
        #         (np.all(possible >= 0) and np.sum(possible == 1) !=
        #             total_filled_in_row) or \
        #         np.all(possible == prior):
        #     return (possibilities_filled, possibilities_empty)

        subconstraint_list = constraints_list[1:]
        (possibilities_filled, possibilities_empty) = \
            generate_row_solutions_recursive(
            subconstraint_list,
            possible,
            prior,
            possibilities_filled, possibilities_empty,
            total_filled_in_row,
            state_len)

    return (possibilities_filled, possibilities_empty)


def generate_solutions_recursive(
        n_rows, n_cols, rows_constraints, cols_constraints, puzzle_state):
    # iterate through all possible row constraints
    # and and all the possibilities

    n_changed = 1
    iters = 0
    while n_changed > 0:
        n_changed = 0
        # iterate through rows
        for row_index, row_constraints in zip(range(n_rows), rows_constraints):
            # for a given row, iterate through all possible
            # combinations
            # if row_index == 3 and iters == 1:
            #     import pudb; pudb.set_trace()
            prior = puzzle_state[row_index, :]

            # possibilities_filled = prior == 1
            # possibilities_empty = prior == 0
            possibilities_filled = np.ones(len(prior)).astype(bool)
            possibilities_empty = np.ones(len(prior)).astype(bool)

            total_filled_in_row = np.sum(row_constraints)
            # import pudb; pudb.set_trace()
            (possibilities_filled, possibilities_empty) = \
                generate_row_solutions_recursive(
                row_constraints, -1 * np.ones(len(prior)),
                prior,
                possibilities_filled, possibilities_empty,
                total_filled_in_row,
                n_cols)

            if np.all(possibilities_empty) and np.all(possibilities_filled):
                continue

            print "row %s, constraints %s" % (row_index, row_constraints)
            print "FINISHED: filled %s" % possibilities_filled
            print "FINISHED: empty %s" % possibilities_empty

            for index in range(len(possibilities_filled)):
                if possibilities_filled[index] and \
                        puzzle_state[row_index, index] == 1:
                    possibilities_filled[index] = False

            for index in range(len(possibilities_empty)):
                if possibilities_empty[index] and \
                        puzzle_state[row_index, index] == 0:
                    possibilities_empty[index] = False

            if np.all(possibilities_filled) is False and \
                    np.all(possibilities_empty) is False:
                continue

            puzzle_state[row_index, :][possibilities_filled] = 1
            puzzle_state[row_index, :][possibilities_empty] = 0

            n_changed += sum(possibilities_filled)
            n_changed += sum(possibilities_empty)

        # iterate through cols
        for col_index, col_constraints in zip(range(n_cols), cols_constraints):
            # for a given row, iterate through all possible
            # combinations
            prior = puzzle_state[:, col_index]

            # possibilities_filled = prior == 1
            # possibilities_empty = prior == 0
            possibilities_filled = np.ones(len(prior)).astype(bool)
            possibilities_empty = np.ones(len(prior)).astype(bool)

            total_filled_in_col = np.sum(col_constraints)
            (possibilities_filled, possibilities_empty) = \
                generate_row_solutions_recursive(
                col_constraints, -1 * np.ones(len(prior)),
                prior,
                possibilities_filled, possibilities_empty,
                total_filled_in_col,
                n_rows)

            if np.all(possibilities_empty) and np.all(possibilities_filled):
                continue

            print "col %s, constraints %s" % (col_index, col_constraints)
            print "FINISHED: filled %s" % possibilities_filled
            print "FINISHED: empty %s" % possibilities_empty

            for index in range(len(possibilities_filled)):
                if possibilities_filled[index] and \
                        puzzle_state[index, col_index] == 1:
                    possibilities_filled[index] = False

            for index in range(len(possibilities_empty)):
                if possibilities_empty[index] and \
                        puzzle_state[index, col_index] == 0:
                    possibilities_empty[index] = False

            if np.all(possibilities_filled) is False and \
                    np.all(possibilities_empty) is False:
                continue

            puzzle_state[:, col_index][possibilities_filled] = 1
            puzzle_state[:, col_index][possibilities_empty] = 0

            n_changed += sum(possibilities_filled)
            n_changed += sum(possibilities_empty)

        print "finished iter %s" % iters
        print "n_changed = %s" % n_changed
        print "inner puzzle state: "
        print puzzle_state
        iters += 1

    # import pudb; pudb.set_trace()
    return puzzle_state

    # # iterate through cols
    # for col_index, col_constraints in zip(range(n_cols), cols_constraints):
    #     prior = puzzle_state[:, col_index]

    #     possibilities_filled = prior == 1
    #     possibilities_empty = prior == 0
    #     for constraint in col_constraints:
    #         pass


class Nonogram(object):
    def __init__(self, braille_box, ordered=True):
        self.braille_box = braille_box
        self.rows_constraints, self.cols_constraints = \
            self._generate_constraints(ordered=ordered)
        self.n_rows = len(braille_box)
        self.n_cols = len(braille_box[0])
        self.solutions = []
        self.ordered = ordered
        self.puzzle_state = -1 * np.ones((self.n_rows, self.n_cols))
        self.filled_positions_hint_eligible = \
            self._generate_filled_positions_from_braille_box()
        self.prefilled_positions = []

    def _generate_filled_positions_from_braille_box(self):
        # given the braille box, generate the filled positions
        # in a list
        filled_positions = []
        for row_index, row in zip(range(self.n_rows), self.braille_box):
            for col_index, col_char in zip(range(self.n_cols), row):
                if col_char == 'o':
                    filled_positions.append((row_index, col_index))
        return filled_positions

    def _generate_constraints(self, ordered=True):
        # write the constraints on the left and top
        # generate the row and col constraints
        cols = []
        rows_constraints = []
        cols_constraints = []
        for row in self.braille_box:
            row_constraints = [
                len(x) for x in row.split('.') if len(x) > 0]
            if not ordered:
                row_constraints = sorted(row_constraints)
            rows_constraints.append(row_constraints)
            for col_char, col_index in zip(
                    row, range(len(self.braille_box[0]))):
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
            if not ordered:
                col_constraints = sorted(col_constraints)
            cols_constraints.append(col_constraints)

        return rows_constraints, cols_constraints

    def _generate_solutions(self):
        self.puzzle_state = generate_solutions(
            self.n_rows, self.n_cols, self.rows_constraints,
            self.cols_constraints, self.puzzle_state)
        # self.puzzle_state = generate_solutions_recursive(
        #     self.n_rows, self.n_cols, self.rows_constraints,
        #     self.cols_constraints, self.puzzle_state)

        print self.puzzle_state

        # update the positions that can be used for hints appropriately
        for row_index, row in zip(range(self.n_rows), self.puzzle_state):
            for col_index, col_elem in zip(range(self.n_cols), row):
                pos = (row_index, col_index)
                if col_elem == 1 and \
                        pos in self.filled_positions_hint_eligible:
                    self.filled_positions_hint_eligible.remove(pos)

    def _pick_help_square(self):
        # randomly pick an element in the filled list
        # add that filled in thing to the puzzle solution
        print "%s squares available" % (
            len(self.filled_positions_hint_eligible))
        if len(self.filled_positions_hint_eligible) == 0:
            raise ValueError("No more positions available")
        filled_square = random.choice(self.filled_positions_hint_eligible)
        if debug:
            print "choosing (%s, %s) to fill" % (
                filled_square[0], filled_square[1])
        self.filled_positions_hint_eligible.remove(filled_square)
        self.prefilled_positions.append(filled_square)

        self.puzzle_state[filled_square[0], filled_square[1]] = 1

    def _puzzle_is_solved(self):
        if np.sum(self.puzzle_state == -1) == 0:
            return True
        return False


def generate_puzzle_constraints(nonogram, starter=None):
    if starter is not None:
        for i in xrange(starter):
            nonogram._pick_help_square()

    nonogram._generate_solutions()

    while not nonogram._puzzle_is_solved():
        # pick a random filled in square
        # generate solutions until the puzzle is solved
        # import pudb; pudb.set_trace()
        nonogram._pick_help_square()
        # print nonogram.prefilled_positions

        nonogram._generate_solutions()

    return nonogram.prefilled_positions
