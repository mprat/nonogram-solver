Nonogram Solver
===============

|  |ci-status|

This project provides a general nonogram solver. It can determine if a puzzle is not solveable by a human, and if it is solveable, can provide the solution.

This solver can be used to create nonogram puzzles given a successful final solution. If the given solution is not solvable, the solver will suggest "hint" squares to be filled in when the nonogram is given to a human solver.

.. |ci-status| image:: https://travis-ci.org/mprat/nonogram-solver.svg?branch=master
    :target: https://travis-ci.org/mprat/nonogram-solver
    :alt: Build status

Example Usage
-----------------------
See example.py for sample use of the library.

.. code:: bash

    python3 example.py json --json example.json
    python3 example.py --svgout example.svg csv --columncsv example.columns.csv --rowcsv example.rows.csv

SVG Example:

.. image:: example.svg
   :height: 420
   :width: 305


Installation
-----------------------
To install, run `pip install nonogram-solver`.


Project Website
-----------------------
PyPI: `https://pypi.python.org/pypi/nonogram-solver <https://pypi.python.org/pypi/nonogram-solver>`_

Github: `https://github.com/mprat/nonogram-solver <https://github.com/mprat/nonogram-solver>`_
