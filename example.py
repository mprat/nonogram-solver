import numpy as np
from nonogram_solver import solver
from nonogram_solver.nonogram import Nonogram
import argparse

parser = argparse.ArgumentParser(description="Nonogram solver")
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--addcontraints",action='store_true',help='add_puzzle_constraints arg to solver')
parser.add_argument("--svgout",help='svgoutfile',type=str)
subparsers=parser.add_subparsers(title='subcommands',dest='subparser_name')
jsongroup = subparsers.add_parser('json',help='JSON file input')
jsongroup.add_argument("--json", help="JSON File Path", required=True, type=str)
csvgroup = subparsers.add_parser('csv',help='CSV file input')
csvgroup.add_argument("--columncsv", help="Column Constraint File Path", required=True, type=str)
csvgroup.add_argument("--rowcsv", help="Row Constraint File Path", required=True, type=str)

args = parser.parse_args()

if not args.subparser_name:
    parser.print_help()
    exit(2)

nonogram = Nonogram()
nonogram.rows_constraints = []
nonogram.cols_constraints = []
nonogram.solution_list=[]

if args.subparser_name == 'json':
    import json
    with open(args.json) as jsonfile:  
        puzzle = json.load(jsonfile)
        if not puzzle['rows'] or not puzzle['columns']:
            print("No column or row info")
            exit(1)
        nonogram.n_cols=len(puzzle['columns'])
        nonogram.n_rows=len(puzzle['rows'])
        for c in puzzle['columns']:
            nonogram.cols_constraints.append(c)
        for r in puzzle['rows']:
            nonogram.rows_constraints.append(r)

if args.subparser_name == 'csv':
    import csv
    with open(args.columncsv, newline='') as columncsv:
        columns = csv.reader(columncsv, quoting=csv.QUOTE_NONNUMERIC)
        for row in columns:
            nonogram.cols_constraints.append(list(map(int, row)))
        nonogram.n_cols=len(nonogram.cols_constraints)
    with open(args.rowcsv, newline='') as rowcsv:
        rows = csv.reader(rowcsv, quoting=csv.QUOTE_NONNUMERIC)
        for row in rows:
            nonogram.rows_constraints.append(list(map(int, row)))
        nonogram.n_rows=len(nonogram.rows_constraints)

import io
from contextlib import redirect_stdout

verbose = io.StringIO()
with redirect_stdout(verbose):
    solveable, nonogram_solver = solver.solve(nonogram, add_puzzle_constraints=args.addcontraints)

if args.verbose:
    print(verbose.getvalue())
if solveable:
    print("The puzzle was solvable")
else:
    print("The puzzle was not solvable")

#import pprint
#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(nonogram_solver.puzzle_state)

for r in nonogram_solver.puzzle_state:
    print(*list(map(lambda x: '█' if x==1 else ' ', r)),sep='')


if args.svgout:
    import svgwrite
    from svgwrite import cm
    dwg = svgwrite.Drawing(args.svgout, profile='tiny')
    shapes = dwg.add(dwg.g(id='shapes', fill='white'))

    rc=0
    for r in nonogram_solver.puzzle_state:
        cc=0
        for c in r:
            if c == 1:
                shapes.add(dwg.rect(insert=(cc*cm, rc*cm), size=(1*cm, 1*cm), fill='black', stroke='black', stroke_width=1))
            else:
                shapes.add(dwg.rect(insert=(cc*cm, rc*cm), size=(1*cm, 1*cm), fill='white', stroke='black', stroke_width=1))
            cc=cc+1
        rc=rc+1
    dwg.save()