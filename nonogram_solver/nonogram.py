"""
Defines the nonogram datatype.
"""
import numpy as np
import svgwrite
import json

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
        #svg constants
        self.mcols,self.mrows=[None,None]    #max num of constraints
        self.wBox,self.margin=[30,30]        #dimensions
        self.numBoxX,self.numBoxY=[0,0]      #constraints+fields
        self.wXstroke,self.wYstroke=[0,0]    #length of strokes
        self.xySize=[0,0]                    #canvas size

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
    def init_from_json(self, jfile):
        """
        Args:
            the path to a json file: 
                json containing the row and col constraints as arrays of arrays i.e:
                {
                    "ver":[[1],[]],
                    "hor":[[1],[]]
                }
                dimensions are deductable by len(json["ver"])/len(json["hor"])
        """
        j=json.loads(open(jfile).read())
        self.n_rows = len(j["ver"])
        self.n_cols = len(j["hor"])

        filled_positions = []
        self.solution_list = filled_positions
        self.rows_constraints = j["ver"]
        self.cols_constraints = j["hor"]
        self._init_puzzle()
        self.solution_state = np.zeros((self.n_rows, self.n_cols))

    def display_number(self,svg,x,y,num):
        if num>9: cor=.15 
        else: cor=0
        svg.add(svg.text(num, insert=(self.margin+(x-2/3-cor)*self.wBox, self.margin+(y+2/3)*self.wBox), font_size=16, fill='black'))
                
    def display_puzzle_svg(self,toSVG="test.svg"):
        #init SVG
        dwg = svgwrite.Drawing(toSVG, profile='tiny')
        #get row/col constraints and max numbers of hints
        rows,cols=[self.rows_constraints,self.cols_constraints]
        self.mcols,self.mrows=[max([len(e) for e in cols]),max([len(e) for e in rows])]
        #set constants
        self.numBoxX,self.numBoxY=[self.mrows+self.n_cols,self.mcols+self.n_rows] #constraints+fields
        self.wXstroke,self.wYstroke=[self.numBoxX*self.wBox,self.wBox*self.numBoxY]         #length of strokes
        self.xySize=[self.wXstroke+2*self.margin,self.wYstroke+2*self.margin]
        #print gray raster
        for i in range(self.numBoxX+1):
            dwg.add(dwg.line((self.margin+i*self.wBox,self.margin), (self.margin+i*self.wBox,self.margin+self.wYstroke), stroke_width=5 ,stroke=svgwrite.rgb(70, 70, 70, '%')))
        for i in range(self.numBoxY+1):
            dwg.add(dwg.line((self.margin,self.margin+i*self.wBox), (self.margin+self.wXstroke,self.margin+i*self.wBox), stroke_width=5 ,stroke=svgwrite.rgb(70, 70, 70, '%')))
        #print black raster
        for i in [0,self.numBoxX]+[b for b in range(self.mrows,self.numBoxX,5)]:
            dwg.add(dwg.line((self.margin+i*self.wBox,self.margin), (self.margin+i*self.wBox,self.margin+self.wYstroke), stroke_width=5 ,stroke=svgwrite.rgb(0, 0, 0, '%')))
        for i in [0,self.numBoxY]+[b for b in range(self.mcols,self.numBoxY,5)]:
            dwg.add(dwg.line((self.margin,self.margin+i*self.wBox), (self.margin+self.wXstroke,self.margin+i*self.wBox), stroke_width=5 ,stroke=svgwrite.rgb(0, 0, 0, '%')))
        #print row constraints
        for r in range(len(rows)):
            ei=self.mrows
            for e in range(1,1+len(rows[r])):
                self.display_number(dwg,ei,self.mcols+r,rows[r][-e])
                ei-=1        
        #print col constraints
        for c in range(len(cols)):
            ei=self.mcols-1
            for e in range(1,1+len(cols[c])):
                self.display_number(dwg,self.mrows+c+1,ei,cols[c][-e])
                ei-=1
        #set viewbox and export
        dwg.viewbox(minx=0, miny=0, width=self.xySize[0], height=self.xySize[1])
        dwg.save()
    def display_solution_svg(self):
        pass