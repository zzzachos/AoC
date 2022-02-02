"""2021_AoC_Wk2.py
Solutions to days 8-15 of 2021 Advent of Code.
Refer to that website for problem statements.
All solutions arrived at independently, with the exception
of the alternate solution to day 8 
and discovering some hidden assumptions we can make of the data on day 13.

Python version: Python 3.7.11
To run from the command line, specify the problem with format dayN_P 
and provide the appropriate input file obtained from AoC or test file of your own creation. Example:
>> python 2021_AoC_Wk1.py day8_2 input_8.txt
Take care to provide the right input files, corresponding to the correct day, 
as input of an unexpected format will produce a world of errors, including unending loops.

All expected answers are positive integers, so -1 was used as a default output 
some of the times if the function were to end in an unexpected state.

Comments: days 13 and 14 have nonideal solutions.
The approach should be rethought and the code improved. 
"""

import sys
import time
import re 
import numpy as np 

### DAY 8 ###
signal_wire_counts = {0: 6, 1: 2, 2:5, 3:5, 4:4, 5:5, 6:6, 7:3, 8:7, 9:6}
easy_digits = [1,4,7,8]

def solve_easy_wire_mystery(second_part):
    #setup
    desired_counts = []
    for d in easy_digits:
        desired_counts.append(signal_wire_counts[d])
    total = 0
    
    #count
    for signal in second_part:
        if len(signal) in desired_counts:
            total+=1
    return total
    

# Solved the problem in 0.0012 seconds
def day8_1(input_file):
    running_count= 0
    
    pattern = re.compile('(([a-g]+\s){10})\|\s(([a-g]+\s?){4})')
    
    with open(input_file) as fi:
        for line in fi:
            #parse each line
            m = re.match(pattern, line)#m[1].split(), 
            #add count
            running_count+=solve_easy_wire_mystery(m[3].split())

    return running_count


# Solved the problem in 0.0068 seconds
# Not an elegantly clever solution, but fairly organized at least
def day8_2(input_file):
    running_count= 0
    
    pattern = re.compile('(([a-g]+\s){10})\|\s(([a-g]+\s?){4})')
    
    with open(input_file) as fi:
        for line in fi:
            #parse each line
            m = re.match(pattern, line)
            input_words = m[1].split()
            output_words = m[3].split()
            
            #solve puzzle
            puzzle = DigitPuzzle()
            puzzle.solve_puzzle(input_words)
            
            #add count
            running_count+=puzzle.decode_output(output_words)

    return running_count
   
#KEY INSIGHT FOR PART 2: each line's input has all digits appearing
#We identify 1,4,7,8 by length
#This allows us to determine a,b,d
#By looking at 2,3,5, we can determine g,f,c,e, which solves everything
#do this line by line
class DigitPuzzle:
    def __init__(self):
        self.false_letters_to_true_letters = {}
        self.words_to_digits = {}
        self.solved = False
        
    #list of four words decoded (might be scrambled in different order)
    def decode_output(self, output):
        if self.solved==False:
            print("We can not decode this output!")
            return -1
        base = 0
        power = 1
        while len(output)>0:
            word = output.pop()
            for reordered_word in self.words_to_digits:
                if set(reordered_word)==set(word):
                    base += self.words_to_digits[reordered_word]*power
                    break
            power = power*10
        return base
    
    #given 10 input words only, which we assume to be unique 
    def solve_puzzle(self, words):
        #first identify 1,4,7,8
        #unique_lengths = {2: 1, 3: 7, 4: 4,  7: 8}
        words.sort(key = len)
        true_eight = words.pop()
        self.words_to_digits[true_eight] = 8
        true_one = words.pop(0)
        self.words_to_digits[true_one] = 1
        true_seven = words.pop(0)
        self.words_to_digits[true_seven] = 7
        true_four = words.pop(0)
        self.words_to_digits[true_four] = 4
        
        c_and_f = set(true_one)
        b_and_d = set(true_four) - set(true_one)
        
        #next identify 2,3,and 5 (5 segments activated)
        current = []
        sets = []
        current.append(words.pop(0))
        sets.append(set(current[-1]))
        current.append(words.pop(0))
        sets.append(set(current[-1]))
        current.append(words.pop(0))
        sets.append(set(current[-1]))
        
        a_d_g = (sets[0].intersection(sets[1])).intersection(sets[2])
        
        #find 5
        for i in range(3):
            if len(b_and_d.intersection(sets[i]))>1:
                true_five = current.pop(i)
                b_and_f = sets.pop(i) - a_d_g
                self.words_to_digits[true_five] = 5
                break
                
        #find 3 and thus 2
        for i in range(2):
            if len(c_and_f.intersection(sets[i]))>1:
                true_three = current.pop(i)
                sets.pop(i)
                self.words_to_digits[true_three] = 3
                self.words_to_digits[current.pop()] = 2
                break
        
        #finally identify 0,6,9 (6 segments activated)
        current = []
        sets = []
        current.append(words.pop(0))
        sets.append(set(current[-1]))
        current.append(words.pop(0))
        sets.append(set(current[-1]))
        current.append(words.pop(0))
        sets.append(set(current[-1]))
        
        #find 0
        for i in range(3):
            if len(b_and_d.intersection(sets[i]))<2:
                true_zero = current.pop(i)
                sets.pop(i)
                self.words_to_digits[true_zero] = 0
                break
                
        #find 6 and therefore 9
        for i in range(3):
            if len(c_and_f.intersection(sets[i]))<2:
                true_six = current.pop(i)
                sets.pop(i)
                self.words_to_digits[true_six] = 6
                self.words_to_digits[current.pop()] = 9
                break
        
        if len(self.words_to_digits) == 10:
            self.solved = True
            #print("Solved!")
            #print(self.words_to_digits)
        else:
            print(words)
            print("Houston, we have a problem. We should have solved it by now!")
            print(self.words_to_digits)



### Other people's code:
##Day 8: https://www.reddit.com/r/adventofcode/comments/rbj87a/comment/hnoyy04/
##Requires python 3.10 for match-case
##has at least 3 better ideas than my solution
# s = 0
# for x,y in [x.split('|') for x in open(0)]:  # split signal and output
#   l = {len(s): set(s) for s in x.split()}    # get number of segments
# 
#   n = ''
#   for o in map(set, y.split()):              # loop over output digits
#     match len(o), len(o&l[4]), len(o&l[2]):  # mask with known digits
#       case 2,_,_: n += '1'
#       case 3,_,_: n += '7'
#       case 4,_,_: n += '4'
#       case 7,_,_: n += '8'
#       case 5,2,_: n += '2'
#       case 5,3,1: n += '5'
#       case 5,3,2: n += '3'
#       case 6,4,_: n += '9'
#       case 6,3,1: n += '6'
#       case 6,3,2: n += '0'
#   s += int(n)
# 
# print(s)



### DAY 9 ###

def get_adjacent_indices(x,y,sizex, sizey):
    indices = []
    if x>0:
        indices+=[(x-1,y)]
    if x<sizex-1:
        indices+=[(x+1,y)]
    if y>0:
        indices+=[(x, y-1)]
    if y<sizey - 1:
        indices+=[(x, y+1)]
    return indices

#Solved the problem in 0.0005 seconds
def day9_1(input_file):

    size = 0
    row_no = 0
    with open(input_file) as fi:
        for line in fi:
            if size==0:
                size = len(line.strip())
                sea_floor = np.zeros((size, size))
            row = [int(s) for s in line.strip()]
            sea_floor[row_no] = row
            row_no+=1
    sea_floor = sea_floor[:row_no, :]
    low_points = []
    for i in range(row_no):
        for j in range(size):
            adjacent = get_adjacent_indices(i,j,row_no,size)
            low = True
            value = sea_floor[i,j]
            for adj in adjacent:
                if sea_floor[adj]<=value:
                    low = False
            if low:
                low_points.append(value)
    
                
            
    return int(sum(low_points))+ len(low_points)


##UNION FIND! IN THE WILD!
##We simultaneously keep track of component sizes
class UnionFindTrack:
    def __init__(self):
        self.nodes = {}
        self.extra_info = {}
    
    #for original population
    def add_disconnected(self,x, default = 1):
        self.nodes[x] = x
        self.extra_info[x] = default
    
    #with path compression
    def root(self,x):
        while self.nodes[x]!=x:
            self.nodes[x]=self.nodes[self.nodes[x]]
            x = self.nodes[x]
        return x
    
     
    def check_same(self,x,y):
        return self.root(x)==self.root(y)
    
    def unite(self,x,y):
        if not self.check_same(x,y):
            to_add = self.extra_info.pop(self.root(x))
            self.extra_info[self.root(y)] += to_add
            self.nodes[self.root(x)] = self.root(y)
        
    
def convert_tuple_to_int(tupl, size):
    return tupl[0] + tupl[1]*size
    
#Solved the problem in 0.0465 seconds
#assume that all basins are separated by 9's (otherwise dividing line would 
#arbitrarily pick the basin for it to be in)  
def day9_2(input_file):
    size = 0
    row_no = 0
    with open(input_file) as fi:
        for line in fi:
            if size==0:
                size = len(line.strip())
                sea_floor = np.zeros((size, size))
            row = [int(s) for s in line.strip()]
            sea_floor[row_no] = row
            row_no+=1
    sea_floor = sea_floor[:row_no, :]
    
    #going through twice... yeah, it's a little inconvenient
    uf  = UnionFindTrack()
    for i in range(row_no):
        for j in range(size):
            if sea_floor[i,j]<9:
                uf.add_disconnected(convert_tuple_to_int((i,j), size))
        
    for i in range(row_no):
        for j in range(size):
            if sea_floor[i,j]<9:
                adjacent = get_adjacent_indices(i,j,row_no,size)
                value = sea_floor[i,j]
                for adj in adjacent:
                    if sea_floor[adj]<value:
                        uf.unite(convert_tuple_to_int((i,j), size), convert_tuple_to_int(adj, size))
    basins = list(uf.extra_info.values())
    basins.sort()
    
    return np.product(basins[-3:])



### DAY 10 ###
class Chunk():
    matching_closer = {'(':')', '[':']', '{':'}', '<':'>'}
    
    def __init__(self):
        self.next_closer = []
    
    #return booleans: Corrupted, Complete
    def addToChunk(self,ch):
        if ch in {'(', '[', '{', '<'}:
            self.next_closer.append(self.matching_closer[ch])
        else:
            if (self.next_closer[-1] != ch):
                return True, False
            else:
                self.next_closer.pop()
        if len(self.next_closer)==0:
            return False, True
        return False, False
        
    def isComplete(self):
        return (len(self.next_closer)==0)


class SyntaxChecker():
    illegal_points = {')': 3, ']': 57, '}': 1197, '>': 25137}
    finishing_points = {')': 1, ']': 2, '}': 3, '>': 4}

    def __init__(self):
        self.current = Chunk()
        self.illegal_score = 0
        self.finishing_scores = []
    
    #return true if line needs completing (and is uncorrupted)
    #updates illegal score if line is corrupted    
    def process_test_line(self,ln):
        self.current = Chunk()
        raw = list(ln)
        for r in raw:
            #complete refers to if Chunk is complete
            corrupted, complete = self.current.addToChunk(r)
            if corrupted:
                self.illegal_score+=self.illegal_points[r]
                return False
            if complete:
                self.current = Chunk()
        return not complete
            
        
    #only for part 2
    def completeLine(self):
        points = 0
        waiting = self.current.next_closer
        while waiting:
            points = points*5 + self.finishing_points[waiting.pop()]
        self.finishing_scores.append(points)
        
    def finalFinishingScore(self):
        n = len(self.finishing_scores)
        if not n%2:
            return -1
        self.finishing_scores.sort()
        return self.finishing_scores[n//2]
        

#Solved the problem in 0.0034 seconds
def day10_1(input_file):
    sc = SyntaxChecker()
    with open(input_file) as fi:
        for ln in fi:
            sc.process_test_line(ln.strip())
    
    return sc.illegal_score
    
    
    
#Solved the problem in 0.0035 seconds
def day10_2(input_file):
    sc = SyntaxChecker()
    with open(input_file) as fi:
        for ln in fi:
            needs_finishing = sc.process_test_line(ln.strip())
            if needs_finishing:
                sc.completeLine()
    
    return sc.finalFinishingScore()

### DAY 11 ### 

def convert_int_to_tuple(n, size):
    x = n%size
    y = n//size
    return (x,y)

def get_adjacent_and_diagonal_indices(x,y,sizex, sizey):
    indices = []
    if x>0:
        indices+=[(x-1,y)]
        if y>0:
            indices+=[(x-1, y-1)]
        if y<sizey-1:
            indices+=[(x-1, y+1)]
    if x<sizex-1:
        indices+=[(x+1,y)]
        if y>0:
            indices+=[(x+1, y-1)]
        if y<sizey-1:
            indices+=[(x+1, y+1)]
    if y>0:
        indices+=[(x, y-1)]
    if y<sizey - 1:
        indices+=[(x, y+1)]
        
    return indices

#assume square size
class Octopodes:
    criterion = 9
    
    def __init__(self, size: int):
        self.size = size
        self.octo = np.zeros((size, size))
        #self.flashed = np.zeros((size, size))
        self.epoch = 0
        self.flashing = set()
        self.epoch_done = True
        self.flash_count = 0
        
    def populate_row(self, i:int, r:str):
        digits = [int(s) for s in list(r)]
        self.octo[i] = digits
        
    def perform_scan_step(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.octo[i,j]>self.criterion:
                    tupl =  convert_tuple_to_int((i,j), self.size) 
                    if tupl not in self.flashing:
                        self.flashing.add(tupl)
                        self.flash_count+=1
                        toadd = get_adjacent_and_diagonal_indices(i,j,self.size, self.size)
                        for t in toadd: 
                            self.octo[t]+=1
                        return
        #if we made it through loop without returning, there are no more to flash                
        self.epoch_done = True
                        
                
        
            
    def perform_epoch(self):
        self.epoch_done = False
        #self.flashed = np.zeros((size, size))
        self.flashing = set()
        self.octo += np.ones((self.size, self.size))
        while not self.epoch_done:
            self.perform_scan_step()
        num_flashed = len(self.flashing)
        
        while self.flashing:
            self.octo[convert_int_to_tuple(self.flashing.pop(), self.size)] = 0
        self.epoch+=1
        return (num_flashed == self.size*self.size)


#Solved
def day11_1(input_file):
    o = Octopodes(size=10)
    i=0
    with open(input_file) as fi:
        for ln in fi:
            o.populate_row(i, ln.strip())
            i+=1
    no_epochs = 100
    for e in range(no_epochs):
        o.perform_epoch()
    
    return o.flash_count

#Solved    
def day11_2(input_file):
    o = Octopodes(size=10)
    i=0
    with open(input_file) as fi:
        for ln in fi:
            o.populate_row(i, ln.strip())
            i+=1
    e = 0
    all_flashed = False
    while not all_flashed:
        all_flashed = o.perform_epoch()
        e+=1

    return e



### DAY 12 ### 

class Caves:
    def __init__(self):
        self.caves = {}
        self.paths = set()
        self.path_count = 0
        
        return
        
    def add_connection(self, a):
        cave0, cave1 = a[0], a[1]
        if cave0 in self.caves:
            self.caves[cave0].append(cave1)
        else:
            self.caves[cave0] = [cave1]
        if cave1 in self.caves:
            self.caves[cave1].append(cave0)
        else:
            self.caves[cave1] = [cave0]
        
    def countPaths(self, s,e):
        #reset
        self.path_count = 0
        self.paths = set()
        
        #count
        visited_locked = []
        self.helperCount(s,e,visited_locked, [s])
        return
        
    def helperCount(self, c, e, visited, current_path):
        if c==e:
            self.path_count+=1
            self.paths.add(', '.join(current_path))
            return
        else:
            next_possibilities = self.caves[c]
            for n in next_possibilities:
                if (n.isupper() or n not in visited):
                    self.helperCount(n,e,visited + [c], current_path+[n])
        return
    
    
    
    ##for Part 2    
    def countLeisurelyPaths(self, s,e):
        #reset
        self.path_count = 0
        self.paths = set()
        
        #count
        self.helperLeisurelyCount(s,e,[s], [s], [s])#important: "start" is always locked
        return
        
    def helperLeisurelyCount(self, c, e, visited, visited_locked, current_path):
        if c==e:
            self.path_count+=1
            self.paths.add(','.join(current_path))
            return
        else:
            next_possibilities = self.caves[c]
            
            if c.isupper():
                new_visited = visited
                new_visited_locked = visited_locked
            elif len(visited_locked)>1: #and (c not in visited_locked):
                new_visited = visited
                new_visited_locked = visited_locked + [c]
            elif c in visited:
                new_visited_locked = visited
                new_visited = visited
            else:
                new_visited = visited + [c]
                new_visited_locked = visited_locked
                 
            for n in next_possibilities:
                if n.isupper() or (n not in new_visited_locked):
                    self.helperLeisurelyCount(n,e,new_visited, new_visited_locked, current_path+[n])
        return
        
    
    
    
#Solved the problem in 0.0144 seconds
def day12_1(input_file):
    c = Caves()
    with open(input_file) as fi:
        for ln in fi:
            c.add_connection(ln.strip().split('-'))    
    c.countPaths('start', 'end')   
    
    return c.path_count
    
#Solved the problem in 0.3424 seconds
def day12_2(input_file):
    c = Caves()
    with open(input_file) as fi:
        for ln in fi:
            c.add_connection(ln.strip().split('-'))
    
    c.countLeisurelyPaths('start', 'end')
    
    return c.path_count





### DAY 13 ###
class Paper:
    def __init__(self, maxsize: int = 2000):
        self.size = (maxsize, maxsize)
        self.pages = np.zeros((maxsize, maxsize))
        self.max_x = -1
        self.max_y = -1
        
    def add_dot(self,x,y):
        self.pages[(y,x)] = 1
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)
        return
    
    #fold up
    #e.g. y = 7 
    #does not keep it a square
    def fold_y(self, y):
        old_size = self.size[0]
        new_size = max(old_size - y, y)
        
        start_displaying_back = new_size - y -1
        start_counting_front = new_size - (old_size - y)
        new_pages = np.zeros((new_size, self.size[1]))
        
        for i in range(new_size):
            if i>=start_displaying_back:
                back_row = self.pages[i - start_displaying_back]
            else:
                back_row = np.zeros((self.size[1]))
            if i>=start_counting_front:
                front_row = self.pages[old_size -1 - (i-start_counting_front)]
            else:
                front_row = np.zeros((self.size[1]))
            new_pages[i] = np.maximum(front_row, back_row)
        self.pages = new_pages[:-1,:]
        self.size = (new_size-1, self.size[1])

        return

    #fold left    
    def fold_x(self, x):
        old_size = self.size[1]
        new_size = max(old_size - x, x)
        
        start_displaying_back = new_size - x -1
        start_counting_front = new_size - (old_size - x)
        new_pages = np.zeros((self.size[0], new_size))
        
        for i in range(new_size):
            if i>=start_displaying_back:
                back_c = self.pages[:, i - start_displaying_back]
            else:
                back_c = np.zeros((self.size[0]))
            if i>=start_counting_front:
                front_c = self.pages[:, old_size -1 - (i-start_counting_front)]
            else:
                front_c = np.zeros((self.size[0]))
            new_pages[:,i] = np.maximum(front_c, back_c)
        self.pages = new_pages[:, :-1]
        self.size = (self.size[0], new_size-1)
        return
        
    def count_dots(self):
        return int(np.sum(self.pages))
    
    #from initial square to a rectangle  
    def reduce_size(self):
        self.size = (self.max_y +1,  self.max_x + 1)
        self.pages = self.pages[0:self.size[0],0:self.size[1]]
        return
                
        
#Solved the problem in 0.0188 seconds
def day13_1(input_file):
    pattern1 = re.compile('(\d+),(\d+)')
    pattern2 = re.compile('fold along (x|y)=(\d+)')
    
    paper = Paper()
    
    with open(input_file) as fi:
        for line in fi:
            match1 = pattern1.match(line)
            if match1:
                paper.add_dot(int(match1[1]), int(match1[2]))
                
            #first fold only!    
            else:
                match2 = pattern2.match(line)
                if match2:
                    if match2[1]=='y':
                        paper.fold_y(int(match2[2]))
                    else:
                        paper.fold_x(int(match2[2]))
                    return paper.count_dots()
                else:
                    paper.reduce_size()
            
    return -1


#had to take a new approach for part 2:
class Dots:
    def __init__(self):
        self.dots = set()
        self.max_x = -1
        self.max_y = -1
        
    def add_dot(self,x,y):
        self.dots.add((y,x))
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)
        return
    
    #fold up
    #e.g. y = 7 
    #does not keep it a square
    def fold_y(self, y):
        new_dots = set()
        for d in self.dots:
            oldy,x = d
            if oldy<y:
                new_dots.add((oldy, x))
            elif oldy>y:
                newy = y - (oldy-y)
                new_dots.add((newy, x))
        self.max_y = y-1
        self.dots = new_dots
        return

    #fold left    
    def fold_x(self, x):
        new_dots = set()
        for d in self.dots:
            y, oldx = d
            if oldx<x:
                new_dots.add((y,oldx))
            elif oldx>x:
                newx = x - (oldx-x)
                new_dots.add((y,newx))
        self.max_x = x-1
        self.dots = new_dots
        return
        
    def count_dots(self):
        return len(self.dots)

    def pretty_print(self):
        pages = np.zeros((self.max_y+1, self.max_x + 1))
        for d in self.dots:
            pages[d]=1
            
        convert_dict = {0:".", 1:"#"}
        for i in range(pages.shape[0]):
            newline = "".join([convert_dict[int(e)] for e in pages[i]])
            print(newline)

#Solved the problem in 0.0051 seconds
def day13_2(input_file):
    pattern1 = re.compile('(\d+),(\d+)')
    pattern2 = re.compile('fold along (x|y)=(\d+)')
    
    paper = Dots()
    
    with open(input_file) as fi:
        for line in fi:
            match1 = pattern1.match(line)
            if match1:
                paper.add_dot(int(match1[1]), int(match1[2]))
                    
            else:
                match2 = pattern2.match(line)
                if match2:
                    if match2[1]=='y':
                        paper.fold_y(int(match2[2]))
                    else:
                        paper.fold_x(int(match2[2]))
            
    paper.pretty_print()    
            
    return -1


### DAY 14 ### 
class Polymer:

    def __init__(self, starting: str):
        self.polymer = starting
        self.epochs = 0
        self.rules = {}
        
    def add_rule(self, rule: str):
        if len(rule)<3:
            return
        self.rules[rule[:2]] = rule[-1]
    
    def progress_epoch(self):
        new_polymer = list(self.polymer)
        to_add = self.calculate_adds()
        added = 0
        for t in to_add:
            new_polymer.insert(t+added, to_add[t])
            added+=1
        self.polymer = "".join(new_polymer)
    
    def calculate_adds(self):
        to_add = {}
        for i in range(len(self.polymer) - 1):
            if self.polymer[i:i+2] in self.rules:
                to_add[i+1] = self.rules[self.polymer[i:i+2]]
        return to_add
        
    def calculate_max_min_counts(self):
        counts = {}
        for i in range(len(self.polymer)):
            ltr = self.polymer[i]
            if ltr in counts:
                counts[ltr]+=1
            else:
                counts[ltr] = 1
        
        only_counts = list(counts.values())
        
        return max(only_counts), min(only_counts)
        
#Solved the problem in 0.0356 seconds
def day14_1(input_file):

    with open(input_file) as fi:
        p = Polymer(fi.readline().strip())
        for ln in fi:
            p.add_rule( ln.strip())

    epochs = 10
    for i in range(epochs):
        p.progress_epoch()

    mx, mn = p.calculate_max_min_counts()    
    return mx-mn



class BetterPolymer:

    def __init__(self, bgn: str):
        self.polymer_pairs = {}
        self.polymer_letters = {}
        self.epochs = 0
        self.rules = {}
        self.length = len(bgn)
        for i in range(len(bgn)-1):
            self.polymer_letters[bgn[i]] = self.polymer_letters.get(bgn[i], 0) + 1
            self.polymer_pairs[bgn[i:i+2]] = self.polymer_pairs.get(bgn[i:i+2],0)+1    
        self.polymer_letters[bgn[-1]] = self.polymer_letters.get(bgn[-1], 0) + 1  
        
        
    def add_rule(self, rule: str):
        if len(rule)<3:
            return
        self.rules[rule[:2]] = rule[-1]
    
    def progress_epoch(self):
        letters, pairs = self.calculate_adds()

        for l in letters:
            self.polymer_letters[l] = self.polymer_letters.get(l,0)+letters[l]
        for p in pairs:
            self.polymer_pairs[p] = self.polymer_pairs.get(p,0) + pairs[p]
        self.epochs+=1
        self.length = sum(list(self.polymer_letters.values()))
    
    
    def calculate_adds(self):
        letters_to_add = {}
        pairs_to_change = {}
        for pair in self.polymer_pairs:
            if pair in self.rules:
                letters_to_add[self.rules[pair]] = letters_to_add.get(self.rules[pair],0)+ self.polymer_pairs[pair]
                begin_pair = pair[0]+self.rules[pair]
                end_pair = self.rules[pair] + pair[1]
                
                pairs_to_change[pair] = pairs_to_change.get(pair,0)-1*self.polymer_pairs[pair]
                pairs_to_change[begin_pair] = pairs_to_change.get(begin_pair,0)+self.polymer_pairs[pair]
                pairs_to_change[end_pair] = pairs_to_change.get(end_pair,0)+self.polymer_pairs[pair]
                
        return letters_to_add, pairs_to_change
        
    def calculate_max_min_counts(self):
        
        only_counts = list(self.polymer_letters.values())
        
        return max(only_counts), min(only_counts)

#Solved the problem in 0.0049 seconds
def day14_2(input_file):
    with open(input_file) as fi:
        p = BetterPolymer(fi.readline().strip())
        for ln in fi:
            p.add_rule( ln.strip())

    epochs = 40
    for i in range(epochs):
        p.progress_epoch()

    mx, mn = p.calculate_max_min_counts()    
    return mx-mn





### DAY 15 ###
#Solved the problem in 0.0023 seconds
def day15_1(input_file):
    size = 0
    row_no = 0
    with open(input_file) as fi:
        for line in fi:
            if size==0:
                size = len(line.strip())
                caves = np.zeros((size*10, size))
            row = [int(s) for s in line.strip()]
            caves[row_no] = row
            row_no+=1
    caves = caves[:row_no, :]
    
    #first pass: assume you only move right and down.
    risk = np.zeros((row_no, size))
    risk[0,0] = 0
    for j in range(1,size):
        risk[0, j] = caves[0, j] + risk[0, j-1]
    for i in range(1,row_no):
        risk[i, 0] = caves[i, 0] + risk[i-1, 0]
    
    for i in range(1,row_no):
        for j in range(1,size):
            risk[i,j] = min(risk[i-1, j], risk[i, j-1]) + caves[i,j]
    

    print(risk)
    return int(risk[row_no - 1, size-1])
    
    
#Solved the problem in 12.0472 seconds   
def day15_2(input_file):
    size = 0
    row_no = 0
    with open(input_file) as fi:
        for line in fi:
            if size==0:
                size = len(line.strip())
                smallcaves = np.zeros((size*10, size))
            row = [int(s) for s in line.strip()]
            smallcaves[row_no] = row
            row_no+=1
    smallcaves = smallcaves[:row_no, :]
    
    caves = np.zeros((row_no*5, size*5))
    for i in range(5):
        for j in range(5):
            rescaled = (smallcaves + i + j -1)%9 + 1
            caves[(i)*row_no:(i+1)*row_no, j*size : (j+1)*size] = rescaled
    m = row_no * 5
    n = size * 5
    
    #okay we can't assume we only go right and down; have to do full Dijsktra
    risk = -np.ones((m, n))
    risk[0,0] = 0
    next_to_add = {(1,0): caves[1,0], (0,1): caves[0,1]}
    
    while next_to_add:
        current = min(next_to_add, key=next_to_add.get)
        risk[current] = next_to_add.pop(current)
        indices_to_check = get_adjacent_indices(current[0], current[1], m, n)
        for i in indices_to_check:
            if risk[i]<0 and i not in next_to_add:
                next_to_add[i] = risk[current] + caves[i] 

    return int(risk[m - 1, n-1])
    

if __name__ == '__main__':

    if len(sys.argv)!=3:
        print("You must specify the problem in the format dayN_P and the input as a text file.")
        exit()
 
    tic = time.perf_counter()
    
    function_name = sys.argv[1]
    input = sys.argv[2]  
    #input = "input.txt"
    #input = "test.txt"
    print(eval(function_name)(input))
    
    toc = time.perf_counter()

    print(f"Solved the problem in {toc - tic:0.4f} seconds")


