"""2021_AoC_Wk1.py
Solutions to days 1-7 of 2021 Advent of Code.
Refer to that website for problem statements.
All solutions arrived at independently.

Python version: Python 3.7.11
To run from the command line, specify the problem with format dayN_P 
and provide the appropriate input file obtained from AoC or test file of your own creation. Example:
>> python 2021_AoC_Wk1.py day4_2 input_4.txt
Take care to provide the right input files, corresponding to the correct day, 
as input of an unexpected format will produce a world of errors, including unending loops.

All expected answers are positive integers, so -1 was used as a default output 
some of the times if the function were to end in an unexpected state. 

Comments: the bingo board simulation on day 4 was FUN.
"""

import sys
import time
import re #day 2
import numpy as np #day 4

### DAY 1 ###

#0.0023 seconds
def day1_1(input_file):
    decreasing = 0
    prev = None
    current = None
    with open(input_file) as fi:
        for line in fi:
            current = int(line)
            if prev and (prev<current):
                decreasing +=1
            prev = current
            
    return decreasing
 
#0.0008 seconds
def day1_2(input_file):
    decreasing = 0
    prev_2 = None
    prev_1 = None
    current = None
    
    prev_sum = None
    current_sum = None
    
    with open(input_file) as fi:
        for line in fi:
            current = int(line)
            if prev_2:
                current_sum = prev_2 + prev_1 + current
                if prev_sum and prev_sum<current_sum:
                    decreasing +=1
            prev_2 = prev_1
            prev_1 = current
            prev_sum = current_sum
            
    return decreasing


### DAY 2 ###

#submarine class for problem 2.1
class Submarine_v0:

    pattern = re.compile('([a-z]+)\s(\d+)')
    
    def __init__(self, start_depth = 0, start_horiz = 0):
        self.depth = start_depth
        self.horiz = start_horiz
        
    def forward(self, n : int):
        self.horiz += n
        
    def down(self, n : int):
        self.depth += n
        
    def up(self, n : int):
        self.depth = self.depth - n
        
    def product(self) -> int:
        return (self.depth)*(self.horiz)

    def process_nav_instruction(self, l):
        processed = Submarine.pattern.findall(l)
        word = processed[0][0]
        n = int(processed[0][1])
        if word=='forward':
            self.forward(n)
        elif word=='down':
            self.down(n)
        elif word=='up':
            self.up(n)
            
    def print_location(self):
        print(f"The submarine is at a depth of {self.depth} with horizational location {self.horiz}.")


class Submarine:

    pattern = re.compile('([a-z]+)\s(\d+)')
    
    def __init__(self, start_depth = 0, start_horiz = 0, start_aim = 0):
        self.depth = start_depth
        self.horiz = start_horiz
        self.aim = start_aim
        
    def forward(self, n : int):
        self.horiz += n
        self.depth += n*self.aim
        
    def down(self, n : int):
        self.aim += n
        
    def up(self, n : int):
        self.aim = self.aim - n
        
    def product(self) -> int:
        return (self.depth)*(self.horiz)

    #assuming perfect instructions, without any errors
    def process_nav_instruction(self, l):
        processed = Submarine.pattern.findall(l)
        word = processed[0][0]
        n = int(processed[0][1])
        if word=='forward':
            self.forward(n)
        elif word=='down':
            self.down(n)
        elif word=='up':
            self.up(n)
            
    def print_location(self):
        print(f"The submarine is at a depth of {self.depth} with horizational location {self.horiz}.")


#0.0014 seconds for first and second parts
#only change from first to second part was the submarine class
def day2_1(input_file):
    sbm = Submarine_v0()
    
    with open(input_file) as fi:
        for line in fi:
            sbm.process_nav_instruction(line)
            
    return sbm.product()

def day2_2(input_file):
    sbm = Submarine()
    
    with open(input_file) as fi:
        for line in fi:
            sbm.process_nav_instruction(line)
            
    return sbm.product()


### DAY 3 ###
def convert_from_binary(bint):
    power = 1
    output = 0
    while bint:
        output+=(bint%10)*power
        bint = bint//10
        power = power*2
    return output

class RateCounter:
    def __init__(self,n):
        self.counts = {i : 0 for i in range(n)}
        self.length = n
        self.processed = 0
        
    
    def process(self,n):
        cnt = 0
        while n:
            if n%10==1:
                self.counts[cnt]+=1
            n = n//10
            cnt +=1
        self.processed+=1
            
    
    def reconstruct(self):
        epsilon = 0
        gamma = 0
        cnt = self.length
        while cnt>0:
            ones = self.counts[cnt-1]
            if ones>self.processed//2:
                gamma +=1
            else:
                epsilon += 1
            gamma = gamma*10
            epsilon = epsilon*10
            cnt = cnt - 1
        gamma = gamma//10
        epsilon = epsilon//10
        return convert_from_binary(gamma)*convert_from_binary(epsilon)

class TreeNode:
    def __init__(self, val: int, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
   
class BinaryTree:
    def __init__(self, depth):
        #breadth first...
        self.root = TreeNode(val = 1)
        self.depth = depth
       
    def traverse_add(self, address: str):
        current = self.root
        for i in range(self.depth):
            adr = address[i]
            current.val+=1
            if adr=="0":
                node_next = current.left
                if node_next:
                    current = node_next
                else:
                    current.left = TreeNode(val=0)
                    current = current.left
            if adr=="1":
                node_next = current.right
                if node_next:
                    current = node_next
                else:
                    current.right = TreeNode(val=0)
                    current = current.right
            
    def traverse_step_check(self, go_towards_common : bool, start_node : TreeNode , record : str):
        #print(f"We are traversing with {start_node.val} remaining and record {record}.")
        if not start_node.left:
            record+="1"
            node = start_node.right
        elif not start_node.right:
            record+="0"
            node = start_node.left
        elif go_towards_common == (start_node.left.val>start_node.right.val):
        #includes tie breaker case where go_towards_common = false and vals equal
            node = start_node.left
            record+="0"
        else: 
        #includes tie breaker case where go_towards_common = true and vals equal
            node = start_node.right
            record+="1"
            
        if node.val>1:
            return (node, record)
        else:#search til end of tree!
            searching = True
            while searching:
                if node.left:
                    record+="0"
                    node = node.left
                if node.right :
                    record+="1"
                    node = node.right
                if (not node.right) and (not node.left):
                    searching = False
                    return (None, record)
       

class SuperRateCounter:
    def __init__(self,n):
        self.counts = {i : 0 for i in range(n)}
        self.length = n
        self.processed = 0
        self.tree = BinaryTree(depth = n)
        #this binary tree at each node states how many terms are in tree!
    
    def process(self,n_string):
        self.tree.traverse_add(n_string)
        
        n = int(n_string)
        cnt = 0
        while n:
            if n%10==1:
                self.counts[cnt]+=1
            n = n//10
            cnt +=1
        self.processed+=1
            
    
    def reconstruct(self):
        oxygen = ""
        cotwo = ""
        oxygen_node = self.tree.root
        cotwo_node = self.tree.root
        
        #find oxygen
        while oxygen_node:
            oxygen_node, oxygen = self.tree.traverse_step_check(go_towards_common = True, start_node = oxygen_node, record = oxygen)
        #find carbon dioxide
        while cotwo_node:
            cotwo_node, cotwo = self.tree.traverse_step_check(go_towards_common = False, start_node = cotwo_node, record = cotwo)
        print(oxygen, cotwo)
        
        return convert_from_binary(int(oxygen))*convert_from_binary(int(cotwo))

#0.0026 seconds
def day3_1(input_file,length = 5):
    rc = RateCounter(length)
    
    with open(input_file) as fi:
        for line in fi:
            rc.process(line.strip()) 
    return rc.reconstruct()
    
#0.0096 seconds    
def day3_2(input_file,length = 5):
    src = SuperRateCounter(length)
    
    with open(input_file) as fi:
        for line in fi:
            src.process(line)
            
    return src.reconstruct()
    
    
### DAY 4 ###

class BingoBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = np.ones((size, size))
        self.markings = np.ones((size, size))
        self.locations = {}
        self.bingo = -1
        
    def populate_row(self, row: str, row_index : int):
        str_vals = row.split()
        column_index = 0
        for s in str_vals:
            self.board[row_index][column_index] = int(s)
            self.locations[int(s)] = (row_index, column_index)
            column_index+=1
    
    def mark(self, val):
        if val not in self.locations:
            return False
        
        ri, ci = self.locations[val]
        self.markings[ri][ci] = 0
    
        #check for bingo!
        if (self.markings[:,ci]==np.zeros((self.size))).all():
            self.bingo = val
            return True
        if (self.markings[ri,:]==np.zeros((self.size))).all():
            self.bingo = val
            return True
        
        return False
        
     
    def calculate_score(self):
        return self.bingo*np.sum(self.markings * self.board)

#added after the fact to refactor code
class BingoGame:
    def __init__(self):
        self.bingo_calls = []
        self.no_bingo_boards = -1
        self.bingo_boards = {}
        self.board_size = -1
        
    def populate(self, input_file):
        bingo_board_next_line_no = 0
        with open(input_file) as fi:

            for line in fi:
                if self.no_bingo_boards == -1:
                    self.bingo_calls = [int(s) for s in line.split(",")]
                    self.no_bingo_boards = 0
                
                elif len(line.strip())==0:
                    continue
                
                else:
                    #new bingo board
                    if bingo_board_next_line_no == 0:
                        self.bingo_board_size = len(line.split())
                        #print("line", line.strip(), "size", bingo_board_size)
                        self.no_bingo_boards+=1
                        temp = BingoBoard(size = self.bingo_board_size)
                        temp.populate_row(row = line, row_index = bingo_board_next_line_no)
                        self.bingo_boards[self.no_bingo_boards] = temp
                    
                    #continuing bingo board
                    else:
                        temp = self.bingo_boards[self.no_bingo_boards]
                        temp.populate_row(row = line, row_index = bingo_board_next_line_no)
                        self.bingo_boards[self.no_bingo_boards] = temp
                
                    bingo_board_next_line_no+=1
                    if bingo_board_next_line_no==self.bingo_board_size:
                        bingo_board_next_line_no = 0
        
    

#0.0095 seconds
def day4_1(input_file):
    game  = BingoGame()
    
    #input
    game.populate(input_file)
                
    #play rounds
    for c in game.bingo_calls:
        for b in range(1,game.no_bingo_boards+1):
            bingo_maybe = game.bingo_boards[b].mark(c)
            #print(f"We have just called {c} and are looking at board {b}.")
            #print(bingo_boards[b].markings)
            if bingo_maybe:
                return game.bingo_boards[b].calculate_score()
                
    return -1



#0.0152 second
#why yes it would be more efficient to not copy paste everything     
def day4_2(input_file):
    game  = BingoGame()
    winning_boards = 0
    
    #input
    game.populate(input_file)
                
    #play rounds
    for c in game.bingo_calls:
        for b in list(game.bingo_boards.keys()):
            bingo_maybe = game.bingo_boards[b].mark(c)
            if bingo_maybe and winning_boards< game.no_bingo_boards - 1:
                winning_boards+=1
                game.bingo_boards.pop(b)
            elif bingo_maybe:
                return game.bingo_boards[b].calculate_score()
                
    return -1
    

### DAY 5 ###

#horizontal and vertical lines only
def get_points_on_line_p1(p0, p1):
    if p0 == p1:
        return [p0]
    x0, y0 = p0
    x1, y1 = p1
    output = []
    if x1==x0:
        for y in range(min(y0, y1), max(y0, y1)+1):
            output.append((x0, y))
    if y1==y0:
        for x in range(min(x0, x1), max(x0, x1)+1):
            output.append((x, y0))
    return output

#Solved the problem in 0.0625 seconds
def day5_1(input_file):
    pattern = re.compile('(\d+),(\d+) -> (\d+),(\d+)')
    size = 1000
    sea_floor = np.zeros((size, size))
    
    with open(input_file) as fi:
        for line in fi:
            ans = pattern.findall(line)
            p0 = (int(ans[0][0]), int(ans[0][1]))
            p1 = (int(ans[0][2]), int(ans[0][3]))
            new_points = get_points_on_line_p1(p0, p1)
            for p in new_points:
                sea_floor[p] +=1
    return np.sum(sea_floor>1)

#including diagonal lines  
def get_points_on_line_p2(p0, p1):
    if p0 == p1:
        return [p0]
    x0, y0 = p0
    x1, y1 = p1
    output = []
    if x1==x0:
        for y in range(min(y0, y1), max(y0, y1)+1):
            output.append((x0, y))
    if y1==y0:
        for x in range(min(x0, x1), max(x0, x1)+1):
            output.append((x, y0))
    if abs(x1-x0) == abs(y1-y0):
        x_step = np.sign(x1 - x0)
        y_step = np.sign(y1 - y0)
        x = x0
        y = y0
        while y!=y1:
            output.append((x,y))
            x+=x_step
            y+=y_step
        output.append((x,y))
            
    return output

#Solved the problem in 0.1436 seconds
def day5_2(input_file):
    pattern = re.compile('(\d+),(\d+) -> (\d+),(\d+)')
    size = 1000
    sea_floor = np.zeros((size, size))
    
    with open(input_file) as fi:
        for line in fi:
            ans = pattern.findall(line)
            p0 = (int(ans[0][0]), int(ans[0][1]))
            p1 = (int(ans[0][2]), int(ans[0][3]))
            new_points = get_points_on_line_p2(p0, p1)
            for p in new_points:
                sea_floor[p] +=1
    return np.sum(sea_floor>1)






### DAY 6 ###

def compute_fish_count(fishies):
    count = 0
    for s in fishies:
        count+= s
        
    return count
    
def fish_after_day(fishies):
    new_fishies = [0]*9
    for s in range(1,len(fishies)):
        new_fishies[s-1] = fishies[s]
    new_fishies[8] += fishies[0]
    new_fishies[6] += fishies[0]
            
    return new_fishies
            
    

#part 1: Solved the problem in 0.0002 seconds
def day6_1(input_file):
    return day6(input_file, part=1)

#part 2: Solved the problem in 0.0005 seconds
def day6_2(input_file):
    return day6(input_file, part=2)


def day6(input_file, part = 2):
    final_day = 80
    if part==2:
        final_day = 256
    with open(input_file) as fi:
        input_string = input_string = fi.read()
        fish_list = [int(s) for s in input_string.split(",")]
        
    fishies = [0]*9
    
    for s in fish_list:
        fishies[s] += 1
        
    day = 0
    while day<final_day:
        fishies = fish_after_day(fishies)
        day+=1
        #print("total number fishies: ", sum(fishies))
        #print("\nday: ",day)

        
    return sum(fishies)

### DAY 7 ###

def compute_fuel(crabs, align_pos):
    return int(sum(abs(crabs-align_pos)))

#Solved the problem in 0.0545 seconds
def day7_1(input_file):
    with open(input_file) as fi:
        input_string = input_string = fi.read()
        crabs = [int(s) for s in input_string.split(",")]

    closest = round(np.mean(crabs))
    current = closest
    fuel_cost = compute_fuel(crabs, current)
    left = current - 1
    right = current + 1
    while compute_fuel(crabs, left)<fuel_cost:
        fuel_cost = compute_fuel(crabs, left)
        current = left
        left = left - 1
    while compute_fuel(crabs, right)<fuel_cost:
        fuel_cost = compute_fuel(crabs, right)
        current = right
        right = right + 1 
    return fuel_cost
    
def compute_fuel_weighted(crabs, align_pos):
    raw = abs(crabs-align_pos)
    return int(sum([i*(i+1)/2 for i in raw]))


#Solved the problem in 0.0044 seconds
def day7_2(input_file):
    with open(input_file) as fi:
        input_string = input_string = fi.read()
        crabs = [int(s) for s in input_string.split(",")]

    closest = round(np.mean(crabs))
    current = closest
    fuel_cost = compute_fuel_weighted(crabs, current)
    left,right = current - 1,current+1
    while compute_fuel_weighted(crabs, left)<fuel_cost:
        fuel_cost = compute_fuel_weighted(crabs, left)
        current = left
        left = left - 1
    while compute_fuel_weighted(crabs, right)<fuel_cost:
        fuel_cost = compute_fuel_weighted(crabs, right)
        current = right
        right = right + 1 
    return fuel_cost 
    
    
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


