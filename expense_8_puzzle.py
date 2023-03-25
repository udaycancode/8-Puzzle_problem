# ***********************************************************
# python-library
# ***********************************************************

import sys
from datetime import datetime


# ***********************************************************
# object
# ***********************************************************

class Node:
    def __init__(self, data=[], parent=object, hueristic=0, depth=0, move="", cost=0):
        self.data = data  # state/matrix
        self.parent = parent
        self.hueristic = hueristic
        self.depth = depth  # depth level
        self.move = move  # up,down,right,left
        self.cost = cost  # number or latter


# ***********************************************************
# global variable
# ***********************************************************

n_n_popped = 0
n_n__expanded = 0
n_n_generated = 0
max_fringe_size = 0

cost = 0
depth_level = 0

steps = []

method = "a*"
dump_flag = False
d_l = 0

fringe = []
visited = []

iteration = 1

now = datetime.now()
dt_string = now.strftime("%m_%d_%Y-%H_%M_%S")

# ***********************************************************
# input
# ***********************************************************

input_file = sys.argv[1]  # Taking input file from command line arguments
output_file = sys.argv[2]  # Taking output file from command line arguments

try:
    if sys.argv[3] == "True" or sys.argv[3] == "true":
        dump_flag = True
    # more complex
    list_algo = ["a*", "bfs", "dfs", "ucs", "greedy", "ids", "dls"]
    if sys.argv[3] == "dls":
        method = "dls"
        d_l = int(input("depth limit : "))
    elif sys.argv[3] in list_algo:
        method = str(sys.argv[3])
except:
    pass

try:
    if sys.argv[4] == "True" or sys.argv[4] == "true":
        dump_flag = True
except:
    pass

infile = open(input_file, 'r')  # Reading the input file in read mode
oufile = open(output_file, 'r')
input_data = infile.read().split('\n')
output_data = oufile.read().split('\n')

input_p_8 = []
output_p_8 = []
for i in range(3):
    input_p_8.append(list(map(int, input_data[i].split())))
    output_p_8.append(list(map(int, output_data[i].split())))


# ***********************************************************
# methods
# ***********************************************************

def print_data_in_matrix(matrix):
    for i in matrix:
        print(i)


def zero_position(matrix):
    for i in range(3):
        for j in range(3):
            if matrix[i][j] == 0:
                return i, j


# to find and number position
def position(matrix, data):
    for i in range(3):
        for j in range(3):
            if matrix[i][j] == data:
                return i, j


# menhaten
def hueristic_2(matrix):
    h = 0
    for i in range(3):
        for j in range(3):
            if matrix[i][j] != output_p_8[i][j]:
                x, y = position(output_p_8, matrix[i][j])
                h += (abs(x - i) + abs(y - j)) * matrix[i][j]

    return h


def copy_matrix(matrix):
    temp_matrix = []
    for i in matrix:
        temp_array = []
        for j in i:
            temp_array.append(j)
        temp_matrix.append(temp_array)
    return temp_matrix


def make_child(matrix):
    global n_n_generated
    x, y = zero_position(matrix.data)

    number_p_m = [[x - 1, y], [x, y - 1], [x + 1, y], [x, y + 1]]

    matrix_childern = []
    for i in number_p_m:

        if i[0] >= 0 and i[1] >= 0 and i[0] <= 2 and i[1] <= 2:
            n_n_generated += 1
            chile_node = Node()
            temp_matrix = copy_matrix(matrix.data)
            temp_matrix[i[0]][i[1]], temp_matrix[x][y] = temp_matrix[x][y], temp_matrix[i[0]][i[1]]
            chile_node.data = temp_matrix
            chile_node.cost = int(temp_matrix[x][y]) + matrix.cost
            if method == "ucs":
                chile_node.hueristic = matrix.hueristic + int(temp_matrix[x][y])
            elif method == "a*":

                chile_node.hueristic = hueristic_2(temp_matrix) + chile_node.cost
            elif method == "greedy":
                chile_node.hueristic = hueristic_2(temp_matrix)

            else:
                chile_node.hueristic = 0
            chile_node.parent = matrix
            chile_node.depth = matrix.depth + 1
            if i[0] < x and i[1] == y:
                chile_node.move = f" {temp_matrix[x][y]} down"
            elif i[0] == x and i[1] < y:
                chile_node.move = f" {temp_matrix[x][y]} right"
            elif i[0] > x and i[1] == y:
                chile_node.move = f" {temp_matrix[x][y]} up"
            elif i[0] == x and i[1] > y:
                chile_node.move = f" {temp_matrix[x][y]} left"

            matrix_childern.append(chile_node)

    return matrix_childern


def backtrack_printing_data(last_node):
    global cost, depth_level
    depth_level = last_node.depth
    cost = last_node.cost
    node = last_node
    while node.parent != "root":
        steps.insert(0, node.move)

        node = node.parent
    print_output()


def print_output():
    print(f"Nodes Popped: {n_n_popped}")
    print(f"Nodes Expanded: {n_n__expanded}")
    print(f"Nodes Generated: {n_n_generated}")
    print(f"Max Fringe Size: {max_fringe_size}")
    print(f"Solution Found at depth {depth_level} with cost of {cost}.")
    print("Steps:")
    for i in steps:
        print(f"    Move {i}")


def trak_file():
    global iteration
    file1 = open("trace-" + dt_string + ".txt", "a")
    file1.write(f"#####################->  iteration : {iteration} <-###########################\n")
    file1.write(f"Nodes Popped: {n_n_popped}\n")
    file1.write(f"Nodes Expanded: {n_n__expanded}\n")
    file1.write(f"Nodes Generated: {n_n_generated}\n")
    file1.write("fringe : \n")
    ll = 0
    for i in fringe:
        if method == "a*" or method == "greedy" or method == "ucs":
            i = i["obj"]
        file1.write(
            f"\t\t\t{str(ll)} -> [current node : {i.data}, parent node : {i.parent.data}, hueristic : {i.hueristic}, current node depth : {i.depth}, move : {i.move}, current node cost : {i.cost}\n]")
        ll += 1
    ll = 0
    file1.write("visited : \n")
    for j in visited:
        file1.write(f"\t\t\t{str(ll)}-> node : {j}\n")
        ll += 1

    file1.write("\n\n")
    iteration += 1
    file1.close()


# ***********************************************************
# track file
# ***********************************************************

if dump_flag == True:
    file1 = open("trace-" + dt_string + ".txt", "a")
    file1.write("SUBJECT : 5360 : Artificial Intelligence(Monday - Wednesday Sections)\n")
    file1.write("Assignment 1\n")
    file1.write("Task 1 : expense_8_puzzle(Uninformed & Informed Search)\n\n")
    file1.write("-----------------------------------------------------------\n")
    file1.write("start file : \t\t goal file : \n")
    for i, j in zip(input_data[:3], output_data[:3]):
        file1.write(i + "\t\t\t\t\t" + j + "\n")
    file1.write(f"\nMETHOD : {method}\n")
    file1.write(f"dump_flag = {dump_flag}\n\n")
    file1.write("-----------------------------------------------------------\n")

    file1.write("Processoing...... : \n\n")
    file1.close()


# ***********************************************************
# all algorithms
# ***********************************************************

def astar():
    global n_n_popped, n_n__expanded, max_fringe_size, fringe
    start = Node(data=input_p_8, parent="root", hueristic=0, depth=0, cost=0)
    fringe.append({"h": start.hueristic, "obj": start})
    temp_max_fringe_size = 1
    print("a*")
    while True:
        fringe = sorted(fringe, key=lambda d: d["h"])
        current_node = fringe[0]
        n_n_popped += 1

        if current_node["obj"].data == output_p_8:
            backtrack_printing_data(current_node["obj"])
            break
        if current_node["obj"].data not in visited:
            n_n__expanded += 1

            for i in make_child(current_node["obj"]):
                fringe.append({"h": i.hueristic, "obj": i})
                temp_max_fringe_size += 1

        del fringe[0]
        temp_max_fringe_size -= 1
        if temp_max_fringe_size > max_fringe_size:
            max_fringe_size = temp_max_fringe_size
        visited.append(current_node["obj"].data)
        if dump_flag == True:
            trak_file()


def greedy():
    global n_n_popped, n_n__expanded, max_fringe_size, fringe
    start = Node(data=input_p_8, parent="root", hueristic=0, depth=0, cost=0)
    fringe.append({"h": start.hueristic, "obj": start})
    temp_max_fringe_size = 1
    print("greedy")
    while True:
        fringe = sorted(fringe, key=lambda d: d["h"])
        current_node = fringe[0]
        n_n_popped += 1

        if current_node["obj"].data == output_p_8:
            backtrack_printing_data(current_node["obj"])
            break
        if current_node["obj"].data not in visited:
            n_n__expanded += 1

            for i in make_child(current_node["obj"]):
                fringe.append({"h": i.hueristic, "obj": i})
                temp_max_fringe_size += 1

        del fringe[0]
        temp_max_fringe_size -= 1
        if temp_max_fringe_size > max_fringe_size:
            max_fringe_size = temp_max_fringe_size
        visited.append(current_node["obj"].data)
        if dump_flag == True:
            trak_file()


def bfs():
    global n_n_popped, n_n__expanded, max_fringe_size, fringe
    start = Node(data=input_p_8, parent="root")
    fringe.append(start)
    temp_max_fringe_size = 1
    while True:
        current_node = fringe[0]
        n_n_popped += 1

        if current_node.data == output_p_8:
            backtrack_printing_data(current_node)
            break
        if current_node.data not in visited:
            n_n__expanded += 1
            for i in make_child(current_node):
                fringe.append(i)
                temp_max_fringe_size += 1

        del fringe[0]
        temp_max_fringe_size -= 1
        if temp_max_fringe_size > max_fringe_size:
            max_fringe_size = temp_max_fringe_size
        visited.append(current_node.data)
        if dump_flag == True:
            trak_file()


def ucs():
    global n_n_popped, n_n__expanded, max_fringe_size, fringe
    start = Node(data=input_p_8, parent="root", hueristic=0, depth=0, cost=0)
    fringe.append({"h": start.hueristic, "obj": start})
    temp_max_fringe_size = 1
    print("ucs running.....")
    while True:
        fringe = sorted(fringe, key=lambda d: d["h"])
        current_node = fringe[0]
        n_n_popped += 1



        if current_node["obj"].data == output_p_8:
            backtrack_printing_data(current_node["obj"])
            break
        if current_node["obj"].data not in visited:
            n_n__expanded += 1

            for i in make_child(current_node["obj"]):
                fringe.append({"h": i.hueristic, "obj": i})
                temp_max_fringe_size += 1

        del fringe[0]
        temp_max_fringe_size -= 1
        if temp_max_fringe_size > max_fringe_size:
            max_fringe_size = temp_max_fringe_size
        visited.append(current_node["obj"].data)
        if dump_flag == True:
            trak_file()


def dfs():
    global n_n_popped, n_n__expanded, max_fringe_size, fringe
    start = Node(data=input_p_8, parent="root")
    fringe.append(start)
    temp_max_fringe_size = 1
    print("dfs running.....")
    while True:
        current_node = fringe[-1]
        del fringe[-1]

        n_n_popped += 1
        if current_node.data == output_p_8:
            backtrack_printing_data(current_node)
            break
        if current_node.data not in visited:
            n_n__expanded += 1
            for i in make_child(current_node):
                fringe.append(i)
                temp_max_fringe_size += 1
        temp_max_fringe_size -= 1
        if temp_max_fringe_size > max_fringe_size:
            max_fringe_size = temp_max_fringe_size
        visited.append(current_node.data)
        if dump_flag == True:
            trak_file()


def dls():
    global n_n_popped, n_n__expanded, max_fringe_size, fringe, d_l
    start = Node(data=input_p_8, parent="root")
    fringe.append(start)
    temp_max_fringe_size = 1
    flag = False
    print("dls running.....")
    while len(fringe) > 0:
        current_node = fringe[-1]
        del fringe[-1]

        n_n_popped += 1
        if current_node.data == output_p_8:
            backtrack_printing_data(current_node)
            flag = True
            break
        if current_node.data not in visited and current_node.depth < d_l:
            n_n__expanded += 1
            for i in make_child(current_node):
                fringe.append(i)
                temp_max_fringe_size += 1
        temp_max_fringe_size -= 1
        if temp_max_fringe_size > max_fringe_size:
            max_fringe_size = temp_max_fringe_size
        visited.append(current_node.data)
        if dump_flag == True:
            trak_file()
    if flag == False:
        print(f"depth of current node more than depth limit : {d_l}...STOP PROGRAM,USE DIFFERENT ALGO.")


def ids():
    global n_n_popped, n_n__expanded, max_fringe_size, fringe, n_n_generated, visited
    print("ids running.....")
    flag = False
    depth_limit=1
    while True:
        if flag == True:
            break
        n_n_popped = 0
        n_n__expanded = 0
        n_n_generated = 0
        max_fringe_size = 0
        fringe = []
        visited = []
        start = Node(data=input_p_8, parent="root", depth=0)
        fringe.append(start)
        temp_max_fringe_size = 1
        while len(fringe) > 0:
            current_node = fringe[-1]
            del fringe[-1]
            n_n_popped += 1
            if current_node.data == output_p_8:
                backtrack_printing_data(current_node)
                flag = True
                break
            if current_node.data not in visited and current_node.depth<depth_limit:
                n_n__expanded += 1
                for i in make_child(current_node):
                    fringe.append(i)
                    temp_max_fringe_size += 1
            temp_max_fringe_size -= 1
            if temp_max_fringe_size > max_fringe_size:
                max_fringe_size = temp_max_fringe_size
            visited.append(current_node.data)
        depth_limit+=1
        if dump_flag == True:
            trak_file()


if method == "a*":
    astar()
elif method == "bfs":
    bfs()
elif method == "ucs":
    ucs()
elif method == "dfs":
    dfs()
elif method == "dls":
    dls()
elif method == "ids":
    ids()
elif method == "greedy":
    greedy()
else:
    astar()