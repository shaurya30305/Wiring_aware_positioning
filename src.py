import random
import math
from collections import defaultdict

class Gate:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0
        self.pins = []

    def add_pins(self, pin_list):
        self.pins.append(pin_list)

class Wire:
    def __init__(self, gate1_name, pin1_index, gate2_name, pin2_index):
        self.gate1_name = gate1_name
        self.pin1_index = pin1_index
        self.gate2_name = gate2_name
        self.pin2_index = pin2_index

class DSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

def parser(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    l1 = []
    l2 = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == 'pins': 
            gate_number = int(parts[1][1:]) - 1
            for i in range(2, len(parts), 2):
                l1[gate_number].append((int(parts[i]), int(parts[i+1])))
        elif parts[0] == 'wire':
            gate1, pin1 = parts[1].split('.')
            gate2, pin2 = parts[2].split('.')
            gate1 = int(gate1[1:])
            gate2 = int(gate2[1:])
            pin1 = int(pin1[1:])
            pin2 = int(pin2[1:])
            l2.append((gate1 - 1, pin1 - 1, gate2 - 1, pin2 - 1))
        elif parts[0].startswith('g'):
            gate_number = int(parts[0][1:]) - 1
            width = int(parts[1])
            height = int(parts[2])
            l1.append([(width, height)])

    return l1, l2

def parser2(file_path):
    gates = []
    wires = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'pins': 
                gate_number = int(parts[1][1:]) - 1
                pins = []
                for i in range(2, len(parts), 2):
                    pins.append((int(parts[i]), int(parts[i+1])))
                gates[gate_number].add_pins(pins)
            elif parts[0] == 'wire':
                gate1, pin1 = parts[1].split('.')
                gate2, pin2 = parts[2].split('.')
                wires.append((int(gate1[1:]) - 1, int(pin1[1:]) - 1, int(gate2[1:]) - 1, int(pin2[1:]) - 1))
            elif parts[0].startswith('g'):
                gate_number = int(parts[0][1:]) - 1
                width = int(parts[1])
                height = int(parts[2])
                gates.append(Gate(f'g{gate_number+1}', width, height))

    return gates, wires


def is_overlap(gatecoordi, config, new_x, new_y, i):
    new_width = gatecoordi[i][0][0]
    new_height = gatecoordi[i][0][1]
    
    for j in range(i):  
        x, y = config[j]
        width = gatecoordi[j][0][0]
        height = gatecoordi[j][0][1]
        
        if not (new_x + new_width <= x or 
                new_x >= x + width or      
                new_y + new_height <= y or 
                new_y >= y + height):    
            return True 
    return False 
def generate_stacked_config_fixed_per_row(gatecoordi):
    n = len(gatecoordi)  
    gates_per_row = int(math.sqrt(n))  
    config = []

    current_x = 0
    current_y = 0
    max_height_in_row = 0
    gates_in_current_row = 0

    for i, gate in enumerate(gatecoordi):
        gate_width = gate[0][0]
        gate_height = gate[0][1]

       
        config.append([current_x, current_y])
        gates_in_current_row += 1
        current_x += gate_width + 1 
        max_height_in_row = max(max_height_in_row, gate_height)  

        if gates_in_current_row >= gates_per_row:
            current_y += max_height_in_row   
            current_x = 0  
            max_height_in_row = 0  
            gates_in_current_row = 0  

    return config
def generate_initial_config(gatecoordi):
    num_gates = len(gatecoordi)
    
    total_width = sum(g[0][0] for g in gatecoordi)  
    total_height = sum(g[0][1] for g in gatecoordi)  
    
    config = [[0, 0]]  
    
    for i in range(1, num_gates):
        placed = False
        
        while not placed:
            random_x = random.randint(0, total_width)
            random_y = random.randint(0, total_height)
            
            if not is_overlap(gatecoordi, config, random_x, random_y, i):
                config.append([random_x, random_y])
                placed = True
    
    return config
def check_overlap(new_x, new_y,width, height,config, coordi):
  
    for gate in range(len(config)):

        existing_x, existing_y, existing_width, existing_height = config[gate][0], config[gate][1], coordi[gate][0][0], coordi[gate][0][1]
        
        if (new_x < existing_x + existing_width and
            new_x + width > existing_x and
            new_y < existing_y + existing_height and
            new_y + height > existing_y):
            return True  
    return False
def swapoverlap(g1,g2,config,coordi):
    new_x, new_y = config[g2][0], config[g2][1]
    for gate in range(len(config)):
        if gate != g2:
            existing_x, existing_y, existing_width, existing_height = config[gate][0], config[gate][1], coordi[gate][0][0], coordi[gate][0][1]
            
            if (new_x < existing_x + existing_width and
                new_x + coordi[g1][0][0] > existing_x and
                new_y < existing_y + existing_height and
                new_y + coordi[g1][0][1] > existing_y):
                return True  
    new_x, new_y = config[g1][0], config[g1][1]
    for gate in range(len(config)):
        if gate != g1:
            existing_x, existing_y, existing_width, existing_height = config[gate][0], config[gate][1], coordi[gate][0][0], coordi[gate][0][1]
            
            if (new_x < existing_x + existing_width and
                new_x + coordi[g2][0][0] > existing_x and
                new_y < existing_y + existing_height and
                new_y + coordi[g2][0][1] > existing_y):
                return True  
    return False
def next_iteration(config, gatecoordi, pins, swap_prob=1):
    new_config = [list(coord) for coord in config]

    try:
        for (g1, p1, g2, p2) in pins:
            gate1_corner = config[g1]  
            pin1_offset = gatecoordi[g1][p1 + 1]  
            pin1_coord = (gate1_corner[0] + pin1_offset[0], gate1_corner[1] + pin1_offset[1])

            gate2_corner = config[g2]  
            pin2_offset = gatecoordi[g2][p2 + 1]  
            pin2_coord = (gate2_corner[0] + pin2_offset[0], gate2_corner[1] + pin2_offset[1])
            hor_dist = pin1_coord[0] - pin2_coord[0]
            ver_dist = pin1_coord[1] - pin2_coord[1]

            if abs(hor_dist) >= 1:
                if hor_dist > 0: 
                    new_x1 = max(0, new_config[g1][0] - 1)  
                    if check_overlap(new_x1, new_config[g1][1], gatecoordi[g1][0][0], gatecoordi[g1][0][1], new_config, gatecoordi) is False:
                        new_config[g1][0] = new_x1
                else:  
                    new_x2 = max(0, new_config[g2][0] - 1) 
                    if check_overlap(new_x2, new_config[g2][1], gatecoordi[g2][0][0], gatecoordi[g2][0][1], new_config, gatecoordi) is False:
                        new_config[g2][0] = new_x2

            if abs(ver_dist) >= 1:
                new_y1 = new_config[g1][1] - 1
                if ver_dist > 0:  
                    new_y1 = max(0, new_y1) 
                    if check_overlap(new_config[g1][0], new_y1, gatecoordi[g1][0][0], gatecoordi[g1][0][1], new_config, gatecoordi) is False:
                        new_config[g1][1] = new_y1
                else:  
                    new_y2 = new_config[g2][1] - 1
                    new_y2 = max(0, new_y2)
                    if check_overlap(new_config[g2][0], new_y2, gatecoordi[g2][0][0], gatecoordi[g2][0][1], new_config, gatecoordi) is False:
                        new_config[g2][1] = new_y2
    except IndexError as e:
        print(f"IndexError: {e} - gate1: {g1}, pin1: {p1}, gate2: {g2}, pin2: {p2}")
        print(f"coordi: {gatecoordi}, pins: {pins}")

    if random.random() < swap_prob:
        g1, g2 = random.sample(range(len(config)), 2)
        while swapoverlap(g1, g2, new_config, gatecoordi) is True:
            g1, g2 = random.sample(range(len(config)), 2)
        new_config[g1], new_config[g2] = new_config[g2], new_config[g1]  

    return new_config
def calc_wirelength(gate_dict, wire_list):
    dsu = DSU()
    for gate in gate_dict.values():
        for i in range(len(gate.pins)):
            pin_name = 'p' + str(i + 1)  # Generate pin name
            dsu.add((gate.name, pin_name))

    # Add all pin connections to DSU
    for wire in wire_list:
        pin1 = (wire.gate1_name, wire.pin1_index)
        pin2 = (wire.gate2_name, wire.pin2_index)
        dsu.union(pin1, pin2)  # Connect the pins

    # Map root components to pin coordinates
    components = {}

    for gate in gate_dict.values():
        for pin_index, pin_coords in enumerate(gate.pins):
            pin_name = 'p' + str(pin_index + 1)
            coord = (gate.x + pin_coords[0], gate.y + pin_coords[1])
            root = dsu.find((gate.name, pin_name))
            if root not in components:
                components[root] = []
            components[root].append((pin_name, coord))

    total_length = 0

    # Calculate wire length for each component
    for coords in components.values():
        min_x = min(coord[1][0] for coord in coords)
        max_x = max(coord[1][0] for coord in coords)
        min_y = min(coord[1][1] for coord in coords)
        max_y = max(coord[1][1] for coord in coords)

        # Calculate the semi-perimeter of the bounding box
        length = (max_x - min_x) + (max_y - min_y)
        total_length += length

    return total_length
def generate_simple_row_config(gatecoordi):
    config = []
    current_x = 0

    for gate in gatecoordi:
        gate_width = gate[0][0]
        config.append([current_x, 0]) 
        current_x += gate_width + 1 

    return config


def generate_initial_config2(gates2):
    num_gates = len(gates2)
    gates_per_row = int(math.sqrt(num_gates))
    config = []
    current_x, current_y, max_height = 0, 0, 0

    for i, gate in enumerate(gates2):
        config.append([current_x, current_y])
        current_x += gate.width + 1
        max_height = max(max_height, gate.height)

        if (i + 1) % gates_per_row == 0:
            current_y += max_height + 1
            current_x, max_height = 0, 0

    return config
def is_overlap2(gates2, config, new_x, new_y, i):
    new_width, new_height = gates2[i].width, gates2[i].height
    
    for j in range(len(config)):
        if j == i:
            continue
        x, y = config[j]
        width, height = gates2[j].width, gates2[j].height
        
        if not (new_x + new_width <= x or new_x >= x + width or
                new_y + new_height <= y or new_y >= y + height):
            return True
    return False
def calc_wirelength2(gates2, wires2, dsu):
    components = defaultdict(lambda: [[float('inf'), float('inf')], [float('-inf'), float('-inf')]])

    for wire in wires2:
        pin1 = (f'g{wire[0]+1}', f'p{wire[1]+1}')
        pin2 = (f'g{wire[2]+1}', f'p{wire[3]+1}')
        root = dsu.find(pin1)
        dsu.union(pin1, pin2)

        gate1, gate2 = gates2[wire[0]], gates2[wire[2]]
        # print(wire[0],wire[1],wire[2],wire[3])
        # print(gate1.pins)
        coord1 = (gate1.x + gate1.pins[0][wire[1]][0], gate1.y + gate1.pins[0][wire[1]][1])
        coord2 = (gate2.x + gate2.pins[0][wire[3]][0], gate2.y + gate2.pins[0][wire[3]][1])

        components[root][0][0] = min(components[root][0][0], coord1[0], coord2[0])
        components[root][0][1] = min(components[root][0][1], coord1[1], coord2[1])
        components[root][1][0] = max(components[root][1][0], coord1[0], coord2[0])
        components[root][1][1] = max(components[root][1][1], coord1[1], coord2[1])

    return sum((max_x - min_x) + (max_y - min_y) for [min_x, min_y], [max_x, max_y] in components.values())
def next_iteration2(config, gates2, wires2, sample_size=50):
    new_config = [list(coord) for coord in config]
    
    for wire in random.sample(wires2, min(len(wires2), sample_size)):
        g1, p1, g2, p2 = wire
        dx = config[g2][0] - config[g1][0]
        dy = config[g2][1] - config[g1][1]
        
        if dx != 0:
            new_x1 = max(0, new_config[g1][0] + (1 if dx > 0 else -1))
            new_x2 = max(0, new_config[g2][0] - (1 if dx > 0 else -1))
            if not is_overlap2(gates2, new_config, new_x1, new_config[g1][1], g1):
                new_config[g1][0] = new_x1
            if not is_overlap2(gates2, new_config, new_x2, new_config[g2][1], g2):
                new_config[g2][0] = new_x2
        
        if dy != 0:
            new_y1 = max(0, new_config[g1][1] + (1 if dy > 0 else -1))
            new_y2 = max(0, new_config[g2][1] - (1 if dy > 0 else -1))
            if not is_overlap2(gates2, new_config, new_config[g1][0], new_y1, g1):
                new_config[g1][1] = new_y1
            if not is_overlap2(gates2, new_config, new_config[g2][0], new_y2, g2):
                new_config[g2][1] = new_y2

    return new_config


def simulated_annealing(config, gates, wires, initial_temp=1000, cooling_rate=0.99, iterations=5):
    current_config = config[:]
    gate_dict = {f'g{i+1}': Gate(f'g{i+1}', gate[0][0], gate[0][1]) for i, gate in enumerate(gates)}
    
    for i, gate in enumerate(gates):
        gate_dict[f'g{i+1}'].x, gate_dict[f'g{i+1}'].y = current_config[i]
        gate_dict[f'g{i+1}'].pins = gate[1:]

    wire_list = [Wire(f'g{w[0]+1}', f'p{w[1]+1}', f'g{w[2]+1}', f'p{w[3]+1}') for w in wires]
    
    current_wire_length = calc_wirelength(gate_dict, wire_list)
    temperature = initial_temp


    while temperature > 0.001:
        for _ in range(iterations):
            new_config = next_iteration(current_config, gates, wires)
            
            for i, (x, y) in enumerate(new_config):
                gate_dict[f'g{i+1}'].x, gate_dict[f'g{i+1}'].y = x, y
            
            new_wire_length = calc_wirelength(gate_dict, wire_list)

            if new_wire_length < current_wire_length or random.random() < math.exp((current_wire_length - new_wire_length) / temperature):
                current_config = new_config
                current_wire_length = new_wire_length

        temperature *= cooling_rate

    return current_config, current_wire_length
def simulated_annealing2(gates2, wires2, initial_temp=100, cooling_rate=0.95, iterations=50):
    current_config = generate_initial_config2(gates2)
    dsu = DSU()
    
    for i, (x, y) in enumerate(current_config):
        gates2[i].x, gates2[i].y = x, y

    current_wire_length = calc_wirelength2(gates2, wires2, dsu)
    best_config = current_config
    best_wire_length = current_wire_length
    temperature = initial_temp

    while temperature > 1:
        for _ in range(iterations):
            new_config = next_iteration2(current_config, gates2, wires2)
            
            for i, (x, y) in enumerate(new_config):
                gates2[i].x, gates2[i].y = x, y
            
            new_wire_length = calc_wirelength2(gates2, wires2, dsu)

            if new_wire_length < current_wire_length or random.random() < math.exp((current_wire_length - new_wire_length) / temperature):
                current_config = new_config
                current_wire_length = new_wire_length
                
                if current_wire_length < best_wire_length:
                    best_config = current_config
                    best_wire_length = current_wire_length

        temperature *= cooling_rate

    return best_config, best_wire_length

def main():
    random.seed(20)

    gates, wires = parser('input.txt')

    if len(gates) < 4:
        config = generate_simple_row_config(gates)
        gate_dict = {f'g{i+1}': Gate(f'g{i+1}', gate[0][0], gate[0][1]) for i, gate in enumerate(gates)}
    
        for i, gate in enumerate(gates):
            gate_dict[f'g{i+1}'].x, gate_dict[f'g{i+1}'].y = config[i]
            gate_dict[f'g{i+1}'].pins = gate[1:]

        wire_list = [Wire(f'g{w[0]+1}', f'p{w[1]+1}', f'g{w[2]+1}', f'p{w[3]+1}') for w in wires]
        length = calc_wirelength(gate_dict, wire_list)
    else:
        if len(gates)<=100:
            current_config = generate_stacked_config_fixed_per_row(gates)
            config,length = simulated_annealing(current_config, gates, wires)
            gate_dict = {f'g{i+1}': Gate(f'g{i+1}', gates[i][0][0], gates[i][0][1]) for i in range(len(gates))}
            for i, (x, y) in enumerate(config):
                gate_dict[f'g{i+1}'].x, gate_dict[f'g{i+1}'].y = x, y
                gate_dict[f'g{i+1}'].pins = gates[i][1:]

            wire_list = [Wire(f'g{w[0]+1}', f'p{w[1]+1}', f'g{w[2]+1}', f'p{w[3]+1}') for w in wires]
            length = calc_wirelength(gate_dict, wire_list)
        else:
            gates2,wires2 = parser2('input.txt')
            config, length = simulated_annealing2(gates2,wires2)
    

    b0 = max(config[i][0] + gates[i][0][0] for i in range(len(config)))
    b1 = max(config[i][1] + gates[i][0][1] for i in range(len(config)))

    with open('output.txt', 'w') as f:
        f.write(f"bounding_box {b0} {b1}\n")
        for i, (x, y) in enumerate(config, 1):
            f.write(f'g{i} {x} {y}\n')
        f.write(f'wire_length {length}\n')


if __name__ == "__main__":
    main()