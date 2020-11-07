"""Simple implementation of FIFO, LIFO, and PriorityQueue."""


import numpy as np
import matplotlib.pyplot as plt
from misc import Timer, LatLongDistance
from queues import FIFO, LIFO, PriorityQueue
from osm import loadOSMmap

def DepthFirst(num_nodes, mission, f_next, heuristic=None, num_controls=0):
    """Depth first planner."""
    t = Timer()
    t.tic()
    
    unvis_node = -1
    previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
    cost_to_come = np.zeros(num_nodes)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)

    startNode = mission['start']['id']
    goalNode = mission['goal']['id']

    q = LIFO()
    q.insert(startNode)
    foundPlan = False

    while not q.IsEmpty():
        x = q.pop()
        if x == goalNode:
            foundPlan = True
            break
        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            if previous[xi] == unvis_node:
                previous[xi] = x
                q.insert(xi)
                cost_to_come[xi] = cost_to_come[x] + di
                if num_controls > 0:
                    control_to_come[xi] = ui

    # Recreate the plan by traversing previous from goal node
    if not foundPlan:
        return []
    else:
        plan = [goalNode]
        length = cost_to_come[goalNode]
        control = []
        while plan[0] != startNode:
            if num_controls > 0:
                control.insert(0, control_to_come[plan[0]])
            plan.insert(0, previous[plan[0]])

        return {'plan': plan,
                'length': length,
                'num_visited_nodes': np.sum(previous != unvis_node),
                'name': 'DepthFirst',
                'time': t.toc(),
                'control': control,
                'visited_nodes': previous[previous != unvis_node]}
    
def BreadthFirst(num_nodes, mission, f_next, heuristic=None, num_controls=0):
    t = Timer()
    t.tic()
    
    unvis_node = -1
    previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
    cost_to_come = np.zeros(num_nodes)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)

    startNode = mission['start']['id']
    goalNode = mission['goal']['id']

    #First in, First Out from Queues
    q = FIFO()
    q.insert(startNode)
    foundPlan = False

    while not q.IsEmpty():
        x = q.pop()
        if x == goalNode:
            foundPlan = True
            break
        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            if previous[xi] == unvis_node:
                previous[xi] = x
                q.insert(xi)
                cost_to_come[xi] = cost_to_come[x] + di
                if num_controls > 0:
                    control_to_come[xi] = ui

    # Recreate the plan by traversing previous from goal node
    if not foundPlan:
        return []
    else:
        plan = [goalNode]
        length = cost_to_come[goalNode]
        control = []
        while plan[0] != startNode:
            if num_controls > 0:
                control.insert(0, control_to_come[plan[0]])
            plan.insert(0, previous[plan[0]])

        return {'plan': plan,
                'length': length,
                'num_visited_nodes': np.sum(previous != unvis_node),
                'name': 'Breadth First',
                'time': t.toc(),
                'control': control,
                'visited_nodes': previous[previous != unvis_node]}

def Dijkstra(num_nodes, mission, f_next, heuristic=None, num_controls=0):
    """Djikstra planner."""
    t = Timer()
    t.tic()
    
    unvis_node = -1
    previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
    cost_to_come = np.zeros(num_nodes)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)

    startNode = mission['start']['id']
    goalNode = mission['goal']['id']

    #Priority queue based on the lower cost-to-go
    q = PriorityQueue()
    q.insert(0,startNode)
    foundPlan = False

    while not q.IsEmpty():
        x_ctc = q.pop()
        x = x_ctc[1]
        if x == goalNode:
            foundPlan = True
            break
        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            if previous[xi] == unvis_node or cost_to_come[xi] > cost_to_come[x] + di:
                previous[xi] = x
                cost_to_come[xi] = cost_to_come[x] + di
                q.insert(cost_to_come[xi],xi)
                if num_controls > 0:
                    control_to_come[xi] = ui

    # Recreate the plan by traversing previous from goal node
    if not foundPlan:
        return []
    else:
        plan = [goalNode]
        length = cost_to_come[goalNode]
        control = []
        while plan[0] != startNode:
            if num_controls > 0:
                control.insert(0, control_to_come[plan[0]])
            plan.insert(0, previous[plan[0]])

        return {'plan': plan,
                'length': length,
                'num_visited_nodes': np.sum(previous != unvis_node),
                'name': 'Djikstra',
                'time': t.toc(),
                'control': control,
                'visited_nodes': previous[previous != unvis_node]}


def Astar(num_nodes, mission, f_next, h, num_controls=0):
    """Astar planner."""
    t = Timer()
    t.tic()
    unvis_node = -1
    previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
    cost_to_come = np.zeros(num_nodes)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)
    startNode = mission['start']['id']
    goalNode = mission['goal']['id']
    q = PriorityQueue()
    q.insert(0+ h(startNode),startNode)
    foundPlan = False

    while not q.IsEmpty():
        x_ctc = q.pop()
        x = x_ctc[1]
        if x == goalNode:
            foundPlan = True
            break

        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            if previous[xi] == unvis_node or cost_to_come[xi] > cost_to_come[x] + di:
                previous[xi] = x
                cost_to_come[xi] = cost_to_come[x] + di
                q.insert(cost_to_come[xi]+h(xi),xi)
                if num_controls > 0:
                    control_to_come[xi] = ui

    # Recreate the plan by traversing previous from goal node
    if not foundPlan:
        return []
    else:
        plan = [goalNode]
        length = cost_to_come[goalNode]
        control = []
        while plan[0] != startNode:
            if num_controls > 0:
                control.insert(0, control_to_come[plan[0]])
            plan.insert(0, previous[plan[0]])

        return {'plan': plan,
                'length': length,
                'num_visited_nodes': np.sum(previous != unvis_node),
                'name': 'Astar',
                'time': t.toc(),
                'control': control,
                'visited_nodes': previous[previous != unvis_node]}




def BestFirst(num_nodes, mission, f_next, h, num_controls=0):
    """BestFirst planner."""
    t = Timer()
    t.tic()
    unvis_node = -1
    previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
    cost_to_come = np.zeros(num_nodes)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)
    startNode = mission['start']['id']
    goalNode = mission['goal']['id']
    q = PriorityQueue()
    q.insert(h(startNode),startNode)
    foundPlan = False

    while not q.IsEmpty():
        x_ctc = q.pop()
        x = x_ctc[1]
        if x == goalNode:
            foundPlan = True
            break

        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            if previous[xi] == unvis_node:
                previous[xi] = x
                cost_to_come[xi] = cost_to_come[x] + di
                q.insert(h(xi),xi)
                if num_controls > 0:
                    control_to_come[xi] = ui

    # Recreate the plan by traversing previous from goal node
    if not foundPlan:
        return []
    else:
        plan = [goalNode]
        length = cost_to_come[goalNode]
        control = []
        while plan[0] != startNode:
            if num_controls > 0:
                control.insert(0, control_to_come[plan[0]])
            plan.insert(0, previous[plan[0]])

        return {'plan': plan,
                'length': length,
                'num_visited_nodes': np.sum(previous != unvis_node),
                'name': 'Astar',
                'time': t.toc(),
                'control': control,
                'visited_nodes': previous[previous != unvis_node]}


def cost_to_go(x, xg):
    p_x = osmMap.nodeposition[x]
    p_g = osmMap.nodeposition[xg]
    return 0.0;

#heuristic function
def h(x):
    goalNode = mission['goal']['id']
    p_x = osmMap.nodeposition[x]
    p_g = osmMap.nodeposition[goalNode]
    return LatLongDistance(p_x, p_g)