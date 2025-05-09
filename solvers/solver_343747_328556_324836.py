

import gurobipy as gp
from gurobipy import GRB
from .abstract_solver import AbstractSolver
import numpy as np
from environments.environment import Environment


class solver_343747_328556_324836(AbstractSolver):
    def __init__(self, env: Environment):
        super().__init__(env)
        self.name = "solver_343747_328556_324836"
        self.env = env
        self.model = gp.Model(self.name)

        
    def solve(self):
        
        weight = self.env.inst.weights
        service = self.env.inst.service
        distances = self.env.inst.distances

        number_of_warehouse,number_of_supermarkets = service.shape
        locations = list(range(number_of_warehouse +1)) 

        # parameters
        cost = weight['construction']
        daily_penalty = weight['missed supermarket']
        travel_cost = weight['travel'] 

        # Decision variables
        y = self.model.addVars(number_of_warehouse, vtype=GRB.BINARY, name="build")  # y[i] = 1 if warehouse i is built
        s = self.model.addVars(number_of_supermarkets, vtype=GRB.BINARY, name="served")  # s[j] = 1 if supermarket j is served
        x = self.model.addVars(locations,locations, vtype=GRB.BINARY, name="route")  # x[i,j] = 1 if route from i to j

        # Objective function
        cost_of_build = gp.quicksum(cost * y[i] for i in range(number_of_warehouse))
        penalty = gp.quicksum((1 - s[j] for j in range(number_of_supermarkets))) * daily_penalty
        vehicel_route = gp.quicksum(travel_cost * distances[i][j] * x[i,j] for i in locations for j in locations)

        self.model.setObjective(cost_of_build + penalty + vehicel_route, GRB.MINIMIZE)

        # Constraints

        # here the constraint define that a supermarket can only be served if at least one warehouses 
        # that can serve it is open 
        # servuve[i,j] = 1 if warehouse i can serve supermarket j
        # x[i] = 1 if warehouse i is open
        # so basically the sum can count how many open warehouses are capable of servivng supermarket j
        # if none are open => s[j] = 0 => unserved => penalty
        for j in range(number_of_warehouse):
            self.model.addConstr(s[j] <= gp.quicksum(service[i, j] * x[i] for i in range(number_of_warehouse)))

        # y[i,j] = 1 if the vehicke traveks from location i to location j
        #  i => goes form 1 to number of warehouses because index 0 is the company
        #  x[i-1] => corresponds to warehouse i-1
        # if warehouse i is built => x[i-1] = 1 => vehicle must leave it once 
        # if not the vehicle can't leave it at all 

        # Meaning : each buotk warehouse must be entered and exited exactly once in the vehicle route 
        for i in range(1, number_of_warehouse + 1):
            self.model.addConstr(gp.quicksum(x[i, j] for j in locations if j != i) ==  x[i-1])
            self.model.addConstr(gp.quicksum(x[j, i] for j in locations if j != i) ==  x[i-1])
        
        self.model.addConstr(gp.quicksum(x[0, j] for j in locations if j != 0) == 1) #leave company => vehicle must leave the company exactly once to start its route 
        self.model.addConstr(gp.quicksum(x[j, 0] for j in locations if j != 0) == 1) #return to company => vehicle must return to the company exactly once to finish its route




