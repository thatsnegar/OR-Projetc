from environments.environment import Environment
from .abstract_solver import AbstractSolver
import gurobipy as gp
import numpy as np

class solver_343747_328556_324836(AbstractSolver):
    def __init__(self, env):
        super().__init__(env)
        self.name = "solver_343747_328556_324836"
    
    def solve(self):
       
        # (N_d × N_s) => number of candidates x no of supermarkets 
        svc = self.env.inst.service # service matrix => svc[j-1, i] = 1 if warehouse j can serve supermarket i 
      
        # ((N_d+1) × (N_d+1))
        dist = self.env.inst.distances

         # dictionary    
        weights = self.env.inst.weights 

         # Infer sizes
        N_total = dist.shape[0] # = 1 (depot index 0) + N_d 
        N_d = N_total - 1 #N_d is the no of candidate warehouse 
        
        # to gurantee that there is exactly one row in service per warehouse in destination
        if svc.shape[0] != N_d:
            raise ValueError(
                f"service.csv has {svc.shape[0]} rows, but distances.csv is {N_total}×{N_total} (implies {N_d} warehouses)."
            )
        
        #no of cols in service matrix = number od suermarkets 
        N_s = svc.shape[1]

        C_constr = weights['construction']
        C_miss = weights['missed_supermarket']
        C_travel = weights['travel']

        # creating Gurobi model
        model = gp.Model()
        # turn off Gurobi console
        model.Params.OutputFlag = 0


        # 1/ Decision variables 
        # if xj = 1 => we open warehouse j other wise we don't = 0
        x = model.addVars(range(1, N_d+1), vtype=gp.GRB.BINARY, name="x")
        
        # if zi = 1 => supermarket i is missed and we have penalty 
        # if zi = 0 => supermarket i is covered by at least one open warehouse
        z = model.addVars(range(N_s), vtype=gp.GRB.BINARY, name="z")

        # we have a single vehicle route that departs from the node 0
        # visit all and only those warehouses j for which xj = 1 and retrun to the node 0
        # if xij = 1 => route goes directly from i to j 
        y = {}
        for i in range(N_total):
            for j in range(N_total):
                if i == j:
                    continue
                y[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}_{j}")
        
        # subtour elimination variables 
        # uj is a real number helps to forbid disconnected subtours in the vehicle route 
        # do not attach any direct cost to uj 
        u = model.addVars(range(1, N_d+1), lb=0.0, ub=N_d, vtype=gp.GRB.CONTINUOUS, name="u")
        model.update()

        # 2/ Objective function 
        
        # construction cost 
        obj_constr = gp.quicksum(C_constr * x[j] for j in range(1, N_d+1))
        # missed supermarket
        obj_missed = gp.quicksum(C_miss * z[i] for i in range(N_s))
        # Travel cost   
        obj_travel = gp.quicksum(C_travel * dist[i,j] * y[i,j] for i in range(N_total) for j in range(N_total) if i != j)


        # minimize the sum of all three cost components 
        model.setObjective(obj_constr + obj_missed + obj_travel , gp.GRB.MINIMIZE)

        # 3/ Constraints 

        # cover or miss each supermarket i :
           # svc[j-1,i] = 1 => warehouse j is capable of serving supermarket i 
        
        # if j is open (xj = 1) exactly one route-arc enters and exactly one leaves  
        # if not no arc should enter or leave 

        for i in range(N_s):
            model.addConstr(gp.quicksum(svc[j-1, i] * x[j] for j in range(1, N_d+1)) + z[i] >= 1, name=f"cover_supermarket_{i}")

     

        # the 4 following constraint tell that how the single vehicle can leave
        # and returen to the depot => there us one vehicele so it can leave the depot once and re-enter at most once 
        # but if at least one warehouse is open => must leave and reenter exactly once
        for j in range(1, N_d+1):
            model.addConstr(gp.quicksum(y[i,j] for i in range(N_total) if i != j) == x[j], name = f"in_degree_{j}")
            model.addConstr(gp.quicksum(y[j,k] for k in range(N_total) if k != j) == x[j], name=f"out_degree_{j}")

        # Depot constraints:
        model.addConstr(
            gp.quicksum(y[0, j] for j in range(1, N_d+1)) <= 1,
            name="depot_out_max"
        )
        model.addConstr(
            N_d * gp.quicksum(y[0, j] for j in range(1, N_d+1))
            >= gp.quicksum(x[j] for j in range(1, N_d+1)),
            name="depot_out_min"
        )
        model.addConstr(
            gp.quicksum(y[i, 0] for i in range(1, N_d+1)) <= 1,
            name="depot_in_max"
        )
        model.addConstr(
            N_d * gp.quicksum(y[i, 0] for i in range(1, N_d+1))
            >= gp.quicksum(x[j] for j in range(1, N_d+1)),
            name="depot_in_min"
        )

        #subtour elimination:
        # if xj = 0 => ware house is close and uj = 0
        # if xj = 1 => then 1≤uj≤Nd

        for j in range(1, N_d+1):
            model.addConstr(u[j] >= x[j],       name=f"u_lb_{j}")
            model.addConstr(u[j] <= N_d * x[j], name=f"u_ub_{j}")

        for i in range(1, N_d+1):
            for j in range(1, N_d+1):
                if i == j:
                    continue
                model.addConstr(
                    u[i] - u[j] + N_d * y[i, j] <= N_d - x[j],
                    name=f"mtz_{i}_{j}"
                )
        
        # Solve
        model.optimize()

        # Extract X and Y
        # optimized value of each binary var xj and convert it to int (0,1)
        # build an array of length Nd
        X = np.array([int(x[j].X) for j in range(1, N_d+1)], dtype=int)

        # for each arc variable y[i,j] => if yi,j = 1 record it and otherwise 0  
        Y = np.zeros((N_total, N_total), dtype=int)
        for (i, j), var in y.items():
            Y[i, j] = int(var.X)

        return X, Y