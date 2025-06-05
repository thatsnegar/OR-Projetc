# solvers/solver_343747_328556_324836.py

from .abstract_solver import AbstractSolver
import gurobipy as gp
import numpy as np

class solver_343747_328556_324836(AbstractSolver):
    def __init__(self, env):
        super().__init__(env)
        self.name = "solver_343747_328556_324836"

    def solve(self):
        svc     = self.env.inst.service      # (N_d × N_s)
        dist    = self.env.inst.distances    # ((N_d+1) × (N_d+1))
        weights = self.env.inst.weights      # dict with keys "construction","missed_supermarket","travel"

        # Debug
        print(">>> DEBUG service.csv shape =", svc.shape)
        print(">>> DEBUG distances.csv shape =", dist.shape)
        
        # Infer sizes:
        N_total = dist.shape[0]    # = 1 (depot) + N_d
        N_d     = N_total - 1
        if svc.shape[0] != N_d:
            raise ValueError(
                f"service.csv has {svc.shape[0]} rows, but distances.csv is {N_total}×{N_total} (implies {N_d} warehouses)."
            )
        N_s = svc.shape[1]

        C_constr = weights['construction']
        C_miss   = weights['missed_supermarket']
        C_trav   = weights['travel']

        model = gp.Model()
        model.Params.OutputFlag = 0

        # 1) Decision variables:
        x = model.addVars(range(1, N_d+1), vtype=gp.GRB.BINARY, name="x")
        z = model.addVars(range(N_s),         vtype=gp.GRB.BINARY, name="z")

        y = {}
        for i in range(N_total):
            for j in range(N_total):
                if i == j:
                    continue
                y[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}_{j}")

        u = model.addVars(range(1, N_d+1), lb=0.0, ub=N_d, vtype=gp.GRB.CONTINUOUS, name="u")
        model.update()

        # 2) Objective:
        obj_constr = gp.quicksum(C_constr * x[j] for j in range(1, N_d+1))
        obj_missed = gp.quicksum(C_miss   * z[i] for i in range(N_s))
        obj_travel = gp.quicksum(
            C_trav * dist[i, j] * y[i, j]
            for i in range(N_total) for j in range(N_total) if i != j
        )
        model.setObjective(obj_constr + obj_missed + obj_travel, gp.GRB.MINIMIZE)

        # 3) Constraints:

        # 3.1 Cover or miss each supermarket i:
        for i in range(N_s):
            model.addConstr(
                gp.quicksum(svc[j-1, i] * x[j] for j in range(1, N_d+1)) + z[i] >= 1,
                name=f"cover_supermarket_{i}"
            )

        # 3.2 If j is open, exactly one arc enters and one leaves:
        for j in range(1, N_d+1):
            model.addConstr(
                gp.quicksum(y[i, j] for i in range(N_total) if i != j) == x[j],
                name=f"in_degree_{j}"
            )
            model.addConstr(
                gp.quicksum(y[j, k] for k in range(N_total) if k != j) == x[j],
                name=f"out_degree_{j}"
            )

        # 3.3 Depot constraints:
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

        # 3.4 MTZ subtour elimination:
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

        # 4) Solve
        model.optimize()

        # 5) Extract X and Y
        X = np.array([int(x[j].X) for j in range(1, N_d+1)], dtype=int)
        Y = np.zeros((N_total, N_total), dtype=int)
        for (i, j), var in y.items():
            Y[i, j] = int(var.X)

        return X, Y
