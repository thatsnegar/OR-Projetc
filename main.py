from instances import *
from environments import *
from solvers import *
from solutions import *
import json 

instance_name = 'solver_343747_328556_324836'

inst = Instance(instance_name)
env = Environment(inst)
solver = solver_343747_328556_324836(env)

X, Y = solver.solve()

sol = Solution(X, Y)
sol.write(instance_name)