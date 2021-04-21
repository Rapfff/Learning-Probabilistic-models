from examples_models import scheduler_uniform
from gridMDP import Grid

m = Grid()
s = scheduler_uniform(m.actions())


print(m.run(150,s))