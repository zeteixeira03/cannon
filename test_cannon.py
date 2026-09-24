from cannon import *

s0 = initial_state(init_params())
(p1, v1), rec = step(s0, None)
print(p1, v1)
