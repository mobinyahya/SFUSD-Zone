from gurobipy import GRB
from gurobipy import *
import gurobipy as gp





#### ** make sure total school capacities = total student capacities
# sch_cap = [0.5, 1.5]
# prefs = [[1,0], [0,1]]
# prefs = [[0,1], [0,1]]

sch_cap = [0.5, 1, 0.5]
# prefs = [[0, 1, 2], [0, 1, 2]]
prefs = [[0, 1, 2], [1, 0, 2]]


##### ** make sure student capacities are equal across groups
stud_cap = [1] * len(prefs)

stud_len = len(stud_cap)
sch_len = len(sch_cap)

# Create a new model
model = Model("matrix1")

# Create variables
d = model.addVars(stud_len, sch_len, lb=0, vtype=GRB.CONTINUOUS, name="D")
t = model.addVars(stud_len, sch_len, lb=0, ub = 1, vtype=GRB.CONTINUOUS, name="T")
p = model.addVars(sch_len, lb=0, ub = 1, vtype=GRB.CONTINUOUS, name="P")



r1 = model.addVar(vtype=GRB.CONTINUOUS, name="r1")
r2 = model.addVar(vtype=GRB.CONTINUOUS, name="r2")


r1 = gp.quicksum([
        gp.quicksum([
            (d[stud, sch] - p[sch] + t[stud, sch]) for stud in range(stud_len)
        ])
        for sch in range(sch_len)
    ])
r2 = gp.quicksum([
        gp.quicksum([
            gp.quicksum([
                (t[stud, sch] - p[prefs[stud][i]]) for i in range(prefs[stud].index(sch))
            ])
            for stud in range(stud_len)
        ])
        for sch in range(sch_len)
    ])

model.setObjective(r1+r2, GRB.MINIMIZE)


for stud in range(stud_len):
    for sch in range(sch_len):
        model.addConstr(d[stud, sch] >= p[sch] - t[stud, sch])
        for i in range(prefs[stud].index(sch)):
            s = prefs[stud][i]
            print("stud: " + str(stud) + " sch: " + str(sch) + " prefs index: " + str(prefs[stud].index(sch)) + " more prefered schools: " + str(s))
            model.addConstr(t[stud, sch] >= p[s])


for stud in range(stud_len):
    model.addConstr(sum(d[stud, j] for j in range(0, sch_len)) == stud_cap[stud], "student allocation")

for sch in range(sch_len):
    model.addConstr(sum(d[i, sch] for i in range(0, stud_len)) == sch_cap[sch], "school capacity")




model.optimize()
for v in model.getVars():
    print(v.varName, v.x)
print('Obj:', model.objVal)