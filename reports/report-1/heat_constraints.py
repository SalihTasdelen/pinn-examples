def pde_residual(phi, stack):
    return stack[1][:,0] - 0.05 * stack[2][:,1,1]

def ic_residual(phi, stack):
    return stack[0] - tf.sin(3.0 * np.pi *  phi[:,1:])

def bc_residual(phi, stack):
    return stack[1][:,1] - 0

pde_con = Constraint(lhs(2, 100), pde_residual)
ic_con = Constraint(np.array([0, 1])*lhs(2, 100), ic_residual)
bc0_con = Constraint(np.array([1, 0])*lhs(2, 100), bc_residual)
bc1_con = Constraint(np.array([1, 0])*lhs(2, 100) + np.array([0,1]),
                      bc_residual)