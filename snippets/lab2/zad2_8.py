result = integrate.solve_ivp(model, [0,15], [0], t_eval=t_eval, args=(A, B, u,))