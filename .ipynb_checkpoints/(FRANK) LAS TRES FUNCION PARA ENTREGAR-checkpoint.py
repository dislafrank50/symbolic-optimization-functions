import sympy as sp

'''=================================== FILTRO PARA EL SOLVE ================================='''

def check_sols(sols):
    """
    Elimina:
    - Soluciones infinitas/simbólicas (ej: {x: 1 - y})
    - Soluciones complejas (ej: {x: 5j})
    - Soluciones infinitas (ej: {x: oo})
    """
    #Poco profecional pero funcional
    valid_sols = []
    for sol in sols:
        es_valid = True
        for key in sol.keys(): # va a chequear cada variable en el diccionario
            val = sol[key]
            # Convertir el valor a un objeto sympy para asegurar
            val_sym = sp.sympify(val) 
            # solver puede dar numeros(5 un int) o symply objeto como sp.exp(2) 
            # 1. Comprobar si es infinito
            if val_sym.free_symbols != set():
                es_valid = False
                break
            
            # 2. Comprobar si es real y finito 
            val_expanded = sp.expand_complex(val_sym)
            if not (val_expanded.is_real and val_expanded.is_finite):
                es_valid = False
                break
        
        if es_valid:
            valid_sols.append(sol)
            
    return valid_sols


'''=================================== CONJUNTOS ABIERTOS ================================='''

def conjuntos_abiertos(fun, vars):
    
    '''Tiene como limitacion que solo funciona para puntos aislados, 
    ya que descarta los infinitos'''

    # Calcula el vector gradiente y los igualamos a 0
    grad = [sp.Eq(sp.diff(fun, v), 0) for v in vars]
    # Resolvemos el sistema de ecuaciones
    sols = sp.solve(grad, vars, dict=True)
    #Applicamos el filtro para quedarnos solo con las soluciones reales 
    sols = check_sols(sols)
    print(f'Los puntos criticos reales son: {sols}')
    # Calcula la matriz Hessiana
    HS = sp.hessian(fun, vars)
    # Evaluacion de la matriz Hessiana en los puntos criticos
    results = []
    for sol in sols:
        # Sustituimos los valores propiosen la Hessian
        HS_eval = HS.subs(sol)
        # Calculamos los menores principales
        HS_mat = sp.Matrix(HS_eval)
        # Determinamos el tipo de punto crítico según los signos de los valores propios
        if HS_mat.is_positive_definite:# En realidad es SEMI DEFINIDA POSITIVA
            tipo = 'minimo local'
        elif HS_mat.is_negative_definite:
            tipo = 'maximo local'
        else:
            tipo = 'punto de silla o indeterminado'

        results.append({'punto': sol, 'tipo': tipo})
    return results

#Ejemplo
'''
     y, z = sp.symbols('y,z')
     fun = (1 * y * z)**2 + (y * z**2 + 2)**2
     vars = [y, z]
     conjuntos_abiertos(fun, vars)'''
     
     
'''=================================== LAGRANGEN ================================='''

def Lagragen(fun, vars, res):
    r = len(res)
    # r es el número de restricciones
    delta = list(sp.symbols(f'delta0:{r}', real=True))
    # habra tantas deltas como restricciones
    L = fun + sum(delta[i] * res[i] for i in range(r))
    # x las deltas por las restricciones y las + a la función objetivo
    eqs = [sp.Eq(sp.diff(L, v), 0) for v in vars+delta]
    # hacemos las derivadas parciales de L respecto a las variables y a delta, luego las igualamos a 0
    sols = list(sp.solve(eqs, vars+delta, dict=True))
    # resolvemos el sistema de ecuaciones para las variables y los multiplicadores
    sols = check_sols(sols)
    # Applicamos el filtro para quedarnos solo con las soluciones reales 

    results = []
    for s in sols:
        
        #Definición y evaluación de las matrices
        HL = sp.Matrix(sp.hessian(L, vars))
        HLS = HL.subs(s)
        J = sp.Matrix([[sp.diff(res[i], v) for v in vars] for i in range(r)])
        JS = J.subs(s) # J no depende de delta
        
        #CÁLCULO de los vectores que anulan a JS
        
        # Obtenemos la matriz RREF y los índices de las columnas pivote
        R_matrix, pivot_indices = JS.rref()
        # el numero de columna de JS es el numero de variables que tendremos
        num_vars = JS.shape[1]
        
        # Identificamos las columnas que NO son pivotes (Variables Libres)
        free_indices = [i for i in range(num_vars) if i not in pivot_indices]
        
        Z_vectors = []
        
        # Construimos un vector base por cada variable libre
        for free_idx in free_indices:
            # Crea un vector columna de ceros de tamaño num_vars
            vec = sp.zeros(num_vars, 1)
            # La variable libre correspondiente toma el valor 1
            vec[free_idx] = 1 
        
            # Despejamos las variables pivote en función de esta libre
            # R_matrix[row_idx, free_idx] es el coeficiente de la variable libre
            # en la ecuación de la fila row_idx.
            for row_idx, piv_col_idx in enumerate(pivot_indices):
                # En R_matrix * vector = 0, se cumple: 
                # (variable pivote) + (coeff * variable libre) = 0
                # Por lo tanto: variable pivote = - (coeff * variable libre)
                coeff = R_matrix[row_idx, free_idx]
                vec[piv_col_idx] = -coeff
        
            Z_vectors.append(vec)
        
        #Ahora hacemos la restricción de la Hessiana
        
        if not Z_vectors:
            # Si no hay vectores en el espacio nulo (dim = 0), no hay espacio tangente
            #las restricciones te han confinado a un único punto (0 grados de libertad). 
            #No puedes moverte, por lo que "mínimo/máximo" carece de sentido en el sentido de una derivada.
            tipo = 'Restricción trivial (posiblemente no es un punto de mínimo/máximo)'
            HRestringida = sp.Matrix([[]])
        else:
            # Apilamos los vectores Z para formar la matriz Z_mat
            Z_mat = sp.Matrix.hstack(*Z_vectors)
            
            # Restringimos la Hessiana a ese espacio nulo (el espacio tangente)
            HRestringida = Z_mat.T * HLS * Z_mat
            
            # Determinamos el tipo de punto crítico
            # is_positive_definite/is_negative_definite comprueba los signos de los valores propios
            if HRestringida.is_positive_definite:
                tipo = 'mínimo local'
            elif HRestringida.is_negative_definite:
                tipo = 'máximo local'
            else:
                tipo = 'punto de silla o indeterminado'
        
        results.append({
            'punto': s, 
            'tipo': tipo,
            'Hessiana_Restringida': HRestringida
        })

    return results
# Ejemplo
'''
    x, y, z = sp.symbols('x y z', real=True)
    vars = [x, y, z]
    fun = x**2 + y**2 + z**2
    res = [x**2 + y**2 - 1, x + y + z - 1]
    Lagragen(fun, vars, res)'''


'''=================================== KUNH TUCKER ================================='''

def check_sols_compl(sols):
    '''Filtro solo para resultados complejos, ya que no es necesario filtrar las soluciones 
    infinitas porque las comprobaciones 2ºKKT y 3ºKKT (que utilizan `sp.reduce_inequalities`)
    toman el conjunto "infinito" de soluciones y definen la región en la que son válidas.'''
    return [sol for sol in sols if all(sp.expand_complex(sol[key]).is_real 
                                       for key in sol.keys())]


def solve_kt(fun, res, vars, t_param=None, max=True):

    r = len(res)
    # r es el número de restricciones
    delta = list(sp.symbols(f'delta0:{r}', real=True))
    # delta son los multiplicadores de Lagrange
    sign = -1 if max else 1
    L = fun + sign * sum(delta[i] * res[i] for i in range(r))
# Condiciones de 1º kKT
    kt_1st = [sp.Eq(sp.diff(L, v), 0) for v in vars]
# TODAS las condiciones de 4º KKT (Holgura Complementaria)
    kt_4th = [sp.Eq(delta[j] * res[j], 0) for j in range(r)]
    # Combinar todas las ecuaciones en una sola lista
    kt_eqs = kt_1st + kt_4th
    # Resuelve el sistema NO LINEAL que combina 1ºKT y 4ºKT

    sols = sp.solve(kt_eqs, vars + delta, dict=True)
    sols = check_sols_compl(sols)

    if not sols:
        return ["Sin solución (sistema no lineal incompatible)"]
        
    result = []
    
    # Este bucle itera sobre las soluciones encontradas
    for s in sols:
        # Rellena los valores que 'solve' pudo encontrar
        sol_completa = {k: s.get(k, k) for k in (vars + delta)}
        var_res = ", ".join([f"{v}={sol_completa.get(v, v)}" for v in vars])

        # Ahora, deducimos el caso mirando los deltas de la solución
        
        casos_part = []
        acti_delta = []   # Para 2º KT (delta_i >= 0)
        inacti_res = []   # Para 3º KT (res_i <= 0)
        
        for j in range(r):
            delta_val = sol_completa.get(delta[j], 0)
            
            # --- Comprobación Simbólica ---
            # Sustituimos todos los valores conocidos de la solución en la expresión de delta[j]
            # (El 'simplify()' es opcional pero ayuda a SymPy a reconocer el 0)
            val_sust = delta_val.subs(sol_completa).simplify()

            if val_sust == 0:
                # CASO: delta_j = 0 (simbólicamente)
                casos_part.append(f"delta{j}=0")
                inacti_res.append(res[j])
            else:
                # CASO: delta_j != 0 (es un número != 0 o una expresión simbólica)
                casos_part.append(f"delta{j}!=0")
                acti_delta.append(delta[j])

        caso = ", ".join(casos_part) if casos_part else "Sin restricciones activas"
        # --- FIN DE DEDUCCIÓN DE CASOS ---
        
        t_cond = []
        posible = True # variable para ver si se cumple 2º y 3º KT

# 2º KT: multiplicadores no negativos (delta_i >= 0)
        for d in acti_delta:
            val = sol_completa.get(d, 0)
            try:
                if t_param:
                    cond = sp.reduce_inequalities(val >= 0, t_param)
                else:
                    cond = (val.subs(sol_completa) >= 0)
                    
                if cond == False: 
                    posible = False
                    break
                elif cond != True: 
                    t_cond.append(f"{cond}")
            except Exception:
                # No se pudo reducir (ej. expresión compleja)
                t_cond.append(f"{val} >= 0")

        if not posible:
            result.append(f"Si {caso} => [{var_res}]. No es posible (2º KT falla, delta < 0)")
            continue # Probar la siguiente solución 's'

# 3º KT: restricciones deben cumplirse (g_i <= 0)
        for g in inacti_res:
            val = g.subs(sol_completa)
            try:
                if t_param:
                    cond = sp.reduce_inequalities(val <= 0, t_param)
                else:
                    cond = (val.subs(sol_completa) <= 0)
                    
                if cond == False: 
                    posible = False
                    break
                elif cond != True: 
                    t_cond.append(f"{cond}")
            except Exception:
                # No se pudo reducir
                t_cond.append(f"{val} <= 0")

        # 5. Guardar el resultado de esta solución 's'
        if posible:
            cond_str = " & ".join([str(c) for c in t_cond]) if t_cond else "Siempre valido"
            result.append(f"Si {caso} => [{var_res}]. Valido cuando: {cond_str}")
        else:
            result.append(f"Si {caso} => [{var_res}]. No es posible (3º KT falla, g > 0)")

    return result

#Ejemplo:
'''
   t = sp.Symbol('t', real=True)
   u = sp.Function('u')(t)
   res = [u - 2, -u]
   lam = 2 * sp.exp(2 - t) - 2
   fun = -u**2 + (lam - 3) * u 
   solve_kt(fun, res, [u], t, max=True)'''
