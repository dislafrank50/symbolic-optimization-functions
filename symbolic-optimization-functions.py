import sympy as sp

'''=================================== FILTRO PARA EL SOLVE ================================='''

def check_sols(sols, vars):
    """
    Normaliza las soluciones. Si una variable falta, asume que es libre.
    Ej: {x:0} en variables (x,y) se convierte en {x:0, y:y}
    """
    valid_sols = []
    # Convertimos vars a lista por si acaso
    vars = list(vars)
    
    for sol in sols:
        # Copiamos para no alterar el original
        full_sol = sol.copy()
        
        # Rellenar variables libres
        # Si 'y' no está en la solución, significa que y = y (es libre)
        for var in vars:
            if var not in full_sol:
                full_sol[var] = var
        
        # Filtramos complejos e infinitos numéricos
        es_valido = True
        for k, v in full_sol.items():
            s_val = sp.sympify(v)
            if s_val.has(sp.I, sp.oo, -sp.oo, sp.zoo) and s_val.is_number:
                es_valido = False
                break
        
        if es_valido:
            valid_sols.append(full_sol)
            
    return valid_sols


def clasificar_punto(HS, sol):
    """
    Clasifica un puntos críticos analizando sus Eigenvalues para distinguir entre indeterminado y silla.
    """
    # Evaluar la Hessiana en el punto
    HS_eval = HS.subs(sol)
    HS_mat = sp.Matrix(HS_eval)
    
    # Calcular valores propios
    ev_dict = HS_mat.eigenvals()
    evals = list(ev_dict.keys()) # Lista de eigenvalues
    
    # Preparamos listas para el análisis
    evals_sym = [sp.sympify(e) for e in evals]
    
    # CASO 1: Si hay variables libres en los eigenvalues 
    if any(e.free_symbols for e in evals_sym):
        # Filtramos los que son cero
        non_zeros = [e for e in evals_sym if e != 0]
        zeros = [e for e in evals_sym if e == 0]
        
        if not non_zeros:
            return "Plano (Hessiana nula)"
            
        # Intentamos resolver condiciones
        condiciones = []
        for e in non_zeros:
            # Buscamos cuándo es positivo y cuándo negativo
            pos_cond = sp.solve(e > 0)
            neg_cond = sp.solve(e < 0)
            condiciones.append(f"Signo de ({e}): Positivo si {pos_cond}, Negativo si {neg_cond}")
        
        base_msg = "Punto Crítico Paramétrico (Línea o Superficie)."
        if zeros:
            base_msg += " (Semidefinido - Hessiana con ceros)."
        
        return f"{base_msg}\n      Condiciones:\n      " + "\n      ".join(condiciones)

    # CASO 2: Si es puramente numérico (Puntos aislados)
    else:
        # Contamos signos
        pos = len([e for e in evals_sym if e > 0])
        neg = len([e for e in evals_sym if e < 0])
        zeros = len([e for e in evals_sym if e == 0])
        total = len(evals_sym)

        if zeros > 0:
            # Matriz Semidefinida
            if pos > 0 and neg == 0:
                return "Mínimo Local (Semidefinida Positiva - Valle plano)"
            elif neg > 0 and pos == 0:
                return "Máximo Local (Semidefinida Negativa - Cresta plana)"
            else:
                return "Punto de Silla Degenerado (o indeterminado)"
        else:
            # Matriz Definida (Caso estándar)
            if pos == total:
                return "Mínimo Local Estricto"
            elif neg == total:
                return "Máximo Local Estricto"
            else:
                return "Punto de Silla"

def analizar_funcion(fun, vars):

    grad = [sp.diff(fun, v) for v in vars]
    # Solve devuelve lista de diccionarios
    sols_raw = sp.solve(grad, vars, dict=True)
 '''Ponemos el nuevo check_sols con vars que ahora si una variable falta, asume que es libre
 '''   
    # Filtramos y normalizamos
    sols = check_sols(sols_raw, vars)

    HS = sp.hessian(fun, vars)
    resultado=[]
    for sol in sols:
        tipo = clasificar_punto(HS, sol)
        resultado.append({'punto': sol, 'tipos': tipo})
    return resultado
''' Ahora funciona para fun1 = 9*x**2 + 6*x*y + y**2
    fun2 = x**2*y
    '''

     
'''=================================== LAGRANGEN ================================'''

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
'''Ponemos el nuevo check_sols con vars que ahora si una variable falta, asume que es libre
'''

fun3=5*x**2+6*y**2+7*x*y"
    sols = check_sols(sols, vars)
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

¿como podemos calcular la matriz asociada a la forma cuadratica 
restringida de esta forma cuadratica "5*x**2+6*y**2+7*x*y" restringida en "x**2 + y**2-1" 
en el punto (0,1)?

'''Para calcular la matriz asociada a la forma cuadrática 
restringida en el punto (0,1),comenzamos construyendo la matriz
simétrica A que representa la forma cuadrática original.
 En esta matriz, los coeficientes de x^2 y y^2 se colocan en la 
 diagonal principal, mientras que la mitad del coeficiente del
 término cruzado x y se coloca en las posiciones fuera de la diagonal. 
 Luego debemos encontrar el espacio tangente a la restricción en el punto 
 dado. 
 
 Para ello, calculamos el Jacobiano de la función 
 de restricción x^2 + y^2 - 1 y lo evaluamos numéricamente en el 
 punto (0,1). Este vector gradiente representa la dirección 
  normal a la curva. El paso clave es hallar el 
 espacio nulo de este gradiente evaluado; esto 
 implica encontrar una matriz base Z compuesta por vectores 
 que sean ortogonales al gradiente. Geométricamente, Z contiene las 
 direcciones en las que nos podemos mover tangencialmente a la 
 restricción sin salirnos de ella. Finalmente, la matriz asociada 
 a la forma cuadrática restringida se obtiene proyectando la matriz 
 original sobre este espacio tangente. La operación matemática 
 precisa es el producto matricial HR = Z^T A Z. El 
 resultado de esta operación elimina los efectos de la curvatura en 
 las direcciones que violan la restricción, dejando únicamente la 
 información de la segunda derivada a lo largo de la curva permitida.'''

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

    # Condiciones de 1º KKT
    kt_1st = [sp.Eq(sp.diff(L, v), 0) for v in vars]
    # Condiciones de 4º KKT (Holgura Complementaria)
    kt_4th = [sp.Eq(delta[j] * res[j], 0) for j in range(r)]
    
    # Combinar ecuaciones
    kt_eqs = kt_1st + kt_4th
    
    # Resolver sistema
    sols = sp.solve(kt_eqs, vars + delta, dict=True)
    sols = check_sols_compl(sols)

    if not sols:
        return [] # Retorna lista vacía si no hay solución
        
    valid_candidates = []
    
    # Iterar sobre soluciones
    for s in sols:
        # Rellena los valores encontrados
        sol_completa = {k: s.get(k, k) for k in (vars + delta)}

        # Clasificación de restricciones (necesaria para la lógica de validación)
        acti_delta = []   # Para 2º KT (delta_i >= 0)
        inacti_res = []   # Para 3º KT (res_i <= 0)
        
        for j in range(r):
            delta_val = sol_completa.get(delta[j], 0)
            val_sust = delta_val.subs(sol_completa).simplify()

            if val_sust == 0:
                inacti_res.append(res[j]) # Restricción inactiva (g < 0), comprobar g
            else:
                acti_delta.append(delta[j]) # Restricción activa (g = 0), comprobar delta

        
        t_cond = []
        posible = True 

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
                    t_cond.append(cond)
            except Exception:
                t_cond.append(val >= 0)

        if not posible:
            continue # Descartamos este punto y pasamos al siguiente

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
                    t_cond.append(cond)
            except Exception:
                t_cond.append(val <= 0)

       
        if posible:
            # Creamos una copia limpia del candidato
            candidate = sol_completa.copy()
            
            # Si hubo condiciones simbólicas las agregamos
            if t_cond:
                candidate['condiciones'] = t_cond
            
            valid_candidates.append(candidate)

    return valid_candidates

#Ejemplo:
'''
   t = sp.Symbol('t', real=True)
   u = sp.Function('u')(t)
   res = [u - 2, -u]
   lam = 2 * sp.exp(2 - t) - 2
   fun = -u**2 + (lam - 3) * u 
   solve_kt(fun, res, [u], t, max=True)'''
