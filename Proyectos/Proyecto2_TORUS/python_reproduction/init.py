import numpy as np

# ------ PARÁMETROS DE INICIALIZACIÓN ------

rmin    = 3.0   # Borde interior del toro 
rmax    = 4.7   # Distancia de máxima densidad
rho_cut = 0.5   # Densidad de corte (potencial vector)
beta    = 200.0 # Beta de plasma
eta     = 1e-4  # Razón entre densidades
GM_a2   = 1/2   # Scale Height

# ------ PARÁMETROS DE TORO ------
gamma  = 5/3
n_poli = 1/(gamma - 1.0)
lk     = np.sqrt(rmax)
lk2    = lk**2
c      = -1/rmin + 0.5*lk2/(rmin**2)
K      = (c + 1/rmax - 0.5*lk2/(rmax**2))/(1 + n_poli)
B0     = np.sqrt(2*K/beta)

NG = 2 # Número de celdas fantasma (ghost) a usar para las condiciones de frontera

# índices de las variables primitivas

RHO, PRS, VX1, VX2, VX3, BX1, BX2, BX3, TRC = range(9)

# Devuelve el estado en un punto arbitrario de la malla
def _state(R, TH):
    r    = R*np.sin(TH)
    r_sf = np.maximum(r, 1e-10)

    # Densidad/presión en el toro
    a    = np.maximum(c + 1/R - 0.5*lk2/(r_sf**2), 0.0)
    rhot = (a/((1 + n_poli)*K))**n_poli
    prt  = K*rhot**gamma

    # Densidad/presión atmosférica
    rhoa = eta*np.exp((1/R - 1/rmin)*GM_a2)
    pra  = rhoa/GM_a2  

    # Máscara de celdas dentro del toro
    torus = (prt > pra) & (r > 2.0)
    
    rho = np.where(torus, rhot, rhoa)
    prs = np.where(torus, prt, pra)
    vx3 = np.where(torus, lk/np.maximum(r, 1e-8), 0.0)
    tr1 = np.where(torus, 1.0, 0.0)

    return rho, prs, vx3, tr1, rhot, r 

def _A3(rhot, r):
    A3 = np.zeros_like(rhot)

    mag   = (rhot > rho_cut) & (r > 2.0)
    A3[mag] = B0*(rhot[mag] - rho_cut)

    return A3 

# Estado inicial para variables primitivas (excepto B) (calculado en los centros)
def init(x1, x2):
    R, TH = np.meshgrid(x1, x2, indexing='ij')

    rho, prs, vx3, tr1, rhot, r = _state(R, TH)
    vx1 = np.zeros_like(R)
    vx2 = np.zeros_like(R)
    A3  = _A3(rhot, r)

    return rho, prs, vx1, vx2, vx3, A3, tr1
    
    # Hallar K y c:
    c = -1/rmin + 0.5*lk2/(rmin**2)
    K = (c + 1/rmax - 0.5*lk2/(rmax**2))/(1 + n_poli)

    # Densidad y presión del toro:
    a    = c + 1/R - 0.5*lk2/(r_sf**2)
    a    = np.maximum(a, 0.0)
    rhot = pow(a/((1 + n_poli)*K), n_poli)
    prt  = K*pow(rhot, gamma)

    # Densidad y presión atmosférica:
    rhoa = eta*np.exp((1/R - 1/rmin)*GM_a2)
    pra  = rhoa/GM_a2

    # Definición de celdas de toro / atmósfera:
    torus_mask = (prt > pra) & (r > 2.0)

    rho = np.where(torus_mask, rhot, rhoa)
    prs = np.where(torus_mask, prt, pra)

    # Velocidades:
    vx1 = np.zeros_like(R)
    vx2 = np.zeros_like(R)
    vx3 = np.where(torus_mask, lk / np.maximum(r, 1e-8), 0.0)

    # Trazador:
    tr1 = np.where(torus_mask, 1.0, 0.0)

    # Campo magnético inicial:
    B0 = np.sqrt(2 * K / beta)
    A3 = np.zeros_like(R)

    mag_mask = (rhot > rho_cut) & (r > 2.0)
    A3[mag_mask] = B0 * (rhot[mag_mask] - rho_cut)

    return rho, prs, vx1, vx2, vx3, A3, tr1

# Estado inicial para B (calculado en las caras) (CONSTRAINED TRANSPORT)
def init_B(x1f, x2f):
    R, TH = np.meshgrid(x1f, x2f, indexing='ij')
    _, _, _, _, rhot, r = _state(R, TH)
    A3 = _A3(rhot, r)

    rf = x1f[:, None]
    si = np.sin(x2f)[None, :]
    co = np.cos(x2f)[None, :]

    # Stokes: 
    #         Radio constante, theta variable
    #         B_prom = int A * dl / Área
    #         "+" caras superiores, "-" caras inferiores
    #         Br = (sinθ+ A+ - sinθ- A-) / (r (cosθ- - cosθ+))
    Bx1s = (si[:, 1:]*A3[:, 1:] - si[:, :-1]*A3[:, :-1]) / (rf*(co[:, :-1] - co[:, 1:]))

    #         Radio variable, theta constante        
    #         B_theta = (-1/r) d(rA)/dr --> -[rA]_[i-]^{i+} / ((r+^2 - r-^2)/2)
    rA   = rf*A3
    Bx2s = -(rA[1:, :] - rA[:-1, :]) / (0.5*(rf[1:]**2 - rf[:-1]**2))

    # Campo en los centros: promedio
    Bx1c = 0.5*(Bx1s[1:, :] + Bx1s[:-1, :])
    Bx2c = 0.5*(Bx2s[:, 1:] + Bx2s[:, :-1])

    # Los campos en las caras son de la forma Bs[i, j]: j-ésima cara izquieda de la i-ésima celda
    return Bx1s, Bx2s, Bx1c, Bx2c

def _areas(x1f, x2f):
    rf = x1f[:, None]
    si = np.sin(x2f)[None, :]
    co = np.cos(x2f)[None, :]
    Ar = rf**2 * (co[:, :-1] - co[:, 1:])
    At = si*0.5*(rf[1:]**2 - rf[:-1]**2)

    return Ar, At

# Flujo neto por las caras (prueba):
def div_B(Bx1s, Bx2s, x1f, x2f):
    Ar, At = _areas(x1f, x2f)

    S1 = Ar[1:]*Bx1s[1:] - Ar[:-1]*Bx1s[:-1]
    S2 = At[:, 1:]*Bx2s[:, 1:] - At[:, :-1]*Bx2s[:, :-1]

    return S1 + S2

# ------ CONDICIONES DE FRONTERA ------

# Extender a celdas fantasma
def extend_faces(f1, f2, ng=NG):
    # Se extrapolan hacia celdas fantasma fuera del dominio original usando los valores de las caras (f1=x1_faces, f2=x2_faces)
    dl, dr = f1[1] - f1[0], f1[-1] - f1[-2]

    # Para r simplemente se extrapola con dl, dr (left/right)
    # Para theta es necesario reflejar respecto de 0 y pi
    f1e    = np.concatenate([f1[0] - dl*np.arange(ng, 0, -1), f1, f1[-1] + dr*np.arange(1, ng + 1)])
    f2e    = np.concatenate([-f2[ng:0:-1], f2, 2*np.pi - f2[-2:-ng-2:-1]])

    return f1e, f2e

def pad_state(vars, Bs1, Bs2, ng=NG):
    # Crear arreglos "totales" que incluyan las celdas vacías fantasma y los valores físicos interiores, en caras y centros
    nv, N1, N2 = vars.shape
    Vt  = np.zeros((nv, N1 + 2*ng, N2 + 2*ng))
    B1t = np.zeros((N1 + 2*ng + 1, N2 + 2*ng))
    B2t = np.zeros((N1 + 2*ng, N2 + 2*ng + 1))
    Vt[:, ng:ng + N1, ng:ng + N2]   = vars # Centros
    B1t[ng:ng + N1 + 1, ng:ng + N2] = Bs1  # Caras radiales
    B2t[ng:ng + N1, ng:ng + N2 + 1] = Bs2  # Caras angulares

    return Vt, B1t, B2t 

def boundary(vars, Bs1, Bs2, x1ft, x2ft, ng=NG):
    # vars[n, i, j] = (variable, índice radial, índice angular)

    # Llenar los valores en las celdas fantasma que salen de pad_state
    n1t, n2t = vars.shape[1:] # Números totales de celdas
    ib, ie = ng, n1t - ng  # Las celdas activas radiales van en los índices ib, ib+1, ... , ie-1
    jb, je = ng, n2t - ng  # Las celdas activas angulares van en los índices jb, ... , je-1

    # Celdas activas:  [ib:ie] = ib, ib + 1, ... ie - 1 (igual con j)
    # Celdas fantasma: [:ib] = 0, 1, ..., ib - 1    (iniciales)
    #                  [ie:] = ie, ie + 1, ..., end (finales)
    
    # Celdas centradas X1 (cambiar sobre celdas radiales (segundo índice))
    vars[  :, :ib, :] = vars[:, ib:ib+1, :]                 # X1-beg userdef (outflow modificado), copiar el valor de la primera celda activa
    vars[VX1, :ib, :] = np.minimum(vars[VX1, :ib, :], 0.0)  # x1-beg para las velocidades: flujo interno está prohibido (la velocidad radial es ya sea menor a cero o 0)
    vars[  :, ie:, :] = vars[:, ie-1:ie, :]                 # x1-end outflow (gradiente cero)

    # Celdas X2: axisymmetric respecto al eje angular theta:

    #                   fantasma                      |                 activas                 |   fantasma  
    #    0, 1, ..., jb - 1 - k, ..., jn - 2, jb - 1   |   jb, jb + 1, ..., jb + k, ..., je - 1  |   je, je + 1, ..., END
    #         (celda espejo de jb + k)                |                                         |
    
    for k in range(ng):
        #                    theta = 0            theta = pi
        #              (fantasma,   activa)  (fantasma,   activa)
        for jl, jr in ((jb - 1 - k, jb + k), (je + k, je - 1 - k)):
            vars[:, :, jl] = vars[:, :, jr]  # Variables paralelas se dejan igual (vr y Br)
            for v in (VX2, VX3, BX2, BX3):   # Variable normales se reflejan (vtheta, Btheta y las de phi)
                vars[v, :, jl] *= -1.0       # Inversión para mantener el flujo al atravesar el polo 

    # B2s (Campo escalonado B_theta ^s) (está evaluado en las caras)
    Bs2[:ib, :] = Bs2[ib:ib+1, :]  # Fronteras radiales, gradiente nulo
    Bs2[ie:, :] = Bs2[ie-1:ie, :]  
    Bs2[:, jb]  = 0.0              # Fronteras angulares, se anula justo en los polos (punto que los centros no tocan)  
    Bs2[:, je]  = 0.0
    for k in range(1, ng + 1):
        Bs2[:, jb - k] = -Bs2[:, jb + k]  # Reflexión antisimétrica en las caras fantasma polares
        Bs2[:, je + k] = -Bs2[:, je - k]

    # B1s (Campo escalonado B_r ^2) (evaluado en las caras)
    for k in range(ng):
        Bs1[ib:ie+1, jb - 1 - k] = Bs1[ib:ie+1, jb + k]
        Bs1[ib:ie + 1, je + k]   = Bs1[ib:ie + 1, je - 1 - k]

    Ar, At = _areas(x1ft, x2ft)       # Áreas esféricas
    for i in range(ib - 1, -1, -1):   # Recorrer indices radiales hacia afuera (atrás)    (despejar Br^i)
        Bs1[i,   :] = (Ar[i+1]*Bs1[i+1, :] + At[i, 1:]*Bs2[i, 1:] - At[i, :-1]*Bs2[i, :-1]) / Ar[i]

    for i in range(ie, n1t):          # Recorres indices radiales hacia afuera (adelante) (despejar Br^{i+1})
        Bs1[i+1, :] = (Ar[i] * Bs1[i,   :] - At[i, 1:]*Bs2[i, 1:] + At[i, :-1]*Bs2[i, :-1]) / Ar[i+1]

    # Así toda celda fantasma tendrá divergencia nula.

    # Campo centrado, promedio
    vars[BX1] = 0.5*(Bs1[1:, :] + Bs1[:-1, :])
    vars[BX2] = 0.5*(Bs2[:, 1:] + Bs2[:, :-1])

    return vars, Bs1, Bs2

# [1, 2, 3, 4, 5]
# [1:]  = [2, 3, 4, 5]
# [:-1] = [1, 2, 3, 4]
# prom = 0.5 * ([1:] + [:-1])