# Este código implementa la solución/integración temporal 
import os, sys, time, argparse
import numpy as np
import init as I
import grid as G

# ----------------------------------------------------------------------
# INTERRUPTORES Y PARÁMETROS
# ----------------------------------------------------------------------

CFL           = 0.4     # Número de Courant
CFL_MAX_VAR   = 1.1     # Límite de crecimiento del dt
FIRST_DT      = 1.0e-4
TSTOP         = 100.0
DT_OUT        = 5.0
RHO_MIN       = 1.0e-6  # Para evitar divisiones por 0 o valores negativos
P_MIN         = 1.0e-6

GAMMA = I.gamma
NG    = I.NG
GM    = 1.0

# ----------------------------------------------------------------------
# MALLA Y GEOMETRÍA
# ----------------------------------------------------------------------

class Mesh:
    def __init__(self):
        # Traer caras activas
        self.x1f, self.x2f   = G.x1_faces, G.x2_faces
        # Calcular caras totales (fantasma + activas)
        self.x1ft, self.x2ft = I.extend_faces(self.x1f, self.x2f)
        # f1 y f2 son el total de las caras
        f1, f2 = self.x1ft, self.x2ft

        # Calcular r y theta en los centros como el promedio de las caras
        self.r   = 0.5*(f1[1:] + f1[:-1])
        self.th  = 0.5*(f2[1:] + f2[:-1])
        # Ancho de cada celda
        self.dr  = np.diff(f1)
        self.dth = np.diff(f2)

        # Calcular número de pasos/celdas totales
        self.n1, self.n2 = self.r.size, self.th.size
        # Calcular índices activos
        self.ib, self.ie = NG, self.n1 - NG
        self.jb, self.je = NG, self.n2 - NG
        # Calcular número de celdas activas
        self.N1, self.N2 = self.ie - self.ib, self.je - self.jb

        # Calcular áreas y volumen integrando y con axisimetría
        self.Ar, self.At = I._areas(f1, f2)
        co = np.cos(f2)
        # Se obtiene el volumen en todas las celdas (activas + fantasma)
        self.Vol = ((f1[1:]**3 - f1[:-1]**3)/3.0)[:, None] * (co[:-1] - co[1:])[None, :]

        # Extraer región activa
        # a1 es el slicing ib:ie, recoge toda la región activa radial, a2 (jb:je) recoge la región activa en theta
        a1, a2 = slice(self.ib, self.ie), slice(self.jb, self.je)
        self.a1, self.a2 = a1, a2
        # Todas las áreas radiales y angulares activas
        self.Ar_a = self.Ar[self.ib:self.ie+1, a2]
        self.At_a = self.At[a1, self.jb:self.je+1]
        # Volúmenes activos
        self.V_a  = self.Vol[a1, a2]
        
        # Centros y caras activos en forma de vectores columna y fila
        self.rc_a  =  self.r[a1][:, None]
        self.thc_a = self.th[a2][None, :]
        self.rf_a  = f1[self.ib:self.ie+1][:, None]
        self.thf_a = f2[self.jb:self.je+1][None, :]

        # Separaciones activas también en vectores columna y fila
        self.dr_a  =  self.dr[a1][:, None]
        self.dth_a = self.dth[a2][None, :] 

        # Senos, cotangentes en caras y centros en celdas activas
        self.sin_c = np.sin(self.thc_a)
        self.cot_c = np.cos(self.thc_a)/self.sin_c
        self.sin_f = np.sin(self.thf_a)         

        # Radio cilíndrico w = r sin(theta)
        self.w_c = self.rc_a * self.sin_c   # Centros
        self.w_1 = self.rf_a * self.sin_c   # Caras radiales
        self.w_2 = self.rc_a * self.sin_f   # Caras angulares

        # Área de celda meridional en el plano dr x dtheta
        self.Am = 0.5 * (self.rf_a[1:]**2 - self.rf_a[:-1]**2) * self.dth_a

        # Divisiones seguras (para Constrained Transport)
        self.At_safe = np.where(self.At_a > 0, self.At_a, 1.0)

# ----------------------------------------------------------------------
# RECONSTRUCCIÓN A LO LARGO DEL EJE 1
# ----------------------------------------------------------------------

def recon(q, xc, w):
    # Con variables primitivas q, centros xc, y anchos w
    # Devuelve los valores extrapolados de las variables en las caras radiales
    n = q.shape[1]  # q = (núm. var, núm de celdas eje 1, num de celdas eje 2)
    dc = np.diff(xc)[None, :, None]  # Distancia entre centros dx
    g  = np.diff(q, axis=1)/dc       # Gradiente entre celdas dq/dx
    a, b = g[:, :-1], g[:, 1:]       # Pendientes hacia izq y der
    den = np.abs(a) + np.abs(b)
    sig = np.where(den > 0, (a*np.abs(b) + np.abs(a)*b)/np.where(den > 0, den, 1.0), 0.0)
    w3  = w[None, :, None]
    # Extrapolación desde el centro hacia los bordes
    qL  = q[:, 1:n-2] + 0.5*w3[:, 1:n-2]*sig[:, 0:n-3]
    qR  = q[:, 2:n-1] - 0.5*w3[:, 2:n-1]*sig[:, 1:n-2]  

    return qL, qR

# ----------------------------------------------------------------------
# Flujo de Rusanov (solucionador, se llama una vez por dirección y etapa)
# ----------------------------------------------------------------------

def _phys(W, Bn):
    # W son las variables rho, p, vn, vt, vp, Bt, Bp, Q desde recon

    rho, p, vn, vt, vp, Bt, Bp, Q = W
    # Norma cuadrada del campo
    B2 = Bn*Bn + Bt*Bt + Bp*Bp
    # Presión total
    pt = p + 0.5*B2
    # Energía total
    E  = p/(GAMMA - 1.0) + 0.5*rho*(vn*vn + vt*vt + vp*vp) + 0.5*B2
    # v * B
    vB = vn*Bn + vt*Bt + vp*Bp

    # Variables (estados) y sus respectivos flujos
    U = np.array([rho, rho*vn, rho*vt, rho*vp, E, Bt, Bp, rho*Q])
    F = np.array([rho*vn, 
                  rho*vn*vn + pt - Bn*Bn,
                  rho*vn*vt - Bn*Bt,
                  rho*vn*vp - Bn*Bp,
                  (E + pt)*vn - Bn*vB,
                  vn*Bt - vt*Bn,
                  vn*Bp - vp*Bn,
                  rho*Q*vn])

    # a^2 y b^2 para la velocidad magnetosónica
    a2 = GAMMA*p/rho
    b2 = B2/rho
    s  = a2 + b2
    cf = np.sqrt(0.5*(s + np.sqrt(np.maximum(s*s - 4.0*a2*Bn*Bn/rho, 0.0))))

    return U, F, np.abs(vn) + cf

def rusanov(WL, WR, Bn):
    UL, FL, sL = _phys(WL, Bn)
    UR, FR, sR = _phys(WR, Bn)
    S = np.maximum(sL, sR)

    # Calcular el flujo de Rusanov con los estados/flujos desde _phys
    return 0.5*(FL + FR) - 0.5*S*(UR - UL)

# ----------------------------------------------------------------------
# CONVERSIÓN ENTRE VARIABLES PRIMITIVAS Y CONSERVADAS - EN REGIÓN ACTIVA
# ----------------------------------------------------------------------

def prim_to_cons(P, M):
    # De primitivas (las que guarda el script): V = (rho, prs, v, B, Q)
    # a conservadas (U), las que cumplen el balance de flujos: U = (rho, m, L, E_t, B, rho*Q)
    a1, a2 = M.a1, M.a2

    # Primitivas
    rho, p, v1, v2, v3, B1, B2, B3, Q = P[:, a1, a2] 
    # En. total
    E = p/(GAMMA - 1.0) + 0.5*rho*(v1*v1 + v2*v2 + v3*v3) + 0.5*(B1*B1 + B2*B2 + B3*B3)

    # Br y B_theta se actualizan por separado (CT)
    return np.array([rho, rho*v1, rho*v2, M.w_c*rho*v3, E, B3, rho*Q])

def restore(P, Bs1, Bs2, U, B1a, B2a, M):

    # Estado del paso anterior (se copian por RK2)
    P, Bs1, Bs2 = P.copy(), Bs1.copy(), Bs2.copy()
    a1, a2 = M.a1, M.a2
    Bs1[M.ib:M.ie+1, a2] = B1a
    Bs2[a1, M.jb:M.je+1] = B2a

    # Nuevo estado conservado U
    D, m1, m2, L, E, B3, DQ = U
    rho = np.maximum(D, RHO_MIN)
    v1, v2, v3, = m1/rho, m2/rho, L/(M.w_c*rho)
    B1c = 0.5*(B1a[1:, :] + B1a[:-1, :])
    B2c = 0.5*(B2a[:, 1:] + B2a[:, :-1])
    p = (GAMMA - 1.0)*(E - 0.5*rho*(v1*v1 + v2*v2 + v3*v3) - 0.5*(B1c**2 + B2c**2 + B3**2))
    p = np.maximum(p, P_MIN)

    # Guardar estado
    P[I.RHO, a1, a2] = rho
    P[I.PRS, a1, a2] = p
    P[I.VX1, a1, a2] = v1
    P[I.VX2, a1, a2] = v2
    P[I.VX3, a1, a2] = v3
    P[I.BX1, a1, a2] = B1c
    P[I.BX2, a1, a2] = B2c
    P[I.BX3, a1, a2] = B3
    P[I.TRC, a1, a2] = DQ/rho

    P, Bs1, Bs2, = I.boundary(P, Bs1, Bs2, M.x1ft, M.x2ft)

    return P, Bs1, Bs2

# ----------------------------------------------------------------------
# LADO DERECHO dU/dt, dB/dt (CT)
# ----------------------------------------------------------------------

# Operador L(U), lado derecho espacial completo
def rhs(P, Bs1, Bs2, M):
    ib, ie, jb, je = M.ib, M.ie, M.jb, M.je
    rho, p, v1, v2, v3, B1, B2, B3, Q = P

    # Flujos en r
    W1 = np.array([rho, p, v1, v2, v3, B2, B3, Q])
    qL, qR = recon(W1, M.r, M.dr)
    F1 = rusanov(qL, qR, Bs1[2:M.n1 - 1, :])[:, :, jb:je]

    # Flujos en theta
    W2 = np.swapaxes(np.array([rho, p, v2, v1, v3, B1, B3, Q]), 1, 2)
    qL, qR = recon(W2, M.th, M.dth)
    F2 = np.swapaxes(rusanov(qL, qR, Bs2.T[2:M.n2 - 1, :]), 1, 2)

    F2[6][:,  0] = 0.0
    F2[6][:, -1] = 0.0
    F2e = F2[:, ib-1:ie+1, :]
    F2  = F2[:, ib:ie, :]

    Ar, At, V = M.Ar_a, M.At_a, M.V_a

    # Calcula parte del balance de flujos
    def div(f1, f2):
        return -((Ar[1:]*f1[1:] - Ar[:-1]*f1[:-1]) + (At[:, 1:]*f2[:, 1:] - At[:, :-1]*f2[:, :-1])) / V

    a1, a2,    = M.a1, M.a2
    rho_a, p_a = rho[a1, a2], p[a1, a2]
    u1, u2, u3 = v1[a1, a2], v2[a1, a2], v3[a1, a2]
    b1, b2, b3 = B1[a1, a2], B2[a1, a2], B3[a1, a2]
    
    pt = p_a + 0.5*(b1*b1 + b2*b2 + b3*b3)
    g  = -GM/M.rc_a**2

    dU    = np.empty((7, ) + V.shape)
    # Masa
    dU[0] = div(F1[0], F2[0])   
    # m_r (+ gravedad)
    dU[1] = div(F1[1], F2[2]) + pt*(Ar[1:] - Ar[:-1])/V + (rho_a*(u2*u2 + u3*u3) - b2*b2 - b3*b3)/M.rc_a + rho_a*g 
    # m_0
    dU[2] = div(F1[2], F2[1]) + pt*(At[:, 1:] - At[:, :-1])/V + (-(rho_a*u1*u2 - b1*b2) + M.cot_c*(rho_a*u3*u3 - b3*b3))/M.rc_a
    # L
    dU[3] = div(F1[3]*M.w_1, F2[3]*M.w_2)
    # Energía
    dU[4] = div(F1[4], F2[4]) + rho_a*u1*g
    # B_phi
    dU[5] = -((M.rf_a[1:]*F1[6][1:] - M.rf_a[:-1]*F1[6][:-1])*M.dth_a + (F2[6][:, 1:] - F2[6][:, :-1])*M.dr_a)/M.Am
    # Rho, Q
    dU[6] = div(F1[7], F2[7])


    # Para el transporte constreñido = promedio de 4 flujos
    Eph = np.zeros((M.N1 + 1, M.N2 + 1))
    # Campo E_phi en las aristas en el eje calculado con el estado actual de B
    Eph[:, 1:-1] = 0.25*(-F1[5][:, :-1] - F1[5][:, 1:] + F2e[5][:-1, 1:-1] + F2e[5][1:, 1:-1])
    # Stokes para la cara radial
    dB1 = -M.rf_a*(M.sin_f[:, 1:]*Eph[:, 1:] - M.sin_f[:, :-1]*Eph[:, :-1])/Ar
    # Stokes para la cara angular
    dB2 = M.sin_f*(M.rf_a[1:]*Eph[1:] - M.rf_a[:-1]*Eph[:-1])/M.At_safe
    dB2[:,  0] = 0.0
    dB2[:, -1] = 0.0

    # Devuelve el cambio del sistema
    return dU, dB1, dB2

def advance(P, Bs1, Bs2, dt, M):
    # Conservadas al inicio del paso temporal
    U0  = prim_to_cons(P, M)
    # Copias de los campos en las caras
    B10 = Bs1[M.ib:M.ie+1, M.a2].copy()
    B20 = Bs2[M.a1, M.jb:M.je+1].copy()

    # Cambio de las variables conservadas y B en las caras
    dU, d1, d2 = rhs(P, Bs1, Bs2, M)
    # Primer sub-paso
    U1, B11, B21 = U0 + dt*dU, B10 + dt*d1, B20 + dt*d2 
    # Estado completo 
    S1 = restore(P, Bs1, Bs2, U1, B11, B21, M)

    # Segundo sub-paso (RK2)
    dU, d1, d2   = rhs(*S1, M)
    U2, B12, B22 = U1 + dt*dU, B11 + dt*d1, B21 + dt*d2 

    return restore(P, Bs1, Bs2, 0.5*(U0 + U2), 0.5*(B10 + B12), 0.5*(B20 + B22), M)

# ----------------------------------------------------------------------
# Paso de tiempo y salidas
# ----------------------------------------------------------------------

def compute_dt(P, M):
    # Calcular paso temporal
    rho, p, v1, v2, v3, B1, B2, B3, _ = P[:, M.a1, M.a2]
    a2  = GAMMA*p/rho
    B2t = (B1*B1 + B2*B2 + B3*B3)/rho
    s = a2 + B2t

    def cf(Bn):
        return np.sqrt(0.5*(s + np.sqrt(np.maximum(s*s - 4*a2*Bn*Bn/rho, 0.0))))

    rate = (np.abs(v1) + cf(B1))/M.dr_a + (np.abs(v2) + cf(B2))/(M.rc_a*M.dth_a)

    # La celda más restrictiva fija el paso dt
    return CFL/rate.max() 

# Recolectar el flujo neto desde div_B y calcular un número adimensional relacionado al máximo flujo neto
def divB_max(Bs1, Bs2, M):
    S = I.div_B(Bs1[M.ib:M.ie+1, M.a2], Bs2[M.a1, M.jb:M.je+1], M.x1f, M.x2f)
    return np.abs(S).max() / np.abs(M.Ar_a*Bs1[M.ib:M.ie+1, M.a2]).max()

def save(P, M, n, OUT_DIR):
    a = (slice(None), M.a1, M.a2)  # Todas las variables, sólo celdas activas
    d = P[a]
    np.savez(os.path.join(OUT_DIR, f'data.{n:04d}.npz'),
             rho=d[I.RHO], prs=d[I.PRS], vx1=d[I.VX1], vx2=d[I.VX2], vx3=d[I.VX3],
             Bx1=d[I.BX1], Bx2=d[I.BX2], Bx3=d[I.BX3], tr1=d[I.TRC],
             x1=M.r[M.a1], x2=M.th[M.a2])

def initial_state(M):
    rho, prs, vx1, vx2, vx3, A3, tr1 = I.init(M.r[M.a1], M.th[M.a2])
    Bs1, Bs2, Bc1, Bc2 = I.init_B(M.x1f, M.x2f)

    vars_ = np.array([rho, prs, vx1, vx2, vx3, Bc1, Bc2, np.zeros_like(rho), tr1])

    P, B1t, B2t = I.pad_state(vars_, Bs1, Bs2)

    return I.boundary(P, B1t, B2t, M.x1ft, M.x2ft)

def main():
    global CFL
    ap = argparse.ArgumentParser()
    ap.add_argument('--tend', type=float, default=TSTOP)
    ap.add_argument('--dtout', type=float, default=DT_OUT)
    # ap.add_argument('--recon', default=RECON, choices=['flat', 'linear'])
    # ap.add_argument('--time', default=TIME_STEPPING, choices=['euler', 'rk2'])
    ap.add_argument('--cfl', type=float, default=CFL)
    ap.add_argument('--outdir', default='out')
    ap.add_argument('--maxsteps', type=int, default=10**9)
    ap.add_argument('--log', type=int, default=200)
    
    args = ap.parse_args()

    CFL = args.cfl

    os.makedirs(args.outdir, exist_ok=True)

    M = Mesh()
    P, Bs1, Bs2 = initial_state(M)
    t, dt_old, nstep, nout = 0.0, FIRST_DT, 0, 0
    save(P, M, nout, args.outdir)

    t_next = args.dtout

    m0 = (M.V_a*P[I.RHO, M.a1, M.a2]).sum()

    print(f'CFL={CFL}, Malla {M.N1}x{M.N2}')

    t_wall = time.time()

    # Bucle temporal
    while t < args.tend - 1e-12 and nstep < args.maxsteps:
        dt = min(compute_dt(P, M), CFL_MAX_VAR*dt_old if nstep else FIRST_DT)
        dt = min(dt, t_next - t, args.tend - t)
        P, Bs1, Bs2 = advance(P, Bs1, Bs2, dt, M)  # Construye el nuevo estado
        t += dt                                    # avanzar el reloj y contador
        dt_old = dt
        nstep += 1

        if not np.all(np.isfinite(P)):
            print(f'Nan/Inf en paso {nstep}, t={t:.4f}'); sys.exit(1)
        if nstep % args.log == 0:
            m = (M.V_a*P[I.RHO, M.a1, M.a2]).sum()
            print(f'Paso {nstep:7d}, t={t:9.4f}, dt={dt:.3e}, divB={divB_max(Bs1, Bs2, M):.1e}'
                  f'dM/M={(m - m0)/m0:+.1e} [{time.time() - t_wall:.0f}s]', flush=True)

        if t >= t_next - 1e-12:
            nout += 1
            save(P, M, nout, args.outdir)
            print(f'   -> data.{nout:04d}.npz  (t={t:.3f})', flush=True)
            t_next += args.dtout

    print(f'Completado: {nstep} paso, t={t:.4f}, {time.time() - t_wall:.0f}s')


if __name__ == '__main__':
    main()

