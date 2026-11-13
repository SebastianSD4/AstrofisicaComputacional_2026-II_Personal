import numpy as np

# Este script debe devolver la malla 2.5D descrita en 'pluto.ini'

# Función para crear el estiramiento de las celdas (s):
def strech_ratio(N, Ldx):
    alpha = 1.01

    for _ in range(50):
        scrh = alpha * (alpha**N - 1)/(alpha - 1)

        f     = np.log(scrh/Ldx)
        df_dx = (1/alpha) + (N * (alpha**(N - 1)) / (alpha**N - 1)) - (1 / (alpha - 1)) 

        # Newton-Raphson:
        d_alpha = f / df_dx
        alpha = alpha - d_alpha

        if abs(d_alpha) < (1e-14 * alpha):
            break

    return alpha

# Función para crear la malla:
def grid(stops, ncells, ctypes):
    nseg = len(ncells)
    total_cells = sum(ncells)

    dx = np.zeros(total_cells)
    xl = np.zeros(total_cells)
    xr = np.zeros(total_cells)

    seg_idx = []
    curr    = 0
    for n in ncells:
        seg_idx.append((curr, curr + n - 1))
        curr += n

    done = [False] * nseg

    # Primero se llenan todos los segmentos de tipo uniforme
    for iseg in range(nseg):
        # Creación de segmento uniforme (u)
        if ctypes[iseg] == 'u':

            L = stops[iseg + 1] - stops[iseg]  # Longitud del segmento
            N = ncells[iseg]                   # Número de celdas en el segmento
            idx_start, idx_end = seg_idx[iseg] # índices iniciales y finales del segmentos, calculados antes
            dx_val = L / N
            for i in range(idx_start, idx_end + 1):
                dx[i] = dx_val                                  # Todas las celdas tienen el mismo ancho dx
                xl[i] = stops[iseg] + (i - idx_start) * dx_val  # Aumentando desde el primer borde izquierdo (stop[iseg] y sumando dx)
                xr[i] = xl[i] + dx_val                          # Sumar a cada borde izquierdo el ancho del segmento para obtener el borde derecho

            done[iseg] = True

    # Luego se crean todos los segmentos de tipo estirado
    all_done = False
    while not all_done:
        for iseg in range(nseg):
            # Creación de segmento estirado (s)
            if ctypes[iseg] == 's' and not done[iseg]:
                next_seg = -1
                if iseg + 1 < nseg and done[iseg + 1]:
                    next_seg = iseg + 1
                elif iseg - 1 >= 0 and done[iseg - 1]:
                    next_seg = iseg - 1

                if next_seg != -1:
                    N = ncells[iseg]
                    L = stops[iseg + 1] - stops[iseg]
                    idx_start, idx_end = seg_idx[iseg]

                    # Estiramiento inverso (decreciente)
                    if next_seg > iseg:
                        dx_ref = dx[seg_idx[next_seg][0]]
                        alpha = strech_ratio(N, L / dx_ref)

                        xr_current = stops[iseg + 1]
                        for i in range(idx_end, idx_start - 1, -1):
                            dx[i] = (alpha ** (idx_end - i + 1)) * dx_ref
                            xr[i] = xr_current 
                            xl[i] = xr[i] - dx[i]
                            xr_current = xl[i]
                    
                    # Estiramiento común (creciente)
                    else:
                        dx_ref = dx[seg_idx[next_seg][1]]
                        alpha  = strech_ratio(N, L / dx_ref)

                        xl_current = stops[iseg]
                        for i in range(idx_start, idx_end + 1):
                            dx[i] = (alpha ** (i - idx_start + 1)) * dx_ref 
                            xl[i] = xl_current
                            xr[i] = xl[i] + dx[i]
                            xl_current = xr[i]

                    done[iseg] = True
                    break

        # Terminar cuando todos los segmentos hayan sido rellenados correctamente según su tipo
        all_done = all(done)

    xc    = 0.5 * (xl + xr)
    faces = np.append(xl, xr[-1])

    # Se obtiene un array con todos los valores extremos de los segmentos (faces) y los valores centrales de cada celda (xc)
    return faces, xc

# ------ PARÁMETROS DE RED ------

# "Paradas" o límites principales de los segmentos:
x1_stops = [1.0, 11.0, 20.0]                 # Dos segmentos : [1, 11]  y [11, 20]
x2_stops = [0.0, np.pi/4, 3*np.pi/4, np.pi]  # Tres segmentos: [0, pi/4], [pi/4, 3pi/4] y [3pi/4, pi]
x3_stops = [0.0, 1.0]                        # Un segmentos (trivial): [0, 1]

# Número de celdas en cada segmento:
x1_ncel = [320, 72]
x2_ncel = [72, 256, 72]
x3_ncel = [1]

# Tipo de celdas ('u': uniforme, 's': estirada)
x1_celtype = ['u', 's']
x2_celtype = ['s', 'u', 's']
x3_celtype = ['u']

# ------ GENERACIÓN DE LA MALLA ------
x1_faces, x1_centers = grid(x1_stops, x1_ncel, x1_celtype)
x2_faces, x2_centers = grid(x2_stops, x2_ncel, x2_celtype)

print(f"\n Malla radial (x1) generada. Celdas totales: {len(x1_centers)}.\n Malla angular (x2) generada. Celdas totales: {len(x2_centers)}")

for faces, stops in [(x1_faces, x1_stops), (x2_faces, x2_stops)]:
    assert np.all(np.diff(faces) > 0)                     # monótona
    assert np.allclose(faces[[0, -1]], [stops[0], stops[-1]], atol=1e-12)
    # los límites internos de segmento deben caer sobre las caras
    for s in stops[1:-1]:
        assert np.min(np.abs(faces - s)) < 1e-10, s