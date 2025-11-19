# -*- coding: utf-8 -*-
"""
Script para resolver numéricamente cuatro problemas de ecuaciones diferenciales
usando los métodos de Runge–Kutta de orden 2 (Heun) y orden 4.

Requisitos cumplidos:
- Lenguaje: Python
- Librerías: numpy, matplotlib
- Sin uso de SymPy ni métodos simbólicos
- Métodos numéricos implementados de forma genérica:
    * rk2(f, y0, t0, tf, h)
    * rk4(f, y0, t0, tf, h)
  que funcionan tanto para ecuaciones escalares como para sistemas (vectores numpy)
- Se comparan los resultados numéricos con las soluciones analíticas conocidas
  (Problemas 1, 2 y 3).
- Para el problema 4 (Lotka–Volterra) no hay solución analítica cerrada, por lo que
  se usa una solución de referencia calculada con RK4 y paso muy pequeño.
- Se calculan errores globales máximos y se grafican:
    * solución analítica / referencia vs RK2 vs RK4
    * error vs tiempo para cada método y cada tamaño de paso

Autor: (puedes poner tu nombre aquí)
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  MÉTODOS NUMÉRICOS GENERALES: RK2 (Heun) y RK4
# ============================================================

def rk2(f, y0, t0, tf, h):
    """
    Implementa el método de Runge–Kutta de orden 2 (método de Heun)
    para resolver el problema de valor inicial:

        y' = f(t, y),   t ∈ [t0, tf]
        y(t0) = y0

    Parámetros
    ----------
    f : función
        Función f(t, y) que devuelve la derivada. Debe aceptar:
        - t: escalar
        - y: escalar o vector (numpy array)
        y devolver un escalar o numpy array de la misma forma que y.
    y0 : escalar o array-like
        Valor inicial de la solución en t = t0.
    t0 : float
        Tiempo inicial.
    tf : float
        Tiempo final.
    h : float
        Tamaño de paso.

    Devuelve
    --------
    t : numpy.ndarray
        Vector de tiempos de dimensión (N+1,), donde N es el número de pasos.
    y : numpy.ndarray
        - Si y0 es escalar, y será un array de dimensión (N+1,)
        - Si y0 es vector, y será un array de dimensión (N+1, dim),
          donde dim = número de componentes de y0.
    """
    # Convertimos y0 a array para tratar de forma genérica
    y0_array = np.atleast_1d(np.array(y0, dtype=float))
    dim = y0_array.size

    # Número de pasos (suponemos que (tf - t0) es múltiplo de h)
    N = int(np.round((tf - t0) / h))

    # Vector de tiempos
    t = t0 + h * np.arange(N + 1)

    # Array para almacenar la solución
    y = np.zeros((N + 1, dim))
    y[0] = y0_array

    # Bucle principal de RK2 (Heun)
    for n in range(N):
        tn = t[n]
        yn = y[n]

        k1 = np.array(f(tn, yn))
        k2 = np.array(f(tn + h, yn + h * k1))

        y[n + 1] = yn + (h / 2.0) * (k1 + k2)

    # Si el problema era escalar, devolvemos un vector 1D
    if np.isscalar(y0):
        return t, y[:, 0]
    else:
        return t, y


def rk4(f, y0, t0, tf, h):
    """
    Implementa el método de Runge–Kutta de orden 4 para resolver:

        y' = f(t, y),   t ∈ [t0, tf]
        y(t0) = y0

    Parámetros
    ----------
    f : función
        Función f(t, y) que devuelve la derivada. Debe aceptar:
        - t: escalar
        - y: escalar o vector (numpy array)
        y devolver un escalar o numpy array de la misma forma que y.
    y0 : escalar o array-like
        Valor inicial de la solución en t = t0.
    t0 : float
        Tiempo inicial.
    tf : float
        Tiempo final.
    h : float
        Tamaño de paso.

    Devuelve
    --------
    t : numpy.ndarray
        Vector de tiempos de dimensión (N+1,).
    y : numpy.ndarray
        - Si y0 es escalar, y será un array de dimensión (N+1,)
        - Si y0 es vector, y será un array de dimensión (N+1, dim).
    """
    y0_array = np.atleast_1d(np.array(y0, dtype=float))
    dim = y0_array.size

    N = int(np.round((tf - t0) / h))
    t = t0 + h * np.arange(N + 1)

    y = np.zeros((N + 1, dim))
    y[0] = y0_array

    for n in range(N):
        tn = t[n]
        yn = y[n]

        k1 = np.array(f(tn, yn))
        k2 = np.array(f(tn + 0.5 * h, yn + 0.5 * h * k1))
        k3 = np.array(f(tn + 0.5 * h, yn + 0.5 * h * k2))
        k4 = np.array(f(tn + h,       yn + h * k3))

        y[n + 1] = yn + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    if np.isscalar(y0):
        return t, y[:, 0]
    else:
        return t, y


# ============================================================
#  PROBLEMA 1: ECUACIÓN DE PRIMER ORDEN
# ============================================================
# y' = -2y + 3*exp(-t),  y(0) = 1
# Solución analítica:
#   y(t) = 3*exp(-t) + C*exp(-2t)
# Condición inicial y(0) = 1:
#   1 = 3 + C  ->  C = -2
# Por tanto:
#   y(t) = 3*exp(-t) - 2*exp(-2t)


def f_orden1(t, y):
    """
    Función f(t, y) para el problema de primer orden:
        y' = -2y + 3*exp(-t)
    """
    return -2.0 * y + 3.0 * np.exp(-t)


def y_exacta_orden1(t):
    """
    Solución analítica explícita del problema de primer orden:

        y(t) = 3*exp(-t) - 2*exp(-2t)

    Parámetros
    ----------
    t : escalar o array-like

    Devuelve
    --------
    y : numpy.ndarray
        Valores de la solución exacta evaluada en t.
    """
    t = np.array(t, dtype=float)
    return 3.0 * np.exp(-t) - 2.0 * np.exp(-2.0 * t)


# ============================================================
#  PROBLEMA 2: ECUACIÓN DE SEGUNDO ORDEN COMO SISTEMA
# ============================================================
# Ecuación:  y'' + 4y = 0
# Condiciones iniciales:
#   y(0)   = 0
#   y'(0)  = 1
#
# Sistema equivalente:
#   y1 = y
#   y2 = y'
#   y1' = y2
#   y2' = -4*y1
#
# Solución analítica:
#   y(t) = (1/2) * sin(2t)
#   y'(t) = y2(t) = cos(2t)


def f_oscilador(t, Y):
    """
    Sistema para el problema de segundo orden y'' + 4y = 0.

    Y = [y1, y2] con:
        y1 = y
        y2 = y'

    Entonces:
        y1' = y2
        y2' = -4*y1
    """
    y1 = Y[0]
    y2 = Y[1]
    dy1 = y2
    dy2 = -4.0 * y1
    return np.array([dy1, dy2])


def exacta_oscilador(t):
    """
    Solución analítica del sistema del oscilador:

        y(t)  = (1/2) * sin(2t)
        y'(t) = cos(2t)

    Parámetros
    ----------
    t : escalar o array-like

    Devuelve
    --------
    Y_exacta : numpy.ndarray de forma (len(t), 2)
        Primera columna: y(t)
        Segunda columna: y'(t)
    """
    t = np.array(t, dtype=float)
    y1 = 0.5 * np.sin(2.0 * t)
    y2 = np.cos(2.0 * t)
    return np.vstack((y1, y2)).T


# ============================================================
#  PROBLEMA 3: SISTEMA 2x2 LINEAL
# ============================================================
# Sistema:
#   y' = x + y
#   x' = 3x - y
#
# Condiciones iniciales:
#   x(0) = 1
#   y(0) = 0
#
# Se puede escribir el vector de estado como:
#   X = [x, y]
#
# Entonces:
#   x' = 3x - y
#   y' = x + y
#
# Solución analítica (derivada externamente, sin SymPy en el código):
#   x(t) = e^{2t} * (t + 1)
#   y(t) = e^{2t} * t


def f_sistema_lineal(t, Y):
    """
    Sistema lineal 2x2:

        y' = x + y
        x' = 3x - y

    Tomamos el vector de estado como:
        Y = [x, y]

    Entonces:
        Y' = [x', y'] = [3x - y, x + y]
    """
    x = Y[0]
    y = Y[1]
    dx = 3.0 * x - y
    dy = x + y
    return np.array([dx, dy])


def exacta_sistema_lineal(t):
    """
    Solución analítica del sistema lineal 2x2:

        x(t) = e^{2t} * (t + 1)
        y(t) = e^{2t} * t

    Parámetros
    ----------
    t : escalar o array-like

    Devuelve
    --------
    XY_exacta : numpy.ndarray de forma (len(t), 2)
        Primera columna: x(t)
        Segunda columna: y(t)
    """
    t = np.array(t, dtype=float)
    x = np.exp(2.0 * t) * (t + 1.0)
    y = np.exp(2.0 * t) * t
    return np.vstack((x, y)).T


# ============================================================
#  PROBLEMA 4: SISTEMA NO LINEAL DE LOTKA–VOLTERRA
# ============================================================
# Sistema depredador–presa:
#
#   dx/dt = α x - β x y
#   dy/dt = δ x y - γ y
#
# donde:
#   x(t): población de presas (conejos)
#   y(t): población de depredadores (zorros)
#
# Parámetros (positivos):
#   α = 1.0
#   β = 0.1
#   δ = 0.075
#   γ = 1.0
#
# Condiciones iniciales:
#   x(0) = 10
#   y(0) = 5
#
# Este sistema no tiene solución analítica cerrada simple, así que
# para estimar el error usaremos una "solución de referencia"
# calculada con RK4 y un paso muy pequeño.


def f_lotka_volterra(t, Y):
    """
    Sistema de Lotka–Volterra (depredador–presa) con parámetros fijos:

        dx/dt = α x - β x y
        dy/dt = δ x y - γ y

    Parámetros elegidos:
        α = 1.0
        β = 0.1
        δ = 0.075
        γ = 1.0
    """
    alpha = 1.0
    beta = 0.1
    delta = 0.075
    gamma = 1.0

    x = Y[0]
    y = Y[1]

    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y

    return np.array([dx, dy])


# ============================================================
#  FUNCIONES PARA CÁLCULO DE ERRORES
# ============================================================

def error_maximo(y_exacto, y_numerico):
    """
    Calcula el error global máximo entre solución exacta y numérica.

    Para problemas escalares:
        error_max = max_t |y_exact(t) - y_num(t)|

    Para sistemas (vectores):
        error_max = max_{t, componente} |Y_exact(t) - Y_num(t)|

    Parámetros
    ----------
    y_exacto : numpy.ndarray
        Valores de la solución exacta.
    y_numerico : numpy.ndarray
        Valores de la solución numérica en los mismos tiempos.

    Devuelve
    --------
    float
        Error global máximo.
    """
    diff = np.abs(y_exacto - y_numerico)
    return np.max(diff)


def error_por_tiempo(y_exacto, y_numerico):
    """
    Calcula el error en función del tiempo.

    Para problemas escalares:
        e(t) = |y_exact(t) - y_num(t)|

    Para sistemas:
        e(t) = max sobre componentes de |Y_exact(t) - Y_num(t)|

    Parámetros
    ----------
    y_exacto : numpy.ndarray
        Valores de la solución exacta.
    y_numerico : numpy.ndarray
        Valores de la solución numérica.

    Devuelve
    --------
    e : numpy.ndarray
        Vector de errores, de la misma longitud que el vector de tiempos.
    """
    diff = np.abs(y_exacto - y_numerico)
    if diff.ndim == 1:
        return diff
    else:
        # Tomamos el máximo sobre componentes para cada instante de tiempo
        return np.max(diff, axis=1)


# ============================================================
#  FUNCIÓN PRINCIPAL
# ============================================================

def main():
    # Tamaños de paso a probar
    pasos = [0.1, 0.05, 0.01]

    # Diccionario de métodos para iterar fácilmente
    metodos = {
        "RK2": rk2,
        "RK4": rk4
    }

    # --------------------------------------------------------
    # PROBLEMA 1: ECUACIÓN DE PRIMER ORDEN
    # --------------------------------------------------------
    print("============================================")
    print("PROBLEMA 1: y' = -2y + 3*exp(-t), y(0) = 1")
    print("Intervalo: [0, 5]")
    print("============================================")

    t0_1 = 0.0
    tf_1 = 5.0
    y0_1 = 1.0

    errores_1 = []

    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, y_num = metodo(f_orden1, y0_1, t0_1, tf_1, h)
            y_ex = y_exacta_orden1(t)
            err_max = error_maximo(y_ex, y_num)
            errores_1.append((h, nombre_metodo, err_max))

    # Imprimir tabla de errores para problema 1
    print("\nh      método    error_maximo")
    for h, metodo_nombre, err in errores_1:
        print(f"{h:<6g}  {metodo_nombre:<6}   {err: .6e}")

    # Gráficas para problema 1
    # (a) Solución analítica vs RK2/RK4 para todos los h
    plt.figure()
    # Solución exacta en una malla fina (la de h más pequeño)
    h_fino = min(pasos)
    t_fino = np.arange(t0_1, tf_1 + h_fino, h_fino)
    y_ex_fino = y_exacta_orden1(t_fino)
    plt.plot(t_fino, y_ex_fino, 'k-', label="Exacta")

    # Añadimos soluciones numéricas
    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, y_num = metodo(f_orden1, y0_1, t0_1, tf_1, h)
            etiqueta = f"{nombre_metodo}, h={h}"
            plt.plot(t, y_num, '--', label=etiqueta)

    plt.title("Problema 1: solución y(t)")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # (b) Error vs tiempo para cada método y cada h
    plt.figure()
    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, y_num = metodo(f_orden1, y0_1, t0_1, tf_1, h)
            y_ex = y_exacta_orden1(t)
            e_t = error_por_tiempo(y_ex, y_num)
            etiqueta = f"{nombre_metodo}, h={h}"
            plt.plot(t, e_t, label=etiqueta)

    plt.title("Problema 1: error vs tiempo")
    plt.xlabel("t")
    plt.ylabel("Error absoluto")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # PROBLEMA 2: ECUACIÓN DE SEGUNDO ORDEN COMO SISTEMA
    # --------------------------------------------------------
    print("\n============================================")
    print("PROBLEMA 2: y'' + 4y = 0, y(0)=0, y'(0)=1")
    print("Sistema: y1' = y2, y2' = -4*y1")
    print("Intervalo: [0, 5]")
    print("============================================")

    t0_2 = 0.0
    tf_2 = 5.0
    # y1(0) = y(0) = 0, y2(0) = y'(0) = 1
    y0_2 = np.array([0.0, 1.0])

    errores_2 = []

    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, Y_num = metodo(f_oscilador, y0_2, t0_2, tf_2, h)
            Y_ex = exacta_oscilador(t)
            err_max = error_maximo(Y_ex, Y_num)
            errores_2.append((h, nombre_metodo, err_max))

    # Imprimir tabla de errores para problema 2
    print("\nh      método    error_maximo (sobre y e y')")
    for h, metodo_nombre, err in errores_2:
        print(f"{h:<6g}  {metodo_nombre:<6}   {err: .6e}")

    # Gráficas para problema 2
    # (a) Solución analítica vs RK2/RK4 (solo y1 = y)
    plt.figure()
    h_fino = min(pasos)
    t_fino = np.arange(t0_2, tf_2 + h_fino, h_fino)
    Y_ex_fino = exacta_oscilador(t_fino)
    plt.plot(t_fino, Y_ex_fino[:, 0], 'k-', label="y exacta")

    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, Y_num = metodo(f_oscilador, y0_2, t0_2, tf_2, h)
            etiqueta = f"{nombre_metodo}, h={h}"
            plt.plot(t, Y_num[:, 0], '--', label=etiqueta)

    plt.title("Problema 2: solución y(t)")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # (b) Error vs tiempo (norma infinita sobre [y, y'])
    plt.figure()
    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, Y_num = metodo(f_oscilador, y0_2, t0_2, tf_2, h)
            Y_ex = exacta_oscilador(t)
            e_t = error_por_tiempo(Y_ex, Y_num)
            etiqueta = f"{nombre_metodo}, h={h}"
            plt.plot(t, e_t, label=etiqueta)

    plt.title("Problema 2: error vs tiempo")
    plt.xlabel("t")
    plt.ylabel("Error absoluto (max sobre componentes)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # PROBLEMA 3: SISTEMA 2x2 LINEAL
    # --------------------------------------------------------
    print("\n============================================")
    print("PROBLEMA 3: sistema 2x2 lineal")
    print("   y' = x + y")
    print("   x' = 3x - y")
    print("   x(0) = 1, y(0) = 0")
    print("Intervalo: [0, 2]")
    print("============================================")

    t0_3 = 0.0
    tf_3 = 2.0
    # Vector de estado: [x, y]
    y0_3 = np.array([1.0, 0.0])

    errores_3 = []

    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, XY_num = metodo(f_sistema_lineal, y0_3, t0_3, tf_3, h)
            XY_ex = exacta_sistema_lineal(t)
            err_max = error_maximo(XY_ex, XY_num)
            errores_3.append((h, nombre_metodo, err_max))

    # Imprimir tabla de errores para problema 3
    print("\nh      método    error_maximo (sobre x e y)")
    for h, metodo_nombre, err in errores_3:
        print(f"{h:<6g}  {metodo_nombre:<6}   {err: .6e}")

    # Gráficas para problema 3
    # (a) Solución analítica vs RK2/RK4 para x(t) e y(t)
    plt.figure()
    h_fino = min(pasos)
    t_fino = np.arange(t0_3, tf_3 + h_fino, h_fino)
    XY_ex_fino = exacta_sistema_lineal(t_fino)
    plt.plot(t_fino, XY_ex_fino[:, 0], 'k-', label="x exacta")
    plt.plot(t_fino, XY_ex_fino[:, 1], 'k--', label="y exacta")

    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, XY_num = metodo(f_sistema_lineal, y0_3, t0_3, tf_3, h)
            plt.plot(t, XY_num[:, 0], ':', label=f"x {nombre_metodo}, h={h}")
            plt.plot(t, XY_num[:, 1], '-.', label=f"y {nombre_metodo}, h={h}")

    plt.title("Problema 3: soluciones x(t) e y(t)")
    plt.xlabel("t")
    plt.ylabel("x, y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # (b) Error vs tiempo (norma infinita sobre [x, y])
    plt.figure()
    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, XY_num = metodo(f_sistema_lineal, y0_3, t0_3, tf_3, h)
            XY_ex = exacta_sistema_lineal(t)
            e_t = error_por_tiempo(XY_ex, XY_num)
            etiqueta = f"{nombre_metodo}, h={h}"
            plt.plot(t, e_t, label=etiqueta)

    plt.title("Problema 3: error vs tiempo")
    plt.xlabel("t")
    plt.ylabel("Error absoluto (max sobre componentes)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # PROBLEMA 4: SISTEMA NO LINEAL DE LOTKA–VOLTERRA
    # --------------------------------------------------------
    print("\n============================================")
    print("PROBLEMA 4: sistema no lineal de Lotka–Volterra")
    print("   dx/dt = α x - β x y")
    print("   dy/dt = δ x y - γ y")
    print("Parámetros: α=1.0, β=0.1, δ=0.075, γ=1.0")
    print("Condiciones iniciales: x(0)=10, y(0)=5")
    print("Intervalo: [0, 30]")
    print("============================================")

    t0_4 = 0.0
    tf_4 = 30.0
    Y0_4 = np.array([10.0, 5.0])  # [x(0), y(0)]

    # Solución de referencia con RK4 y paso muy pequeño
    h_ref = 0.001
    t_ref, Y_ref = rk4(f_lotka_volterra, Y0_4, t0_4, tf_4, h_ref)

    def referencia_en_t(t):
        """
        Devuelve la solución de referencia (RK4 con h_ref) evaluada
        en los tiempos del vector t, suponiendo que t0_4 = 0 y
        que cada t es múltiplo entero de h_ref.
        """
        indices = np.round((t - t0_4) / h_ref).astype(int)
        indices = np.clip(indices, 0, len(t_ref) - 1)
        return Y_ref[indices]

    errores_4 = []

    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, Y_num = metodo(f_lotka_volterra, Y0_4, t0_4, tf_4, h)
            Y_ex = referencia_en_t(t)
            err_max = error_maximo(Y_ex, Y_num)
            errores_4.append((h, nombre_metodo, err_max))

    # Imprimir tabla de errores para problema 4
    print("\nh      método    error_maximo (vs referencia RK4, h=0.001)")
    for h, metodo_nombre, err in errores_4:
        print(f"{h:<6g}  {metodo_nombre:<6}   {err: .6e}")

    # Gráficas para problema 4
    # (a) Poblaciones x(t) e y(t): referencia vs RK2/RK4
    plt.figure()
    plt.plot(t_ref, Y_ref[:, 0], 'k-', label="x referencia")
    plt.plot(t_ref, Y_ref[:, 1], 'k--', label="y referencia")

    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, Y_num = metodo(f_lotka_volterra, Y0_4, t0_4, tf_4, h)
            plt.plot(t, Y_num[:, 0], ':', label=f"x {nombre_metodo}, h={h}")
            plt.plot(t, Y_num[:, 1], '-.', label=f"y {nombre_metodo}, h={h}")

    plt.title("Problema 4 (Lotka–Volterra): x(t) e y(t)")
    plt.xlabel("t")
    plt.ylabel("Población")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # (b) Error vs tiempo (norma infinita sobre [x, y])
    plt.figure()
    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, Y_num = metodo(f_lotka_volterra, Y0_4, t0_4, tf_4, h)
            Y_ex = referencia_en_t(t)
            e_t = error_por_tiempo(Y_ex, Y_num)
            etiqueta = f"{nombre_metodo}, h={h}"
            plt.plot(t, e_t, label=etiqueta)

    plt.title("Problema 4 (Lotka–Volterra): error vs tiempo")
    plt.xlabel("t")
    plt.ylabel("Error absoluto (vs referencia)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # (c) Diagrama de fase x vs y
    plt.figure()
    plt.plot(Y_ref[:, 0], Y_ref[:, 1], 'k-', label="Referencia")

    for h in pasos:
        for nombre_metodo, metodo in metodos.items():
            t, Y_num = metodo(f_lotka_volterra, Y0_4, t0_4, tf_4, h)
            plt.plot(Y_num[:, 0], Y_num[:, 1], '--',
                     label=f"{nombre_metodo}, h={h}")

    plt.title("Problema 4 (Lotka–Volterra): diagrama de fase")
    plt.xlabel("x(t) - presas")
    plt.ylabel("y(t) - depredadores")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Mostrar todas las figuras
    plt.show()


# ============================================================
#  PUNTO DE ENTRADA DEL SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
