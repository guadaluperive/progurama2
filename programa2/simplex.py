# simplex.py
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def print_tableau(tableau):
    """Función para imprimir la tabla simplex."""
    print("Tabla Simplex:")
    print(tableau)
    print()

def simplex(c, A, b):
    m, n = A.shape
    tableau = np.hstack((A, np.eye(m), b.reshape(-1, 1)))
    tableau = np.vstack((tableau, np.hstack((c, np.zeros(m + 1)))))
    
    while np.any(tableau[-1, :-1] > 0):
        pivot_col = np.argmax(tableau[-1, :-1])
        ratios = np.divide(tableau[:-1, -1], tableau[:-1, pivot_col], out=np.full(m, np.inf), where=tableau[:-1, pivot_col] > 0)
        pivot_row = np.argmin(ratios)
        
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
        
    solution = np.zeros(n)
    for i in range(n):
        column = tableau[:-1, i]
        if np.sum(column == 1) == 1 and np.sum(column == 0) == m - 1:
            solution[i] = tableau[np.where(column == 1)[0][0], -1]
    
    optimal_value = abs(tableau[-1, -1])
    return optimal_value, solution

def plot_graph(c, A, b):
    n = len(c)
    x = np.linspace(0, 50, 400)
    y = np.linspace(0, 50, 400)
    X, Y = np.meshgrid(x, y)
    
    for i in range(len(A)):
        plt.plot(x, (b[i] - A[i, 0] * x) / A[i, 1], label=f'Restricción {i+1}')

    z = c[0] * X + c[1] * Y
    plt.contour(X, Y, z, levels=[-100, -50, 0, 50, 100, 150], colors='k', linestyles='dashed')
    
    plt.xlim((0, 50))
    plt.ylim((0, 50))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Gráfica de las restricciones y función objetivo')
    plt.legend()
    plt.grid(True)
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url