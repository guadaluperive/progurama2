from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from simplex import simplex, plot_graph

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    c = np.array(list(map(float, request.form['c'].split())))
    num_restricciones = int(request.form['num_restricciones'])
    A = []
    b = []
    for i in range(num_restricciones):
        restriccion = list(map(float, request.form[f'restriccion_{i+1}'].split()))
        A.append(restriccion[:-1])
        b.append(restriccion[-1])
    A = np.array(A)
    b = np.array(b)
    
    optimal_value, solution = simplex(c, A, b)
    plot_url = plot_graph(c, A, b)

    return render_template('result.html', optimal_value=optimal_value, solution=solution, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
