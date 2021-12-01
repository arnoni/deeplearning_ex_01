import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sp

def derivative_tanh(x):
    print(f"last_update_2021_11_25_16_12 derivative_tanh")
    return 1.0 - np.tanh(x)**2

def jacobian_f(X,W,b):
    return np.tanh((W@X) +b)

def grad_g(X,W,b,u): # todo_2.2.1 gradient for 1 layer
    print(f"last_update_2021_11_25_16_12 derivative_tanh")
    layer_output = W@X + b
    db = np.sum(derivative_tanh(layer_output)* u, axis=1, keepdims=True )
    dw = (derivative_tanh(layer_output)* u) @ X.T
    dx = (W.T @ (derivative_tanh(layer_output)* u))
    return dx, dw, db

def do_jacobian_test_for_dense_layer():
    print(f"last_update_2021_12_01_15_31 do_jacobian_test_for_dense_layer")

    input_size =10
    output_size =12
    num_examples = 4

    W = np.random.randn(output_size, input_size)
    print(f"W shape {W.shape}")
    X = np.random.randn(input_size, num_examples)
    print(f"X shape {X.shape}")
    b = np.random.randn(output_size, 1)
    print(f"b shape {b.shape}")
    u = np.random.randn(output_size, num_examples)
    print(f"u shape {u.shape}")
    d = np.random.randn(input_size, num_examples)
    print(f"d shape {d.shape}")
    d = d/np.linalg.norm(d)
    epsilon = 0.1
    g0 = np.vdot(jacobian_f(X,W,b),u)
    print(f"g0 is {g0}")
    grad_X, grad_W, grad_b = grad_g(X,W,b,u)

    y0 = []
    y1 =[]
    for i in range(10):
        epsk = epsilon * (0.5**i)
        gk = np.vdot(jacobian_f(X+epsk*d,W,b),u)  # See: 2.10.4 The direct Jacobian transposed test
        #print(f"gk {gk}")
        F0 = gk-g0  # zero order approx
        Fk = F0 - np.vdot(grad_X, d*epsk)  #with grad_x # first order approx

        y0.append(abs(F0))
        y1.append(abs(Fk))

    plt.figure(figsize=(8,6))
    plt.semilogy(y0, label =r'$|F(X+\epsilon*d)-F(X)|$' )  # zero order approx
    plt.plot(y1, label=r'$|F(X+\epsilon*d)-F(x)-JacTMV(x,u)|$')  # first order approx
    plt.legend(loc="upper right")



