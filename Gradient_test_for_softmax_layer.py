import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from Utils import Utils

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.T

def derivative_softmax_wrt_x_fix(X, W, b, c):
    print("shapes:")
    print(f"X:{X.shape}")
    print(f"X.T:{X.T.shape}")
    print(f"W:{W.shape}")
    print(f"b:{b.shape}")
    print(f"c:{c.shape}")
    m = c.shape[1]
    print(f"m = {m}")
    # z = softmax_with_CE_loss_no_mean(X, W, b, c)
    V = (X.T @ W).T + b
    print(f"V:{V.shape}")
    V_T = V.T
    # V = (X.T @ W) + b
    # exp_values = np.exp(V - np.max(V, axis=1, keepdims=True))
    exp_values = np.exp(V_T - np.max(V_T, axis=1, keepdims=True))
    print(f"exp_values:{exp_values.shape}")
    y_pred = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    Z = np.clip(y_pred, 1e-7, 1 - 1e-7)
    Z_T = Z.T
    print(f"Z_T={Z_T}")
    print(f"c={c}")
    print(f"W={W}")
    print(f"Z_T:{Z_T.shape}")
    z_c = (Z_T - c)
    print(f"z_c:{z_c.shape}")
    return (1 / m) * W @ z_c

def softmax_with_CE_loss_w_mean(X, W, b, y_true):
    # print("shapes:")
    # print(f"X:{X.shape}")
    # print(f"X.T:{X.T.shape}")
    # print(f"W:{W.shape}")
    # print(f"b:{b.shape}")
    # print(f"y_true:{y_true.shape}")

    V = (X.T @ W).T + b
    # print(f"V:{V.shape}")
    V_T = V.T
    # print(f"V_T:{V_T.shape}")
    # exp_values = np.exp(V - np.max(V, axis=1, keepdims=True))
    exp_values = np.exp(V_T - np.max(V_T, axis=1, keepdims=True))
    print(f"exp_values:{exp_values.shape}")
    y_pred = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    correct_confidences = np.sum(y_pred_clipped * y_true.T, axis=1)
    negative_log_likeliihood = -np.log(correct_confidences)
    return np.mean(negative_log_likeliihood)

def do_gradient_test_for_softmax_layer():

    #Note:  intentionally using numbers: 2,3,5 to be able to debug the matrices shape easily
    input_size = 3  # we 3 neaurons at the input
    output_size = 2  # two class at the last layer
    num_examples = 5  # 5 examples

    W = np.random.randn(input_size, output_size)

    b = np.random.randn(output_size, 1)

    # c = get_one_hot([[0,1]],2))  # get_Y_swissroll(num_examples)
    c = get_one_hot([[0,1,0,0,1]],output_size)  # get_Y_swissroll(num_examples)

    X = np.random.randn(input_size, num_examples) # x_i are standing vectors

    d = np.random.randn(input_size, num_examples)

    d = d / np.linalg.norm(d)
    epsilon = 0.1

    F0 = softmax_with_CE_loss_w_mean(X, W, b, c)  # loss function evaluated at X


    # grad_X = derivative_softmax_wrt_x(X, W, b, c)  # gradient wrt X
    grad_X = derivative_softmax_wrt_x_fix(X, W, b, c)  # gradient wrt X

    y0 = []
    y1 = []
    for i in range(8):
        epsk = epsilon * (0.5 ** i)  # added noise

        Fk = softmax_with_CE_loss_w_mean(X + epsk * d, W, b, c)  # loss function evaluated at X + noise

        Fk_minus_F0 = Fk - F0  #  Fk_minus_F0

        F1 = F0 + np.vdot(grad_X, d * epsk)  # loss function evaluated at X + gradient of loss function wrt x

        Fk_minus_F1 = Fk - F1  # Fk_minus_F1

        y0.append(abs(Fk_minus_F0))   # Zero order approx
        y1.append(abs(Fk_minus_F1))   # First order approx


    x_axis = np.arange(1,9,1)
    plt.figure(figsize=(8, 6))
    plt.semilogy(x_axis,y0)
    plt.semilogy(x_axis,y1)
    plt.show()
