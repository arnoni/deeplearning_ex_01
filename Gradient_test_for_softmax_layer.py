import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sp

from Utils import Utils


def get_X_swissroll(num_examples):
    print(f"last_update_2021_11_25_21_11 get_X_swissroll")
    util_Inst = Utils()
    SwissRollData = util_Inst.loadData("SwissRollData.mat")
    SwissRollTrainX = SwissRollData[0]
    return SwissRollTrainX[:, 0:num_examples]


def get_Y_swissroll(num_examples):
    print(f"last_update_2021_11_25_21_11 get_X_swissroll")
    util_Inst = Utils()
    SwissRollData = util_Inst.loadData("SwissRollData.mat")
    # SwissRollTrainX = SwissRollData[0]
    SwissRollTrainY_labels = SwissRollData[1]
    return SwissRollTrainY_labels[:, 0:num_examples]


def derivative_softmax_wrt_w(X, W, b, c): # need this for the gradient test, goes into backprob
    print(f"last_update_2021_12_01_15_16 derivative_softmax_wrt_w")
    print(f"derivative_softmax_wrt_w: shape_check: W shape {W.shape}")
    print(f"derivative_softmax_wrt_w: shape_check: X shape {X.shape}")
    print(f"derivative_softmax_wrt_w: shape_check: b shape {b.shape}")
    print(f"derivative_softmax_wrt_w: shape_check: c shape {c.shape}")
    m = c.shape[1]
    # z = softmax_f2(X, W)
    z = softmax_with_CE_loss_no_mean(X, W, b, c)
    print(f"derivative_softmax_wrt_w: shape_check: z shape {z.shape}")
    return (1 / m) * X.T @ (z - c)


def derivative_softmax_wrt_x(X, W, b, c):
    print(f"last_update_2021_12_01_15_1 derivative_softmax_wrt_x")
    print(f"derivative_softmax_wrt_x: shape_check: W shape {W.shape}")
    print(f"derivative_softmax_wrt_x: shape_check: X shape {X.shape}")
    print(f"derivative_softmax_wrt_x: shape_check: b shape {b.shape}")
    print(f"derivative_softmax_wrt_x: shape_check: c shape {c.shape}")

    m = c.shape[1]
    print(f"derivative_softmax_wrt_x: m num of examples = {m}")
    z = softmax_with_CE_loss_no_mean(X, W, b, c)
    return (1 / m) * W.T @ (z - c)

    # layer_output = W@X + b
    # db = np.sum(derivative_tanh(layer_output)* u, axis=1, keepdims=True )
    # dw = (derivative_tanh(layer_output)* u) @ X.T
    # dx = (W.T @ (derivative_tanh(layer_output)* u))

    # return 1.0 - np.tanh(x)**2

def softmax_f2(X, W):
    print(f"last_update_2021_11_29_12_25 softmax_f2")
    V = X.T @ W
    print(f"softmax_f: shape_check: V shape {V.shape}")
    exp_values = np.exp(V - np.max(V, axis=1, keepdims=True))
    softmax_ret = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    print(f"softmax_f2: shape_check: softmax_ret shape {softmax_ret.shape}")
    return softmax_ret



def softmax_with_CE_loss_w_mean(X, W, b, y_true):
    print(f"last_update_2021_12_01_15_15 softmax_with_CE_loss")
    V = (X.T @ W) + b
    print(f"softmax_with_CE_loss: shape_check: V shape {V.shape}")
    # b_T = b.T
    # V2 = V[:,]+b_T
    # print(f"softmax_f: shape_check: V2 shape {V2.shape}")
    exp_values = np.exp(V - np.max(V, axis=1, keepdims=True))
    print(f"softmax_with_CE_loss: shape_check: exp_values shape {exp_values.shape}")
    # return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    y_pred = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    correct_confidences = np.sum(y_pred_clipped * y_true.T, axis=1)
    print(f"softmax_with_CE_loss: shape_check: correct_confidences shape {correct_confidences.shape}")
    negative_log_likeliihood = -np.log(correct_confidences)
    print(f"softmax_with_CE_loss: shape_check: negative_log_likeliihood shape {negative_log_likeliihood.shape}")
    return np.mean(negative_log_likeliihood)

def softmax_with_CE_loss_no_mean(X, W, b, y_true):
    print(f"last_update_2021_12_01_15_15 softmax_with_CE_loss")
    V = (X.T @ W) + b
    print(f"softmax_with_CE_loss: shape_check: V shape {V.shape}")
    # b_T = b.T
    # V2 = V[:,]+b_T
    # print(f"softmax_f: shape_check: V2 shape {V2.shape}")
    exp_values = np.exp(V - np.max(V, axis=1, keepdims=True))
    print(f"softmax_with_CE_loss: shape_check: exp_values shape {exp_values.shape}")
    # return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    y_pred = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    correct_confidences = np.sum(y_pred_clipped * y_true.T, axis=1)
    print(f"softmax_with_CE_loss: shape_check: correct_confidences shape {correct_confidences.shape}")
    negative_log_likeliihood = -np.log(correct_confidences)
    print(f"softmax_with_CE_loss: shape_check: negative_log_likeliihood shape {negative_log_likeliihood.shape}")
    return negative_log_likeliihood
    # return np.mean(negative_log_likeliihood)

    # Z = self.activation(self.v)
    # return self.Z
    # return np.tanh((W@X) +b)


# def grad_g(X, W, b, u):  # todo_2.2.1 gradient for 1 layer
#     print(f"last_update_2021_11_25_16_12 derivative_tanh")
#     layer_output = W @ X + b
#     db = np.sum(derivative_tanh(layer_output) * u, axis=1, keepdims=True)
#     dw = (derivative_tanh(layer_output) * u) @ X.T
#     dx = (W.T @ (derivative_tanh(layer_output) * u))
#     return dx, dw, db

def do_gradient_test_for_softmax_layer():
    print(f"last_update_2021_12_01_15_19 do_gradient_test_for_softmax_layer")
    input_size = 2  # we give x with dim 2 at the input
    output_size = 2  # two class at the last layer
    num_examples = 2

    W = np.random.randn(output_size, input_size)
    print(f"W shape {W.shape}")
    #X = np.random.randn(input_size, num_examples)
    #print(f"X shape {X.shape}")
    b = np.random.randn(output_size, 1)
    print(f"b shape {b.shape}")
    # u = np.random.randn(output_size, num_examples)
    # print(f"u shape {u.shape}")
    c = get_Y_swissroll(num_examples)
    print(f"c (swissroll 1-Hot labels) shape {c.shape}")
    X = get_X_swissroll(num_examples)
    print(f"X (swissroll) shape {X.shape}")
    d = np.random.randn(input_size, num_examples)
    print(f"d shape {d.shape}")
    d = d / np.linalg.norm(d)
    epsilon = 0.1
    # g0 = np.vdot(softmax_f(X, W, b), u)
    #g0 = np.vdot(softmax_f(X, W, b))
    # g0 = np.vdot(softmax_f(X, W, b, c))
    # g0 = softmax_f2(X, W, b, c)
    # g0 = softmax_f2(X, W)
    # print(f"np.vdot(softmax_f return: shape_check: g0 shape {g0.shape}")
    # print(f"g0 is {g0}")
    # F0 = softmax_f2(X, W)
    F0 = softmax_with_CE_loss_w_mean(X, W, b, c)  # loss function evaluated at X

    print(f"np.vdot(softmax_f return: shape_check: F0 shape {F0.shape}")
    print(f"F0 is {F0}")
    # grad_X, grad_W, grad_b = grad_g(X,W,b,u)
    # grad_W = derivative_softmax_wrt_w(X, W, b, c)
    # print(f"derivative_softmax_wrt_w return: shape_check: grad_W shape {grad_W.shape}")
    grad_X = derivative_softmax_wrt_x(X, W, b, c)
    print(f"derivative_softmax_wrt_x return: shape_check: grad_W shape {grad_X.shape}")

    y0 = []
    y1 = []
    for i in range(8):
        epsk = epsilon * (0.5 ** i)  # added noise
        # gk = np.vdot(softmax_f(X+epsk*d,W,b),u)
        # SM = softmax_f(X + epsk * d, W, b, c)
        # print(f"softmax_f return: shape_check: softmax_f shape {SM.shape}")
        # # gk = np.vdot(SM)
        # gk = SM
        # Fk = F(x+epsk*d); Eran's code
        # Fk = softmax_f2(X + epsk * d, W)
        Fk = softmax_with_CE_loss_w_mean(X + epsk * d, W, b, c)  # loss function evaluated at X + noise

        print(f"softmax_with_CE_loss return: shape_check: Fk shape {Fk.shape}")
        print(f"softmax_with_CE_loss return: Fk  {Fk}")

        # F0 = gk - g0  #  Fk_minus_F0
        Fk_minus_F0 = Fk - F0  #  Fk_minus_F0
        print(f"softmax_with_CE_loss return: shape_check: Fk_minus_F0 shape {Fk_minus_F0.shape}")
        print(f"softmax_with_CE_loss return:  Fk_minus_F0  {Fk_minus_F0}")
        # Fk = F0 - np.vdot(grad_X, d*epsk)
        # Fk = F0 - np.vdot(grad_W, d * epsk)  # Fk_minus_F1
        # F1 = F0 + epsk*dot(g0,d); Eran's code
        # F1 = F0 + np.vdot(grad_W, d * epsk)  # loss function evaluated at X + gradient of loss function wrt w
        F1 = F0 + np.vdot(grad_X, d * epsk)  # loss function evaluated at X + gradient of loss function wrt x
        print(f"softmax_with_CE_loss return: shape_check: F1 shape {F1.shape}")
        print(f"softmax_with_CE_loss return:  F1  {F1}")
        Fk_minus_F1 = Fk - F1  # Fk_minus_F1
        print(f"softmax_with_CE_loss return: shape_check: Fk_minus_F1 shape {Fk_minus_F1.shape}")
        print(f"softmax_with_CE_loss return:  Fk_minus_F1  {Fk_minus_F1}")



        # print(f": shape_check: Fk shape {Fk.shape}")

        y0.append(abs(Fk_minus_F0))  #Zero order approx
        y1.append(abs(Fk_minus_F1))

    print('matplotlib: {}'.format(matplotlib.__version__))

    # y0_fake = [10,100,100,1000,10000]
    # y1_fake = [40,400,400,4000,40000]
    x_axis = np.arange(1,9,1)
    x_axis_fake = np.arange(1,6,1)
    # print(f"x_axis shape = {x_axis.shape}")
    plt.figure(figsize=(8, 6))
    plt.semilogy(x_axis,y0)
    plt.semilogy(x_axis,y1)
    # plt.semilogy(y0_fake, label=r"Zero order approx")
    # plt.plot(y1_fake, label=r"First order approx")
    # plt.semilogy(x_axis_fake,y0_fake)
    # plt.semilogy(x_axis_fake,y1_fake)
    # plt.legend(("Zero order approx","First order approx"))
    # plt.title("Successful Grad test in semilogarithmic plot")
    # plt.xlabel("k");
    # plt.ylabel("error");
    plt.show()

    #
    # print("test_111")
    # plt.figure(figsize=(8, 6))
    # print("test_2222")
    # plt.semilogy(y0, label='Zero order approx')
    # print("test_3333")
    # plt.plot(y1, label='First order approx')
    # print("test_444")
    # #plt.legend(loc="upper right")
    # plt.semilogy(x_axis,y1);
    # plt.legend(("Zero order approx","First order approx"));
    # plt.title("Successful Grad test in semilogarithmic plot");
    # plt.xlabel("k");
    # plt.ylabel("error");
    # plt.show()
    # print("test_555")
