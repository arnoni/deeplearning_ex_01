import numpy as np


class LayerDesne:
    def __init__(self, name, n_inputs, n_neurons, activation_type):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.layer_name = name
        if activation_type == 1:  #ReLU
            self.activation = self.activationReLU
        else:  # tanh
            self.activation = self.activation_tanh
        self.V = None  # after multiplying with W and adding
        self.X = None  # input to the layer: output from prev layer/original input
        self.Z = None  # value after activation
        self.nn_gradient = None

    def print_gradient(self):
        print(f" NN gradient: {self.nn_gradient}")
        print(f" NN gradient shape: {self.nn_gradient.shape}")

    def __repr__(self):
        return f"{self.layer_name}"

    def forward(self, X):
        self.X = X  # need for gradient calc later
        print(f"last_update_2021_11_30_18_44 forward")

        self.V = self.X @ self.weights + self.biases
        self.Z = self.activation(self.V)
        print(f"forward: self.Z shape {self.Z} ")
        return  self.Z

    def backward(self, prev_dx):
        print(f"last_update_2021_11_28_10_26 backward")
        #  V/grad_input
        #  grad
        # V = ??? see lecture: https://moodle2.bgu.ac.il/moodle/local/kalturamediagallery/index.php?courseid=38690 at 01:12
        # 1. calc & save gradient:
        V = (self.grad_wrt_w * prev_dx)  # self.grad of the activation
        self.dx = self.w.T @ V  # to be use as prev_dx on the prev layer (keeping the things from left-to-right orientation) ARNON_GOOD
        self.dw = V @ self.X.T  # used to calc update weights later in the update-phase (when backprob is complete) ARNON_GOOD
        self.db = V.sum(keepdim=True,
                        axis=1)  # used to calc update weights later in the update-phase (when backprob is complete)
        self.nn_gradient.append(self.dw)  # accumulate gradient from each layer for the NN gradient test! for todo_2.2.3
        self.print_gradient()
        return self.dx  # this will be received as prev_dx # todo_2.2.1_1 for backprob
        # def calcGradient(self, input_to_output_layer, y_true , ):
        # #from: https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
        # #from: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1299s
        # # self.dZ2 = A2 - one_hot_Y
        # # self.dW2 = 1 / m * dZ2.dot(A1.T)
        # print(f"calcGradient: probabilities shape {self.probabilities.T.shape} should be 2X20000")
        # print(f"calcGradient: y_true shape {y_true.shape} should be 2X20000")
        # print(f"calcGradient: input_to_output_layer shape {input_to_output_layer.shape} should be 20000X2")
        # m = input_to_output_layer.shape[0]
        # print(f"calcGradient: m {m} should be 20000")
        #
        # self.dZ2 = self.probabilities.T - y_true
        # self.dW2 = (1 / m) * self.dZ2.dot(input_to_output_layer)
        # return self.dW2

    def calc_layer_jacobian(self, X,W,b,u):  # todo_2.2.1_1 for backprob
        print(f"last_update_2021_11_28_10_30 calc_layer_jacobian")
        layer_output = W@X + b
        db = np.sum(self.derivative_of_activationReLU(layer_output)* u, axis=1, keepdims=True )
        dw = (self.derivative_of_activationReLU(layer_output)* u) @ X.T
        dx = (W.T @ (self.derivative_of_activationReLU(layer_output)* u))
        return dx, dw, db

    def activationReLU(self, input):
        print(f"last_update_2021_11_28_10_30 activationReLU - LEARN_ME")
        return np.maximum(0, input)

    def derivative_of_activationReLU(self, input):
        print(f"last_update_2021_11_28_10_37 derivative_of_activationReLU - LEARN_ME")
        der_ReLU = (input > 0)*1.0
        print(f"derivative_of_activationReLU - der_ReLU shape {der_ReLU.shape}")
        print(f"derivative_of_activationReLU - der_ReLU {der_ReLU}")
        return der_ReLU
        #return np.maximum(0, input)

    def derivative_of_tanh(self, input):
        print(f"last_update_2021_11_28_10_40 derivative_of_tanh - LEARN_ME")
        return 1.0 - np.tanh(input)**2

    def activation_tanh(self, input):
        print(f"last_update_2021_11_28_10_40 activation_tanh")
        return np.tanh(input)


class last_layer:

    def __init__(self, name, n_inputs, num_classes_output):
        self.W = 0.10 * np.random.randn(n_inputs, num_classes_output)
        self.b = np.random.randn(num_classes_output, 1)
        self.layer_name = name
        # self.activation = activation
        self.V = None  # after multiplying with W and adding
        self.X = None  # input to the layer: output from prev layer/original input
        self.Z = None  # value after activation

    def __repr__(self):
        return f"{self.layer_name}"

    def name(self):
        return self.layer_name

    def forward(self, input_X, y_true_in=None):
        print(f"forward: input_X shape {input_X} ")
        if y_true_in.all() is not None:
                self.y_true = y_true_in

        self.X = input_X  # need for gradient calc later

        print(f"last_update_2021_11_28_16_09 forward")
        # self.V = self.X @ self.W + self.b
        # self.Z = self.activation(self.V)
        print(f"forward: self.X shape {self.X.shape} ")
        print(f"forward: self.W shape {self.W.shape} ")
        print(f"forward: self.b shape {self.b.shape} ")
        print(f"forward: (self.X @ self.W) shape {(self.X @ self.W).shape} ")
        # if input_X input is already X_examples laying horizontally we need self.X else we need self.X.T
        self.V = (self.X @ self.W).T + self.b
        # self.V = (self.X.T @ self.W).T + self.b
        print(f"forward: before Softmax - self.V {self.V} ")
        self.V_T = self.V.T
        # return self.Z

    def backprob(self, input_to_Softmax, y_true):
        print(f"last_update_2021_11_25_19_20 backward w input_to_Softmax {input_to_Softmax}")
        # from: https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
        # from: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1299s
        # self.dZ2 = A2 - one_hot_Y
        # self.dW2 = 1 / m * dZ2.dot(A1.T)
        print(f"calcGradient: probabilities shape {self.probabilities.T.shape} should be 2X20000")
        print(f"calcGradient: y_true shape {y_true.shape} should be 2X20000")
        print(f"calcGradient: input_to_Softmax shape {input_to_Softmax.shape} should be 20000X2")
        m = input_to_Softmax.shape[0]
        print(f"calcGradient: m {m} should be 20000")

        self.dZ = self.probabilities.T - y_true
        self.dW = self.grad_softmax_wrt_w(m)  # (1 / m) * self.dZ2.dot(input_to_Softmax)
        self.db = (1 / m) * self.dZ
        self.dx = self.grad_softmax_wrt_x(m, input_to_Softmax)
        return self.dW, self.db

    def gradient(self):
        db = self.derivative_softmax_wrt_b(self.X, self.W, self.b, self.y_true.T)
        print(f"gradient - db = {db}")
        dW = self.derivative_softmax_wrt_w(self.X, self.W, self.b, self.y_true.T)
        print(f"gradient - dW = {dW}")
        grad = np.concatenate((dW,db),axis=1)
        return grad

    def update_model_hyper_parameters(self, model_hyper_parameters):
        if model_hyper_parameters.all() is not None:
            print(f"last_update_2021_12_02_19_33 update_model_hyper_parameters ")
            W, b = np.hsplit(model_hyper_parameters, [model_hyper_parameters.shape[1]-1])
            print(f"CHECK_ME_002 update_model_hyper_parameters check shape W shape {W.shape}")
            print(f"update_model_hyper_parameters check shape b shape {b.shape}")
            print(f"update_model_hyper_parameters check shape b.T shape {b.T.shape}")
            self.W = W
            self.b = b

    def derivative_softmax_wrt_w(self, X, W, b, c):
        print(f"last_update_2021_12_02_20_44 derivative_softmax_wrt_w ")
        print("derivative_softmax_wrt_w - shapes:")
        print(f"X:{X.shape}")
        print(f"X.T:{X.T.shape}")
        print(f"W:{W.shape}")
        print(f"b:{b.shape}")
        print(f"c:{c.shape}")
        m = c.shape[1]
        print(f"m = {m}")
        # z = softmax_with_CE_loss_no_mean(X, W, b, c)
        print(f"(X @ W) shape {(X @ W).shape}")
        V = (X @ W).T + b        # we need to that X shape will be X_examples laying horizontally!
        # V = (X.T @ W).T + b
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
        print(f"X.T:{X.T.shape}")
        return (1 / m) * X.T @ z_c.T
        # return (1 / m) * self.W @ self.dZ

    # def grad_softmax_wrt_x(self, m, x):
    #     print(f"last_update_2021_11_25_20_39 grad_softmax_wrt_x ")
    #     return (1 / m) * self.dZ.dot(x)

    def derivative_softmax_wrt_x(self, X, W, b, c):
        print("derivative_softmax_wrt_x - shapes:")
        print(f"X:{X.shape}")
        print(f"X.T:{X.T.shape}")
        print(f"W:{W.shape}")
        print(f"b:{b.shape}")
        print(f"c:{c.shape}")
        m = c.shape[1]
        print(f"m = {m}")
        # z = softmax_with_CE_loss_no_mean(X, W, b, c)
        print(f"(X @ W) shape {(X @ W).shape}")
        V = (X @ W).T + b        # we need to that X shape will be X_examples laying horizontally!
        # V = (X.T @ W).T + b
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

    def derivative_softmax_wrt_b(self, X, W, b, c):
        print("derivative_softmax_wrt_b - shapes:")
        print(f"X:{X.shape}")
        print(f"X.T:{X.T.shape}")
        print(f"W:{W.shape}")
        print(f"b:{b.shape}")
        print(f"c:{c.shape}")
        m = c.shape[1]
        print(f"m = {m}")
        # z = softmax_with_CE_loss_no_mean(X, W, b, c)
        # V = (X.T @ W).T + b
        print(f"(X @ W) shape {(X @ W).shape}")
        V = (X @ W).T + b   # we need to that X shape will be X_examples laying horizontally!
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
        z_c_average = (1 / m) * z_c
        print(f"z_c_average shape {z_c_average.shape}")
        print(f"sum axis=1 z_c_average shape {np.sum(z_c_average, axis=1, keepdims=True)}")
        print(f"sum axis=0 z_c_average shape {np.sum(z_c_average, axis=0, keepdims=True)}")
        return np.sum(z_c_average, axis=1, keepdims=True)

    def loss(self, y_true = None):
        if y_true:
            self.y_true = y_true

        print(f"loss - self.y_true = {self.y_true}")
        print(f"loss - self.V_T = {self.V_T}")
        print(f"loss - np.max(self.V_T, axis=1, keepdims=True) = {np.max(self.V_T, axis=1, keepdims=True)}")
        self.exp_values = np.exp(self.V_T - np.max(self.V_T, axis=1, keepdims=True))
        print(f"loss - self.exp_values = {self.exp_values}")
        # print(f"exp_values:{exp_values.shape}")
        y_pred = self.exp_values / np.sum(self.exp_values, axis=1, keepdims=True)
        print(f"loss - y_pred = {y_pred}")
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        print(f"loss - y_pred_clipped = {y_pred_clipped}")
        print(f"loss - self.y_true.shape {self.y_true.shape}")
        if self.y_true.shape[1] == 1:
            print(f"range(self.y_true.shape[0]) {range(self.y_true.shape[0])}")
            print(f"CHECK_ME_005 self.y_true.astype(int).ravel() {self.y_true.astype(int).ravel()}")
            my_range = np.arange(self.y_true.shape[0])
            labels_target = self.y_true.astype(int).ravel()
            range_list = my_range.tolist()
            labels_target_list = labels_target.tolist()
            print(f" range {range_list}")
            print(f"labels_target_list {labels_target_list}")
            # correct_confidences = y_pred_clipped[:,self.y_true.astype(int).ravel()]
            # correct_confidences = np.array([y_pred_clipped[0,0], y_pred_clipped[1,1],  y_pred_clipped[2,0]])  #2021_12_02_fix_me
            correct_confidences = y_pred_clipped[range_list,labels_target_list]
        else:
            correct_confidences = np.sum(y_pred_clipped * self.y_true, axis=1)
        print(f"CHECK_ME_005 loss - correct_confidences = {correct_confidences}")
        negative_log_likeliihood = -np.log(correct_confidences)
        return np.mean(negative_log_likeliihood)


class DenseNet:
    def __init__(self):
        self.num_of_layers_L = None
        self.num_classes = None
        self.num_of_neurons_per_hidden_layer_vec = None
        self.input_dimensionality = None
        self.denseLayers = []
        self.activation_functions = None

        # model = dict(
        #     W1=np.random.randn(n_feature, n_hidden),
        #     W2=np.random.randn(n_hidden, n_class)
        # )

    def CreateDenseNet(self, num_of_layers_L, num_classes, input_dimensionality, num_of_neurons_per_hidden_layer_vec,
                       activationFunction):  #  todo: add a struct to NN conf input instead of multiple args
        self.num_of_layers_L = num_of_layers_L
        self.num_classes = num_classes
        self.num_of_neurons_per_hidden_layer_vec = num_of_neurons_per_hidden_layer_vec
        self.input_dimensionality = input_dimensionality

        dense_layer_first = LayerDesne("layer_0", self.input_dimensionality,
                                       self.num_of_neurons_per_hidden_layer_vec[0])
        print(f"Adding dense_layer_first {dense_layer_first}")

        num_neaurons_in = self.num_of_neurons_per_hidden_layer_vec[0]
        self.denseLayers.append(dense_layer_first)
        self.activation_functions.append(activationFunction)
        print(f"Adding activationFunction  {activationFunction}")
        for i in self.num_of_layers_L:
            dense_layer = LayerDesne(f"layer_{i}", num_neaurons_in, self.num_of_neurons_per_hidden_layer_vec[i])
            print(f"Adding dense_layer {dense_layer}")
            num_neaurons_in = self.num_of_neurons_per_hidden_layer_vec[i]
            self.denseLayers.append(dense_layer)
            self.activation_functions.append(activationFunction)
            print(f"Adding activationFunction_{i} {activationFunction}")

    def forward(self, examples_X):
        print(f"last_update_2021_11_24_12_03 forward")
        self.denseLayers[0].forward(examples_X)
        self.activation_functions[0].forward(self.denseLayers[0].output)
        for idx in range(len(self.denseLayers)):
            self.denseLayers[idx].forward(self.activation_functions[idx - 1].output)
            self.activation_functions[idx].forward(self.denseLayers[idx].output)

    def backprob(self):
        print(f"last_update_2021_11_24_12_03 backprob")
