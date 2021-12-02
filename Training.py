import numpy as np
from Classifier import SoftmaxClassifier

# class Loss:
#
#     def __init__(self):
#         pass
#
#     def calc(self, output, y):
#         sample_losses = self.forward(output, y)
#         data_loss = np.mean(sample_losses)
#         return data_loss
#
#
# class Loss_CategoricalCrossEntropy(Loss):
#
#
#     def __init__(self):
#         pass
#
#     def forward(self, y_pred, y_true):
#         samples = len(y_pred)
#         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
#         correct_confidences = np.sum(y_pred_clipped * y_true.T, axis=1)
#         negative_log_likeliihood = -np.log(correct_confidences)
#         return negative_log_likeliihood


class Train:

    def __init__(self):
        self.input_dim = None
        self.number_of_classes = None
        self.error_list = []

    # def train_softmax_classifier_only(self, x, y, num_of_examples, dataset_type):
    #
    #     print(f"last_update_2021_11_24_14_16 train_softmax_classifier_only num_of_example {num_of_examples}")
    #     print(f"train_softmax_classifier_only y {y}")
    #     print(f"train_softmax_classifier_only y.T {y.T}")
    #     print(f"train_softmax_classifier_only cookie2")
    #     print(f"train_softmax_classifier_only X {x}")
    #     print(f"train_softmax_classifier_only X.T {x.T}")
    #     print(f"train_softmax_classifier_only x shape {x.shape}")
    #     print(f"train_softmax_classifier_only y shape {y.shape}")
    #     print(f"train_softmax_classifier_only type(y) {type(y)}")
    #     if dataset_type == 1:  # SwissRoll_dimensionality = 2, SwissRoll_number_of_classes = 2
    #         self.input_dim = 2
    #         self.number_of_classes = 2
    #         print(f"for SwissRoll Dataset")
    #     elif dataset_type == 2:  # GMM_dimensionality = 5, GMM_number_of_classes = 5
    #         self.input_dim = 5
    #         self.number_of_classes = 5
    #     else:
    #         print("ERROR! dataset type")
    #     W = 0.01 * np.random.randn(self.input_dim, self.number_of_classes)
    #     b = np.zeros((1, self.number_of_classes))
    #     print(f"W [before gradient fix] {W}")
    #     print(f"b [before gradient fix] {b}")
    #     # some hyperparameters
    #     step_size = 1e-0
    #     reg = 1e-3  # regularization strength
    #     X_T = np.transpose(x)
    #     # X = X_T
    #     softmax_classifier = SoftmaxClassifier()
    #     loss_function = Loss_CategoricalCrossEntropy()
    #     num_examples = num_of_examples
    #     for i in range(num_examples):
    #         print(f"do loop {i}")
    #         # evaluate class scores, [N x K]
    #         scores = np.dot(X_T, W) + b
    #         # compute the class probabilities
    #         probabilities = softmax_classifier.calcProbForClasses(scores)
    #         # compute the loss: average cross-entropy loss and regularization
    #
    #         data_loss = loss_function.calc(probabilities, y.astype(int))
    #         reg_loss = 0.5 * reg * np.sum(W * W)
    #         loss = data_loss + reg_loss
    #         # if i % 10 == 0:
    #         # print (f"iteration {i}: loss {loss}")
    #         print(f"after example# {i}: loss {loss}")
    #         # # compute the gradient on scores
    #         # dscores = probabilities
    #         # dscores[range(num_examples),y.astype(int)] -= 1
    #         # dscores /= num_examples
    #         #
    #         # # backprop the gradient to the parameters (W,b)
    #         # dW = np.dot(X.T, dscores)
    #         # db = np.sum(dscores, axis=0, keepdims=True)
    #         #
    #         # dW += reg*W # regularization gradient
    #         # perform a parameter update
    #         # W += -step_size * dW
    #         # b += -step_size * db
    #
    #         softmax_gradient_dw, softmax_gradient_db = softmax_classifier.calcGradient(X_T, y)
    #         print(f"softmax_gradient_dw shape: {softmax_gradient_dw.shape}")
    #         print(f"softmax_gradient_dw: {softmax_gradient_dw}")
    #         print(f"softmax_gradient_db shape: {softmax_gradient_db.shape}")
    #         print(f"softmax_gradient_db: {softmax_gradient_db}")
    #
    #         # perform a parameter update
    #         W += -step_size * softmax_gradient_dw
    #         b += -step_size * softmax_gradient_db
    #
    #         print(f"W [after gradient fix] {W}")
    #         print(f"b [after gradient fix] {b}")
    #
    # def FullGradientDescent(self, x, y):
    #     # Train a Linear Classifier
    #     SwissRoll_dimensionality = 2
    #     SwissRoll_number_of_classes = 2
    #     # initialize parameters randomly
    #     W = 0.01 * np.random.randn(SwissRoll_dimensionality, SwissRoll_number_of_classes)
    #     b = np.zeros((1, SwissRoll_number_of_classes))
    #
    #     # some hyperparameters
    #     step_size = 1e-0
    #     reg = 1e-3  # regularization strength
    #
    #     # gradient descent loop
    #     SwissRollTrainX_T = np.transpose(x)
    #     X = SwissRollTrainX_T
    #
    #
    #     softmax_classifier = SoftmaxClassifier()
    #     loss_function = Loss_CategoricalCrossEntropy()
    #
    #     num_examples = X.shape[0]
    #     for i in range(200):
    #
    #         # evaluate class scores, [N x K]
    #         scores = np.dot(X, W) + b
    #
    #         # compute the class probabilities
    #         # exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    #         # exp_scores = np.exp(scores)
    #         # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    #
    #         # compute the loss: average cross-entropy loss and regularization
    #         # correct_logprobs = -np.log(probs[range(num_examples),y.astype(int)])
    #         # data_loss = np.sum(correct_logprobs)/num_examples
    #         probabilities = softmax_classifier.calcProbForClasses(scores)
    #
    #         softmax_gradient = softmax_classifier.calcGradient(X, y)
    #         print(f"softmax_gradient: {softmax_gradient}")
    #
    #         data_loss = loss_function.calc(probabilities, y.astype(int))
    #
    #         reg_loss = 0.5 * reg * np.sum(W * W)
    #         loss = data_loss + reg_loss
    #         if i % 10 == 0:
    #             print(f"iteration {i}: loss {loss}")
    #
    #         # compute the gradient on scores
    #         dscores = probabilities
    #         dscores[range(num_examples), y.astype(int)] -= 1
    #         dscores /= num_examples
    #
    #         # backpropate the gradient to the parameters (W,b)
    #         dW = np.dot(X.T, dscores)
    #         db = np.sum(dscores, axis=0, keepdims=True)
    #
    #         dW += reg * W  # regularization gradient
    #
    #         # perform a parameter update
    #         W += -step_size * dW
    #         b += -step_size * db
    #
    # def StochasticGradientDescentVanilla(self, x, y):
    #     print(f"last_update_2021_11_24_10_14 StochasticGradientDescentVanilla")
    #
    # def StochasticGradientDescentMomentum(self, x, y):
    #     print(f"last_update_2021_11_24_10_14 StochasticGradientDescentMomentum")

    # function to create a list containing mini-batches
    def create_mini_batches(self, X, y, batch_size):
        print(f"last_update_2021_12_02_21_20 create_mini_batches")
        mini_batches = []
        print(f"X shape {X.shape}")
        print(f"y shape {y.shape}")
        print(f"batch_size {batch_size}")
        data = np.hstack((X, y.reshape(X.shape[0],1)))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0

        # for i in range(n_minibatches + 1):
        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def SGD(self, data_x, data_y, model, epochs, mini_batch_size, model_hyper_parameters, mode=1):
        print(f"last_update_2021_12_02_21_12 SGD")

        print(f"SGD:model name {model.name()}")

        lr_step_size = 1e-0
        reg = 1e-3  # regularization strength
        print(f"SGD:model model_hyper_parameters {model_hyper_parameters}")
        model.update_model_hyper_parameters(model_hyper_parameters)

        for j in range(epochs):
            mini_batches = self.create_mini_batches(data_x, data_y, mini_batch_size)
            print(f"CHECK_ME_004 mini_batches {mini_batches}")
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                print(f"CHECK_ME_003 X_mini shape{X_mini.shape}")
                print(f"y_mini shape{y_mini.shape}")
                #forward
                # For the forward we need to that X_mini shape will be X_examples laying horizontally!
                if X_mini.shape[0]>X_mini.shape[1]:
                    print(f"X_mini shape is with X_examples laying horizontally! - GOOD")
                model.forward(X_mini, y_mini)
                model_gradient = model.gradient()
                print(f"model_gradient = {model_gradient}")
                print(f"model_gradient shape{model_gradient.shape}")
                mini_batch_loss = model.loss()
                print(f"mini_batch_loss = {mini_batch_loss}")
                print(f"mini_batch_loss shape{mini_batch_loss.shape}")


                #backward

                #update model_hyper_parameters
                # perform a parameter update
                dW, db = np.hsplit(model_gradient, [model_gradient.shape[1]-1])
                print(f"CHECK_ME_001 model_hyper_parameters = {model_hyper_parameters}")
                print(f"model_hyper_parameters shape{model_hyper_parameters.shape}")
                W, b = np.hsplit(model_hyper_parameters, [model_hyper_parameters.shape[1]-1])

                print(f"dW = {dW}")
                print(f"dW shape{dW.shape}")
                print(f"db = {db}")
                print(f"db shape{db.shape}")

                print(f"W = {W}")
                print(f"W shape{W.shape}")
                print(f"b = {b}")
                print(f"b shape{b.shape}")

                W += -lr_step_size * dW
                b += -lr_step_size * db
                # model_hyper_parameters = model_hyper_parameters - lr_step_size * model.gradient(X_mini, y_mini, theta)
                model_hyper_parameters = grad = np.concatenate((W, b), axis=1)

                print(f"model_hyper_parameters = {model_hyper_parameters}")
                print(f"model_hyper_parameters shape{model_hyper_parameters.shape}")

                model.update_model_hyper_parameters(model_hyper_parameters)

                self.error_list.append(mini_batch_loss)



