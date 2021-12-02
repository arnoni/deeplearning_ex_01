# This is a sample Python script.
# from test_class import test1
# from test_class import test2
import matplotlib.pyplot as plt

from Classifier import SoftmaxClassifier
from FullyConnectedNeuralNetwork import last_layer
from Gradient_Test_for_a_single_layer import do_jacobian_test_for_dense_layer
from Gradient_test_for_softmax_layer import do_gradient_test_for_softmax_layer
from Utils import Utils
from Training import Train as Tr
# from FullyConnectedNeuralNetwork import DenseNet
# from ResidualNeuralNetwork import ResNet
# from Activations import ActivationReLU
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def do_TrainSoftmaxClassifierOnly(x, y, num_of_examples, dataset_type):
    print(f"last_update_2021_11_24_11_56 do_TrainSoftmaxClassifierOnly")
    t = Tr()
    t.train_softmax_classifier_only(x, y, num_of_examples, dataset_type)
    # t.train_softmax_classifier_only(1, 1, 1, 1)

# def setup_fully_connected_nn(num_of_layers_L,num_classes, input_dimensionality, num_of_neurons_per_hidden_layer_vec, activationFunction):
#     print(f"last_update_2021_11_23_21_26 setup_fully_connected_nn")
#     FC_NN = DenseNet()
#     FC_NN.CreateDenseNet(num_of_layers_L,num_classes, input_dimensionality, num_of_neurons_per_hidden_layer_vec, activationFunction)
#     return FC_NN
#
# def setup_ResNet_nn(num_of_layers_L,num_classes, input_dimensionality, num_of_neurons_per_hidden_layer_vec, activationFunction):
#     print(f"last_update_2021_11_23_21_26 setup_fully_connected_nn")
#     ResNet_NN = ResNet()
#     ResNet_NN.CreateResNet(num_of_layers_L,num_classes, input_dimensionality, num_of_neurons_per_hidden_layer_vec, activationFunction)
#     return ResNet_NN

def train_SwissRoll(x,y):
    print(f"last_update_2021_11_23_21_15 train_SwissRoll")
    train_SwissRoll = Tr()
    train_SwissRoll.FullGradientDescent(x.T,y)

def loadAllData():
    print(f"last_update_2021_11_23_21_15 loadAllData")
    util_Inst = Utils()
    # load_Data = util_Inst.loadData()
    # PeaksData = load_PeaksData.loadDataPeaksData("PeaksData.mat")
    PeaksData = util_Inst.loadData("PeaksData.mat")
    SwissRollData = util_Inst.loadData("SwissRollData.mat")
    GMMData = util_Inst.loadData("GMMData.mat")

    return PeaksData, SwissRollData, GMMData
    # PeaksData = load_PeaksData.loadDataPeaksData("PeaksData.mat")
    # PeaksDataTrainX = PeaksData[0]
    # PeaksDataTrainY_labels = PeaksData[1]
    # PeaksDataValidationX = PeaksData[2]
    # PeaksDataValidationY_labels = PeaksData[3]
# Press the green button in the gutter to run the script.





if __name__ == '__main__':

    print(f"Start of Ex1, last_update_2021_11_12_01_56")

    np.random.seed(0)

    print(f"1. Read: PeaksData, SwissRollData, GMMData")
    PeaksData, SwissRollData, GMMData = loadAllData()
    PeaksDataTrainX = PeaksData[0]
    PeaksDataTrainY_labels = PeaksData[1]
    PeaksDataValidationX = PeaksData[2]
    PeaksDataValidationY_labels = PeaksData[3]

    SwissRollTrainX = SwissRollData[0]
    SwissRollTrainY_labels = SwissRollData[1]
    SwissRollValidationX = SwissRollData[2]
    SwissRollValidationY_labels = SwissRollData[3]

    GMM_TrainX = GMMData[0]
    GMM_TrainY_labels = GMMData[1]
    GMM_ValidationX = GMMData[2]
    GMM_ValidationY_labels = GMMData[3]

    SwissRoll_dimensionality = 2
    SwissRoll_number_of_classes = 2
    num_of_layers_L = 3
    num_classes = SwissRoll_number_of_classes
    input_dimensionality = SwissRoll_dimensionality

    x = SwissRollTrainX[:, 0:2]
    y = SwissRollTrainY_labels[:, 0:2]
    print(f"main: x shape {x.shape}")
    print(f"main: y shape {y.shape}")
    print(f"main: y  {y}")
    print(f"main: y.T  {y.T}")
    print(f"main: type(y)  {type(y)}")


    # x = range(100)
    # y = range(100,200)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    #
    # ax1.scatter(x[:4], y[:4], s=10, c='b', marker="s", label='first')
    # ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
    # plt.legend(loc='upper left');
    # plt.show()

    # y0_fake = [10,100,100,1000,10000]
    # y1_fake = [40,400,400,4000,40000]
    # x_axis = np.arange(1,9,1)
    # x_axis_fake = np.arange(1,6,1)
    # # print(f"x_axis shape = {x_axis.shape}")
    # plt.figure(figsize=(8, 6))
    # # plt.semilogy(x_axis,y0)
    # # plt.semilogy(x_axis,y1)
    # plt.semilogy(y0_fake, label=r"Zero order approx")
    # plt.plot(y1_fake, label=r"First order approx")
    # # plt.semilogy(x_axis_fake,y0_fake)
    # # plt.semilogy(x_axis_fake,y1_fake)
    # # plt.legend(("Zero order approx","First order approx"))
    # # plt.title("Successful Grad test in semilogarithmic plot")
    # # plt.xlabel("k");
    # # plt.ylabel("error");
    # plt.show()


    # Gradient test for softmax layer:
    print(f"main: 2.1.1 Gradient test for softmax layer and plot graph: - DEV_PHASE: CLEAN_UP")
    #do_gradient_test_for_softmax_layer()
    # Run SGD on a small least squares example:
    print(f"main: 2.1.2 run SGD on softmax layer for a small least squares example and plot graph:")
    util_Inst2 = Utils()
    ls_dataset_x, ls_dataset_y = util_Inst2.generate_LS_data_simple(200, 2, 0.5)
    ls_dataset_train_x, ls_dataset_test_x = np.split(ls_dataset_x, [180])
    ls_dataset_train_y, ls_dataset_test_y = np.split(ls_dataset_y, [180])

    trainer_LS = Tr()
    ls_dataset_num_classes = 2
    inputs_neurons_to_last_layer = 2
    model_softmax = last_layer("softmax",inputs_neurons_to_last_layer,ls_dataset_num_classes)

    # W = 0.10 * np.random.randn(n_inputs, num_classes_output)
    # b = np.random.randn(num_classes_output, 1)

    W = 0.10 * np.random.randn(inputs_neurons_to_last_layer, ls_dataset_num_classes)
    print(f"main - check shape W {W.shape}")
    print(f"main - check type W {type(W)}")
    b = np.random.randn(ls_dataset_num_classes, 1)
    print(f"main - check shape b {b.shape}")
    print(f"main - check type b {type(b)}")
    # model_hyper_parameters = np.empty()
    model_hyper_parameters = np.concatenate((W,b),axis=1)
 # np.concatenate(W,b.T])
    trainer_LS.SGD(ls_dataset_train_x,ls_dataset_train_y, model_softmax, epochs=2, mini_batch_size=10, model_hyper_parameters =model_hyper_parameters, mode=1)

# Run SGD on softmax layer for Swissroll and GMM datasets:
    print(f"main: 2.1.3 run SGD on softmax layer for Swissroll and GMM datasets with various configuration and plot best graph:")

    # Construct a FC NN:
    print(f"main: 2.2.1 1. Construct a FC NN:")
    # Jacobian test for FC layer:
    print(f"main: 2.2.1 2. Jacobian test for 1 FC layer and plot graph:")
    #do_jacobian_test_for_dense_layer()
    # Construct a ResNet NN:
    print(f"main: 2.2.2 1. Construct a ResNet NN:")
    # Jacobian test for ResNet layer:
    print(f"main: 2.2.2 2. Jacobian test for 1 ResNet layer and plot graph:")

    # Gradient test for a whole FC NN and ResNet NN:
    print(f"main: 2.2.3 Jacobian test for the whole FC NN and ResNet NN and plot graph:")

    # Run multiple varients of SGDs on different FC NN and ResNet NN configurations for Swissroll and GMM datasets:
    print(f"main: 2.2.4 1. Run multiple varients of SGDs on different FC NN and ResNet NN configurations for Swissroll and GMM datasets and plot best graph:")
    print(f"main: 2.2.4 2. write conclusions :")

    # Run multiple varients of SGDs on different FC NN and ResNet NN configurations for Swissroll and GMM datasets - a training size of 200 examples only
    print(f"main: 2.2.5 Run multiple varients of SGDs on different FC NN and ResNet NN configurations for Swissroll and GMM datasets and plot best graph - a training size of 200 examples only")
    print(f"main: 2.2.5 write conclusions :")






















    # trrr = Tr()
    # trrr.train_softmax_classifier_only(1,1,1,1)




    # do_TrainSoftmaxClassifierOnly(x, y, 2, 1)
    #
    # num_of_neurons_per_hidden_layer_vec = 3
    # activationFunction = ActivationReLU()
    # SwissRoll_NN_01 = setup_fully_connected_nn(num_of_layers_L,num_classes, input_dimensionality, num_of_neurons_per_hidden_layer_vec, activationFunction)
    # train_SwissRoll(SwissRollTrainX,SwissRollTrainY_labels, SwissRoll_NN_01)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/


