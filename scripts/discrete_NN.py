"""
Code for the article 'Multi-Objective Linear Ensembles for Robust and Sparse Training of Few-Bit Neural Networks'
by A. M. Bernardelli, S. Gualandi, H. C. Lau, S. Milanesi, N. Yorke-Smith


The PARAMETERS can be modified accordingly to the experiments

For every value in the list "different_p", the file produces 3*n*m + 1 .csv files
where n is the number of instances, namely "instances", and m is the number of different numbers of
training images, namely the length of the list "training_images"
Fixed the training images i and the instance j, the file produces
- labels_i_j.csv
    for every image used in the tests, how many networks labelled a certain image with a certain label
- test_inn_i_j.csv
    for every network of the ensemble, some infos about gap, time, etc., are collected
- test_inn_i_j_weights.csv
    for every network of the ensemble, the weights distribution is collected
The last file, test_inn.csv, contains results on accuracy and label statuses


Last version: 2023/06/27
"""

# The code uses GUROBI as the linear solver
import gurobipy as gp
from gurobipy import GRB
# Time package
import time
# MNIST dataset
from keras.datasets import mnist
# Numpy package for fast vector operations
import numpy as np


def index_function(training_images_total, index_training_images, instances_total, instance):
    """Function for the indexes of the pictures used in the training so that for different instances
    different pictures are used"""
    first_image_index = sum(training_images_total[:index_training_images])*instances_total + training_images_total[index_training_images]*instance
    last_image_index = first_image_index + training_images_total[index_training_images]
    return first_image_index, last_image_index


def activation(p):
    """Activation function used for the networks"""
    if p >= 0:
        return 1
    else:
        return -1


def normed_sign(r):
    """Function that counts the non-zero weights"""
    if r == 0:
        return 0
    else:
        return 1


def neural_network_evaluator(input_data_e, weights_of_network, network_structure, numbers):
    """Given input data as list of values, weights as a dictionary with keys (i, l, j) and values the value of the
    weight, network structure as the list of the number of neurons for every layer, numbers as the list of the two
    numbers the network is trained to distinguish between, the output is the predicted label of the input data"""
    network_length = len(network_structure)
    z = np.array(input_data_e)
    for level in range(1, network_length):
        z = [np.dot(z, np.array([weights_of_network[(i, level, k)] for i in range(1, network_structure[level-1]+1)])) for k in range(1, network_structure[level]+1)]
        z = np.array([activation(z[i]) for i in range(len(z))])
    if z[0] == 1:
        label_of_image = numbers[0]
    else:
        label_of_image = numbers[1]
    return label_of_image


def inn_hybrid(x, y, N, bounds, numbers, instance_number):

    """ Given x list of images, y list of labels, N network structure, bounds = max(abs(x)), numbers = [i, j] to
    distinguish between, and the instance number, it gives the dictionary of the weights of the trained network with
    keys (i, l, j) """

    # Setting value of epsilon
    eps = 0.1
    # Retrieving article notation
    L = len(N) - 1
    T = len(x)

    '''Preprocessing for the weights automatically set to 0'''
    
    t2 = np.zeros(len(x[0]))
    for t in range(T):
        t2 = np.add(t2, np.array(x[t]))
    indexes_list = list(np.where(t2 != 0)[0] + np.ones(len(np.where(t2 != 0)[0])))
    active_indexes = []
    for h in range(len(indexes_list)):
        active_indexes.append(int(indexes_list[h]))

    '''Starting the first model SM'''

    start_first_model = time.time()
    model = gp.Model()
    # weight variables for the first layer
    w_init = model.addVars([(i, 1, j) for i in active_indexes for j in range(1, N[1] + 1)], lb=-P, ub=P,
                           vtype=GRB.INTEGER)
    # weight variables for the other layers
    w = model.addVars([(i, l, j) for l in range(2, L + 1) for i in range(1, N[l - 1] + 1) for j in range(1, N[l] + 1)],
                      lb=-P, ub=P, vtype=GRB.INTEGER)
    # u variables modelling the activations
    u = model.addVars([(l, j, k) for l in range(1, L) for j in range(1, N[l] + 1) for k in range(1, T + 1)],
                      vtype=GRB.BINARY)
    # c variables modelling the products for the first layer
    c_init = model.addVars([(i, 1, j, k) for i in active_indexes for j in range(1, N[1] + 1)
                            for k in range(1, T + 1)], lb=-P*bounds, ub=P*bounds, vtype=GRB.CONTINUOUS)
    # c variables modelling the products for the other layers
    c = model.addVars([(i, l, j, k) for l in range(2, L + 1) for i in range(1, N[l - 1] + 1) for j in range(1, N[l] + 1)
                       for k in range(1, T + 1)], lb=-P, ub=P, vtype=GRB.INTEGER)
    # margin variables
    m = model.addVars([(l, j) for l in range(1, L + 1) for j in range(1, N[l] + 1)], lb=0, vtype=GRB.CONTINUOUS)
    # q variables counting the confidently corrected predicted images
    q = model.addVars([(j,k) for k in range(1, T + 1) for j in range(1, N[L] + 1)], vtype=GRB.BINARY)

    '''Constraints modelling the products c = uw'''
    for i in active_indexes:
        for j in range(1, N[1] + 1):
            for k in range(1, T + 1):
                model.addLConstr(c_init[(i, 1, j, k)] == x[k - 1][i - 1] * w_init[(i, 1, j)],
                                 name="delete3_{}_{}_{}".format(i, j, k))
    for l in range(2, L + 1):
        for j in range(1, N[l] + 1):
            for k in range(1, T + 1):
                for i in range(1, N[l - 1] + 1):
                    model.addLConstr(c[(i, l, j, k)] - w[(i, l, j)] + 2 * P * u[(l - 1, i, k)] <= 2 * P,
                                     name="delete4_{}_{}_{}_{}".format(l, j, k, i))
                    model.addLConstr(c[(i, l, j, k)] + w[(i, l, j)] - 2 * P * u[(l - 1, i, k)] <= 0,
                                     name="delete5_{}_{}_{}_{}".format(l, j, k, i))
                    model.addLConstr(c[(i, l, j, k)] - w[(i, l, j)] - 2 * P * u[(l - 1, i, k)] >= -2 * P,
                                     name="delete6_{}_{}_{}_{}".format(l, j, k, i))
                    model.addLConstr(c[(i, l, j, k)] + w[(i, l, j)] + 2 * P * u[(l - 1, i, k)] >= 0,
                                     name="delete7_{}_{}_{}_{}".format(l, j, k, i))

    '''Constraints for the last layer prediction'''
    for j in range(1, N[L] + 1):
        for k in range(1, T + 1):
            if L != 1:
                new_eps = eps / (2 * (P*(N[L-1]+1)))
                model.addLConstr((1 - q[(j, k)]) * (5/2) + y[k - 1][j - 1] * (2/(P*(N[L-1]+1))) * gp.quicksum(
                    c[(i, L, j, k)] for i in range(1, N[L - 1] + 1)) >= 1 / 2, name="delete1_{}_{}".format(j, k))
                model.addLConstr(- q[(j, k)] * (5/2) + y[k - 1][j - 1] * (2/(P*(N[L-1]+1))) * gp.quicksum(
                    c[(i, L, j, k)] for i in range(1, N[L - 1] + 1)) <= 1 / 2 - new_eps, name="delete2_{}_{}".format(j, k))
            else:
                new_eps = eps / (2 * (P * (len(active_indexes)+1)))
                model.addLConstr((1 - q[(j, k)]) * (5 / 2) + y[k - 1][j - 1] * (2 / (P * (len(active_indexes)+1))) * gp.quicksum(
                    c_init[(i, 1, j, k)] for i in active_indexes) >= 1 / 2, name="delete1_{}_{}".format(j, k))
                model.addLConstr(- q[(j, k)] * (5 / 2) + y[k - 1][j - 1] * (2 / (P * (len(active_indexes)+1))) * gp.quicksum(
                    c_init[(i, 1, j, k)] for i in active_indexes) <= 1 / 2 - new_eps, name="delete2_{}_{}".format(j, k))

    '''Constraints for the activation function modelling'''
    for l in range(2, L):
        for j in range(1, N[l] + 1):
            for k in range(1, T + 1):
                model.addLConstr(gp.quicksum(c[(i, l, j, k)] for i in range(1, N[l - 1] + 1)) >= (u[(l, j, k)] - 1) *
                                 (2 * P * (N[l - 1] + 1)) + m[(l, j)], name="delete8_{}_{}_{}".format(l, j, k))
                model.addLConstr(gp.quicksum(c[(i, l, j, k)] for i in range(1, N[l - 1] + 1)) <= u[(l, j, k)] *
                                 (2 * P * (N[l - 1] + 1)) - (m[(l, j)] + eps), name="delete9_{}_{}_{}".format(l, j, k))
    if L >= 2:
        for j in range(1, N[1] + 1):
            for k in range(1, T + 1):
                model.addLConstr(gp.quicksum(c_init[(i, 1, j, k)] for i in active_indexes) >= (u[(1, j, k)] - 1) *
                                 (2 * bounds * P * (N[0] + 1)) + m[(1, j)], name="delete10_{}_{}".format(j, k))
                model.addLConstr(gp.quicksum(c_init[(i, 1, j, k)] for i in active_indexes) <= u[(1, j, k)] *
                                 (2 * bounds * P * (N[0] + 1)) - (m[(1, j)] + eps), name="delete11_{}_{}".format(j, k))

    '''Objective function: maximizing the confidently corrected predicted images'''
    model.setObjective(gp.quicksum(q[(j, k)] for j in range(1, N[L] + 1) for k in range(1, T + 1)), GRB.MAXIMIZE)
    model.Params.Timelimit = fm_time
    model.update()
    model.optimize()
    # Retrieving model time
    time_first_model = time.time() - start_first_model

    '''Staring the second model MM'''
    start_max_margin = time.time()
    # Retrieving solution of first model
    weight_mip_fm_init = {k: round(v.X) for k, v in w_init.items()}
    weight_mip_fm = {k: round(v.X) for k, v in w.items()}
    margin_fm = {k: v.X for k, v in m.items()}
    for j in range(1, N[L]+1):
        margin_fm[(L, j)] = ((N[L-1]+1)/4)-eps
    c_ws_fm = {k: round(v.X) for k, v in c.items()}
    c_init_ws_fm = {k: v.X for k, v in c_init.items()}
    u_ws_fm = {k: round(v.X) for k, v in u.items()}
    q_results = {k: v.X for k, v in q.items()}
    # Find the images that were confidently correctly predicted
    new_T = []
    for k in range(1, T+1):
        cont = 0
        for j in range(1, N[L]+1):
            if q_results[(j, k)] == 1:
                cont += 1
        if cont == N[L]:
            new_T.append(k)
    # Give previous solution as warm start
    for z0 in weight_mip_fm.keys():
        w[z0].start = weight_mip_fm[z0]
    for z1 in u_ws_fm.keys():
        u[z1].start = u_ws_fm[z1]
    for z2 in c_init_ws_fm.keys():
        c_init[z2].start = c_init_ws_fm[z2]
    for z3 in c_ws_fm.keys():
        c[z3].start = c_ws_fm[z3]
    for z4 in weight_mip_fm_init.keys():
        w_init[z4].start = weight_mip_fm_init[z4]
    for z5 in margin_fm.keys():
        m[z5].start = margin_fm[z5]

    '''Deleting constraints for the images that were not confidently correctly classified'''
    for j in range(1, N[L] + 1):
        for k in range(1, T + 1):
            model.remove(model.getConstrByName("delete1_{}_{}".format(j, k)))
            model.remove(model.getConstrByName("delete2_{}_{}".format(j, k)))
    for k in range(1, T + 1):
        if k not in new_T:
            for i in active_indexes:
                for j in range(1, N[1] + 1):
                    model.remove(model.getConstrByName("delete3_{}_{}_{}".format(i, j, k)))
    for k in range(1, T + 1):
        if k not in new_T:
            for l in range(2, L + 1):
                for j in range(1, N[l] + 1):
                    for i in range(1, N[l - 1] + 1):
                        model.remove(model.getConstrByName("delete4_{}_{}_{}_{}".format(l, j, k, i)))
                        model.remove(model.getConstrByName("delete5_{}_{}_{}_{}".format(l, j, k, i)))
                        model.remove(model.getConstrByName("delete6_{}_{}_{}_{}".format(l, j, k, i)))
                        model.remove(model.getConstrByName("delete7_{}_{}_{}_{}".format(l, j, k, i)))
    for k in range(1, T + 1):
        if k not in new_T:
            for l in range(2, L):
                for j in range(1, N[l] + 1):
                    model.remove(model.getConstrByName("delete8_{}_{}_{}".format(l, j, k)))
                    model.remove(model.getConstrByName("delete9_{}_{}_{}".format(l, j, k)))
    for k in range(1, T + 1):
        if k not in new_T:
            if L >= 2:
                for j in range(1, N[1] + 1):
                    model.remove(model.getConstrByName("delete10_{}_{}".format(j, k)))
                    model.remove(model.getConstrByName("delete11_{}_{}".format(j, k)))
    # Margin constraints
    for j in range(1, N[L] + 1):
        for k in new_T:
            if y[k - 1][j - 1] == 1:
                if L == 1:
                    model.addLConstr(gp.quicksum(c_init[(i, 1, j, k)] for i in active_indexes) >= m[(1, j)])
                else:
                    model.addLConstr(gp.quicksum(c[(i, L, j, k)] for i in range(1, N[L - 1] + 1)) >= m[(L, j)])
            else:
                if L == 1:
                    model.addLConstr(gp.quicksum(c_init[(i, 1, j, k)] for i in active_indexes) <= -eps - m[(1, j)])
                else:
                    model.addLConstr(gp.quicksum(c[(i, L, j, k)] for i in range(1, N[L - 1] + 1)) <= -eps - m[(L, j)])

    '''Objective function: maximizing the margins'''
    model.setObjective(gp.quicksum(m[(l, j)] for l in range(1, L + 1) for j in range(1, N[l] + 1)), GRB.MAXIMIZE)
    model.Params.Timelimit = margin_time + max(fm_time - time_first_model, 0)
    model.update()
    model.optimize()
    # Retrieving model time
    time_max_margin = time.time() - start_max_margin

    '''Staring the third model MW'''
    start_min_weight = time.time()
    # Retrieving solution of second model
    weight_mip_m_init = {k: round(v.X) for k, v in w_init.items()}
    weight_mip_m = {k: round(v.X) for k, v in w.items()}
    margin = {k: v.X for k, v in m.items()}
    c_ws = {k: round(v.X) for k, v in c.items()}
    c_init_ws = {k: v.X for k, v in c_init.items()}
    u_ws = {k: round(v.X) for k, v in u.items()}
    gap_max_margin = model.MIPGap
    number_of_non_zeros_weight_mm = total_links - (N[0]-len(active_indexes))*N[1] - (sum(value == 0 for value in weight_mip_m.values()) + sum(value == 0 for value in weight_mip_m_init.values()))
    v_ws = {k: normed_sign(v) for k, v in weight_mip_m.items()}
    v_ws_init = {k: normed_sign(v) for k, v in weight_mip_m_init.items()}
    # Variables for absolute value of weights
    v_init = model.addVars([(i, 1, j) for i in active_indexes for j in range(1, N[1] + 1)], vtype=GRB.BINARY)
    v = model.addVars([(i, l, j) for l in range(2, L + 1) for i in range(1, N[l - 1] + 1) for j in range(1, N[l] + 1)],
                      vtype=GRB.BINARY)
    # Fixing the margins
    for l in range(1, L + 1):
        for j in range(1, N[l] + 1):
            model.addLConstr(m[(l, j)] == margin[(l, j)])
    # Giving the rest as a warm start
    for z0 in weight_mip_m.keys():
        w[z0].start = weight_mip_m[z0]
        v[z0].start = v_ws[z0]
    for z1 in u_ws.keys():
        u[z1].start = u_ws[z1]
    for z2 in c_init_ws.keys():
        c_init[z2].start = c_init_ws[z2]
    for z3 in c_ws.keys():
        c[z3].start = c_ws[z3]
    for z4 in weight_mip_m_init.keys():
        w_init[z4].start = weight_mip_m_init[z4]
    for z5 in v_ws_init.keys():
        v_init[z5].start = v_ws_init[z5]
    # Constraints for absolute value of weights
    for l in range(2, L + 1):
        for i in range(1, N[l - 1] + 1):
            for j in range(1, N[l] + 1):
                model.addLConstr(-v[(i, l, j)] * P <= w[(i, l, j)])
                model.addLConstr(w[(i, l, j)] <= v[(i, l, j)] * P)
    for i in active_indexes:
        for j in range(1, N[1] + 1):
            model.addLConstr(-v_init[(i, 1, j)] * P <= w_init[(i, 1, j)])
            model.addLConstr(w_init[(i, 1, j)] <= v_init[(i, 1, j)] * P)

    model.setObjective((gp.quicksum(v[(i, l, j)] for l in range(2, L + 1) for i in range(1, N[l - 1] + 1)
                                    for j in range(1, N[l] + 1)) + gp.quicksum(v_init[(i, 1, j)] for i in active_indexes
                                                                               for j in range(1, N[1] + 1))),
                       GRB.MINIMIZE)
    model.Params.Timelimit = weight_time + max(margin_time + fm_time - time_first_model - time_max_margin, 0)
    model.update()
    model.optimize()
    weight_hybrid = {}
    try:
        weight_hybrid_partial = {k: round(v.X) for k, v in w.items()}
        weight_hybrid_init = {k: round(v.X) for k, v in w_init.items()}
        gap = model.MIPGap
        for l in range(2, L + 1):
            for i in range(1, N[l - 1] + 1):
                for j in range(1, N[l] + 1):
                    weight_hybrid[(i, l, j)] = weight_hybrid_partial[(i, l, j)]
        for i in range(1, N[0] + 1):
            for j in range(1, N[1] + 1):
                if i in active_indexes:
                    weight_hybrid[(i, 1, j)] = weight_hybrid_init[(i, 1, j)]
                else:
                    weight_hybrid[(i, 1, j)] = 0
    except:
        gap = -1
        for l in range(2, L + 1):
            for i in range(1, N[l - 1] + 1):
                for j in range(1, N[l] + 1):
                    weight_hybrid[(i, l, j)] = 0

    '''Objective function: minimmizing the number of non-zero weights'''
    total_weight_final = total_links - sum(value == 0 for value in weight_hybrid.values())
    end_min_weight = time.time()

    '''Retrieving some information'''
    time_min_weight = end_min_weight - start_min_weight
    info_one = [numbers[0], numbers[1], time_first_model, time_max_margin, time_min_weight, len(new_T), number_of_non_zeros_weight_mm,
                total_weight_final, gap_max_margin, gap]
    '''
    test_inn_i_j.csv
    
    info_one = first training number / second training number / total time first model /total time for max margin solution 
                / total time min weight solution / correctly classified training images / total number of non-zero 
                weights after max margin / total number of non-zero weights after min weight / gap max margin / 
                gap min weight 
    '''

    F = open("results_{}/test_inn_{}_{}.csv".format(P, round(len(y) / 2), instance_number), "a")
    F.write(",".join([str(o) for o in info_one]))
    F.write("\n")
    F.close()

    info_two = [sum(value == p for value in weight_hybrid.values()) for p in range(-P, P+1)]

    '''
    test_inn_i_j_weights.csv
    
    info_two = first training number / second training number / how many weights set to -P / how many weights set to -P + 1 / 
                / how many weights set to -P + 2 / ... / how many weights set to P - 1 / how many weights set to P
    '''

    E = open("results_{}/test_inn_{}_{}_weights.csv".format(P, round(len(y) / 2), instance_number), "a")
    E.write(",".join([str(o) for o in info_two]))
    E.write("\n")
    E.close()

    return weight_hybrid


""" PARAMETERS """

# Network structure
N = [784, 4, 4, 1]
# Total number of weights
total_links = sum([N[i]*N[i+1] for i in range(len(N)-1)])
# Number of images per digit for the test
testing_images = 800
# Time limit for SM model
fm_time = 10
# Time limit for MM model
margin_time = 10
# Time limit for MW model
weight_time = 5
# Different values of training images
training_images = [2, 6, 10]
# Number of instances to see the average behaviour
instances = 2
# Digits to be distinguished, for a complete MNIST, set classes = [i for i in range(10)]
classes = [4, 9]
# Different values of P, for inns only, set different_p = [1]
# for every value p in different_p, a folder named "results_p" needs to be created in the directory
different_p = [1, 3]


""" RETRIEVING DATA """

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# collection of training data as lists of lists
train_X_list = []
for i in range(len(train_X)):
    v = []
    for j in range(len(train_X[i])):
        for k in range(len(train_X[i][j])):
            v.append(train_X[i][j][k])
    train_X_list.append(v)

# collection of test data as lists of lists
test_X_list = []
for i in range(len(test_X)):
    v = []
    for j in range(len(test_X[i])):
        for k in range(len(test_X[i][j])):
            v.append(test_X[i][j][k])
    test_X_list.append(v)

# indexes of data regarding their labels
indices_of_labels_train = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
for i in range(len(train_y)):
    indices_of_labels_train[train_y[i]].append(i)

indices_of_labels_test = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
for i in range(len(test_y)):
    indices_of_labels_test[test_y[i]].append(i)


""" SETTING A DATA DICTIONARY FOR ALL THE EXPERIMENTS OF THE CODE"""

X = {}
Y = {}
for z in range(instances):
    for t1 in range(len(training_images)):
        t = training_images[t1]
        x_train = {}
        y_train = {}
        for i1 in range(len(classes)):
            for j1 in range(i1 + 1, len(classes)):
                i = classes[i1]
                j = classes[j1]
                a, b = index_function(training_images, t1, instances, z)
                x1 = [train_X_list[indices_of_labels_train[i][k]] for k in range(a, b)]
                x2 = [train_X_list[indices_of_labels_train[j][k]] for k in range(a, b)]
                y1 = [[1] for _ in range(t)]
                y2 = [[-1] for _ in range(t)]
                x_train[(i, j)] = x1 + x2
                y_train[(i, j)] = y1 + y2
        '''X is a dictionary with keys (number_of_images_for_the_training, number_of_instance)
        so fixed a number of images and an instance, we get a dictionary x_train with keys (i, j)
        x_train[(i, j)] is a list of images that have to be given as training data to the network that distinguishes
        between i and j. Y is the same for the labels'''
        X[(t, z)] = x_train
        Y[(t, z)] = y_train


""" STARTING THE OPTIMIZATION """

for P in different_p:
    for i1 in training_images:
        for z1 in range(instances):
            count_status = [0, 0, 0, 0, 0, 0, 0]
            data = (X[(i1, z1)], Y[(i1, z1)])
            (x, y) = data
            # constructing alle the 45 nets
            labels = []
            weights = {}
            for i2 in range(len(classes)):
                for j2 in range(i2 + 1, len(classes)):
                    i = classes[i2]
                    j = classes[j2]
                    maximum = []
                    for k in range(len(x[(i, j)])):
                        maximum.append(max([abs(t) for t in x[(i, j)][k]]))
                    bound = max(maximum) + 1
                    ''' weights is a dictionary of dictionaries
                    weights[(i, j)] is the dictionary of the weights of the network that distinguishes
                    between i and j '''
                    weights[(i, j)] = inn_hybrid(x[(i, j)], y[(i, j)], N, bound, [i, j], z1)

            # TESTS
            tests = []
            for i in classes:
                for j in range(testing_images):
                    tests.append(indices_of_labels_test[i][j])
            my_label = []
            right_label = []
            for k in range(len(tests)):
                x = test_X_list[tests[k]]
                y = test_y[tests[k]]
                right_label.append(y)
                tentative = {}
                '''the input data is pass onto every network'''
                for q in weights:
                    (i, j) = q
                    w = neural_network_evaluator(x, weights[q], N, [i, j])
                    tentative[q] = w
                label_nets = {}
                for d in range(10):
                    vec = []
                    for q in tentative.keys():
                        if tentative[q] == d:
                            vec.append(q)
                    label_nets[d] = vec
                # saving how many networks voted for each possible label
                label_nets_len = {k: len(v) for k, v in label_nets.items()}
                H = open("results_{}/labels_{}_{}.csv".format(P, i1, z1), "a")
                info2 = [y] + [label_nets_len[i] for i in range(10)]

                '''voting scheme and label statuses'''
                sort_label = {k: v for k, v in sorted(label_nets_len.items(), key=lambda item: item[1], reverse=True)}
                if label_nets_len[list(sort_label.keys())[0]] >= label_nets_len[list(sort_label.keys())[1]] + 1:
                    this_label = list(sort_label.keys())[0]
                    my_label.append(this_label)
                    info2.append(this_label)
                    if this_label == y:
                        status = 0
                        count_status[0] += 1
                    else:
                        status = 6
                        count_status[6] += 1
                elif label_nets_len[list(sort_label.keys())[0]] == label_nets_len[list(sort_label.keys())[1]] and \
                        label_nets_len[list(sort_label.keys())[1]] >= label_nets_len[list(sort_label.keys())[2]] + 1:
                    if list(sort_label.keys())[0] < list(sort_label.keys())[1]:
                        this_label = tentative[(list(sort_label.keys())[0], list(sort_label.keys())[1])]
                    else:
                        this_label = tentative[(list(sort_label.keys())[1], list(sort_label.keys())[0])]
                    my_label.append(this_label)
                    info2.append(this_label)
                    if this_label == y:
                        status = 1
                        count_status[1] += 1
                    elif this_label != y and (y == list(sort_label.keys())[0] or y == list(sort_label.keys())[1]):
                        status = 2
                        count_status[2] += 1
                    else:
                        status = 5
                        count_status[5] += 1
                else:
                    my_label.append(-1)
                    info2.append(-1)
                    if label_nets_len[y] == label_nets_len[list(sort_label.keys())[0]]:
                        status = 3
                        count_status[3] += 1
                    else:
                        status = 4
                        count_status[4] += 1
                info2.append(status)

                '''
                labels_i_j.csv
                
                info2 = right label / how many networks labelled it as a 0 / how many as a 1 / ... / how many as a 9 / 
                        label given / status
                status :    0 = there was one maximum value and the label was correct
                            1 = there were two maximum values and the label was correct
                            2 = there were two maximum values, the label was not the one given but the other one
                            3 = there were more than three maximum value and one of these was correct
                            4 = there were more than three maximum values, all wrong
                            5 = there were two maximum values, both wrong
                            6 = there was one maximum value and the label was wrong       
                '''

                H.write(",".join([str(x) for x in info2]))
                H.write("\n")
                H.close()

            # counting how many correct
            count = 0
            # counting how many not classified
            count1 = 0
            for h in range(len(my_label)):
                if int(my_label[h]) == right_label[h]:
                    count += 1
                elif int(my_label[h]) == -1:
                    count1 += 1
            percentage_status = []
            for i in range(len(count_status)):
                percentage_status.append(count_status[i] / len(my_label))
            info = [i1, z1, count / len(my_label), (len(my_label) - count - count1) / len(my_label),
                    count1 / len(my_label)] + percentage_status

            '''
            test_inn.csv file
            
            info = number of images of each digit given to each net(i,j) / instances
                   / percentage of correct labels / percentage of wrong labels / percentage of non-labelled 
                   / percentage of each status 
            '''

            G = open("results_{}/test_inn.csv".format(P), "a")
            G.write(",".join([str(x) for x in info]))
            G.write("\n")
            G.close()
