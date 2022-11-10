import random
import json
import os
import copy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error

def read_data(dataset):
    train_data_dir = os.path.join('data', dataset, 'data', 'train')
    test_data_dir = os.path.join('data', dataset, 'data', 'test')
    users = {}
    train_data = {}
    test_data = {}

    flag = False
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        
        train_data.update(cdata['user_data'])
        if not flag:
            users = cdata['users']
            flag = True

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for user_id, value in train_data.items():
        x_data = value['x']
        y_data = value['y']
        num_samples = len(value['y'])
        
        x_samples = []
        y_samples = []
        for i in range(num_samples):
            x_samples.append(x_data[i][0])
            y_samples.append([y_data[i]])
            
        X_train.append(x_samples)
        Y_train.append(y_samples)
    
    for user_id, value in test_data.items():
        x_data = value['x']
        y_data = value['y']
        num_samples = len(value['y'])
        
        x_samples = []
        y_samples = []
        for i in range(num_samples):
            x_samples.append(x_data[i][0])
            y_samples.append([y_data[i]])
            
        X_test.append(x_samples)
        Y_test.append(y_samples)

    return users, X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    users, X_train, Y_train, X_test, Y_test = read_data("human_activity")
    num_of_users = len(users)
    num_of_train_samples = []
    num_of_test_samples = []
    train_user_data = {}
    test_user_data = {}

    for user_id in range(num_of_users):
        
        x_train = X_train[user_id]
        y_train = Y_train[user_id]
        x_test = X_test[user_id]
        y_test = Y_test[user_id]
        #print(len(X_train[user_id][0]), len(Y_train[user_id][0]))
        #print(user_id, len(X_train[user_id]), len(Y_train[user_id]), \
        #        len(X_test[user_id]), len(Y_test[user_id]))
        
        #print(user_id, len(x_train), len(y_train), len(x_test), len(y_test))
        
        x_train = np.array(x_train)
        y_train = np.array(y_train).flatten()
        x_test = np.array(x_test)
        y_test = np.array(y_test).flatten()
        #print(user_id, x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # fit the model
        model = LinearRegression()
        model.fit(x_train, y_train)
        # evaluate the model
        yhat = model.predict(x_test)
        # evaluate predictions
        old_mae = mean_absolute_error(y_test, yhat)
        print(user_id, 'MAE: %.3f' % old_mae)

        # identify outliers in the training dataset
        ee = OneClassSVM(nu=0.001)
        yhat = ee.fit_predict(x_train)

        # select all rows that are not outliers
        mask = yhat != -1
        new_x_train, new_y_train = x_train[mask, :], y_train[mask]

        # summarize the shape of the updated training dataset
        #print(new_x_train.shape, new_y_train.shape)

        # fit the model
        model = LinearRegression()
        model.fit(new_x_train, new_y_train)

        # evaluate the model
        yhat = model.predict(x_test)
        
        # evaluate predictions
        new_mae = mean_absolute_error(y_test, yhat)
        print(user_id, 'MAE: %.3f' % new_mae)

        
        if new_mae < old_mae:
            temp = new_x_train.tolist()
            new_x_train = [[t] for t in temp]

            temp = x_test.tolist()
            new_x_test = [[t] for t in temp]

            new_y_train = new_y_train.tolist()
            
            new_y_test = y_test.tolist()
            
            X_train[user_id] = new_x_train
            X_test[user_id] = new_x_test
            Y_train[user_id] = new_y_train
            Y_test[user_id] = new_y_test
        
        else:
            temp = x_train.tolist()
            new_x_train = [[t] for t in temp]

            temp = x_test.tolist()
            new_x_test = [[t] for t in temp]

            new_y_train = y_train.tolist()

            new_y_test = y_test.tolist()

            X_train[user_id] = new_x_train
            X_test[user_id] = new_x_test
            Y_train[user_id] = new_y_train
            Y_test[user_id] = new_y_test
            
        #print(user_id, len(X_train[user_id]), len(Y_train[user_id]), \
        #        len(X_test[user_id]), len(Y_test[user_id]))
        #print(len(X_train[user_id][0]), len(Y_train[user_id][0]))

        #print("-"*30)

    for user_id in range(num_of_users):
        
        num_of_train_samples.append(len(X_train[user_id]))
        num_of_test_samples.append(len(X_test[user_id]))

        train_user_data[users[user_id]] = {}
        test_user_data[users[user_id]] = {}

        train_user_data[users[user_id]]['x'] = X_train[user_id]
        train_user_data[users[user_id]]['y'] = Y_train[user_id]

        test_user_data[users[user_id]]['x'] = X_test[user_id]
        test_user_data[users[user_id]]['y'] = Y_test[user_id]

    train_data = {}
    test_data = {}

    train_data['users'] = users
    test_data['users'] = users

    train_data['user_data'] = train_user_data
    test_data['user_data'] = test_user_data

    train_data['num_samples'] = num_of_train_samples
    test_data['num_samples'] = num_of_test_samples

    #json_string = json.dumps(train_data)
    with open('new_train_data.json', 'w') as outfile:
        json.dump(train_data, outfile)

    #json_string = json.dumps(test_data)
    with open('new_test_data.json', 'w') as outfile:
        json.dump(test_data, outfile)
    
    print("SUCCESS")
        
        
        