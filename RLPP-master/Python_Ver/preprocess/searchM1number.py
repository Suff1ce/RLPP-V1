import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.X_PART import X_PART

def searchM1number(x_full, y_onehot, M1num, M1order, his, hiddenLayerSize):
    num_of_iter = 24
    percentErrors = np.zeros((M1num, num_of_iter))
    testPercentErrors = np.zeros((M1num, num_of_iter))
    trainPercentErrors = np.zeros((M1num, num_of_iter))
    net_vec = [[None for _ in range(num_of_iter)] for _ in range(M1num)]

    # Define the neural network class
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x

    # Loop over M1 neurons
    for i in range(M1num):
        # print(M1order[:i+1].shape)
        x = X_PART(x_full, M1order[:i+1], M1num, his)
        t = y_onehot
        x = torch.tensor(x.T, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        
        for j in range(num_of_iter):
            print(f"Start {i+1} M1 neurons {j+1} iter.")
            
            model = NeuralNet(x.shape[1], hiddenLayerSize, t.shape[1])
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.05)
            
            # Splitting data into train, 
            train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.3, stratify=t.numpy())
            valid_index, test_index = train_test_split(test_index, test_size=0.5, stratify=t[test_index].numpy())
            x_train, y_train = x[train_index], t[train_index]
            x_valid, y_valid = x[valid_index], t[valid_index]
            x_test, y_test = x[test_index], t[test_index] 
            
            # training
            epochs = 2000
            losses = []
            valid_losses = []
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                if epoch % 200 == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                #validation
                with torch.no_grad():
                    valid_outputs = model(x_valid)
                    valid_loss = criterion(valid_outputs, y_valid)
                    valid_losses.append(valid_loss.item())
            
            # testing
            with torch.no_grad():
                y_pred = model(x)
                y_pred_indices = torch.argmax(y_pred, dim=1)
                y_true_indices = torch.argmax(t, dim=1)
            
                percentErrors[i, j] = (y_pred_indices != y_true_indices).float().mean().item()
                testPercentErrors[i, j] = (y_pred_indices[test_index] != y_true_indices[test_index]).float().mean().item()
                trainPercentErrors[i, j] = (y_pred_indices[train_index] != y_true_indices[train_index]).float().mean().item()
            
                net_vec[i][j] = model

    # Plot results
    def plot_errors(title, errors):
        mean_errors = np.mean(errors, axis=1)
        min_errors = np.min(errors, axis=1)
        max_errors = np.max(errors, axis=1)
        plt.errorbar(range(1, M1num+1), mean_errors, yerr=[mean_errors - min_errors, max_errors - mean_errors], fmt='-o')
        plt.xlim([0, M1num+1])
        plt.ylim([0, 0.4])
        plt.title(title)
        plt.xlabel('No of M1 neurons')
        plt.ylabel('Error Rate')
        plt.grid()

    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plot_errors('All Percent Errors', percentErrors)
    plt.subplot(3, 1, 2)
    plot_errors('Test Percent Errors', testPercentErrors)
    plt.subplot(3, 1, 3)
    plot_errors('Train Percent Errors', trainPercentErrors)
    plt.tight_layout()
    plt.show()

    # Select number of M1 neurons for transregional spike prediction
    M1num_pre = 2
    DATA_INDEX_SHORT = '01'
    bestIndex = np.argmin(testPercentErrors[M1num_pre, :])
    print(f'Decoding accuracy: {1 - percentErrors[M1num_pre, bestIndex]}')

    # Save the best model
    modelName = f'decodingModel_{DATA_INDEX_SHORT}'
    torch.save(net_vec[M1num_pre][bestIndex].state_dict(), f'decoding/{modelName}.pt')

    
