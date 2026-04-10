import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def searchHiddenSize(x_full, y_onehot):
    x = torch.tensor(x_full, dtype=torch.float32)
    t = torch.tensor(y_onehot, dtype=torch.float32)

    NumOfPower = 8
    NumOfIter = 24
    percentErrors = np.zeros((NumOfPower, NumOfIter))
    testPercentErrors = np.zeros((NumOfPower, NumOfIter))
    trainPercentErrors = np.zeros((NumOfPower, NumOfIter))

    # define Neural Network
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
            return self.softmax(x)

    for i in range(NumOfPower):
        hiddenLayerSize = 2 ** (i + 1)
        for j in range(NumOfIter):
            print(f"Start {hiddenLayerSize} Hidden units... {j + 1} iter.")
            
            train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.3, stratify=t.numpy())
            valid_index, test_index = train_test_split(test_index, test_size=0.5, stratify=t[test_index].numpy())
            x_train, y_train = x[train_index], t[train_index]
            x_valid, y_valid = x[valid_index], t[valid_index]
            x_test, y_test = x[test_index], t[test_index] 
            
            model = NeuralNet(x.shape[1], hiddenLayerSize, t.shape[1])
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=2.0)
            
            # training
            epochs = 1000
            valid_losses = []
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                if epoch % 200 == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                #validation
                with torch.no_grad():
                    valid_outputs = model(x_valid)
                    valid_loss = criterion(valid_outputs, y_valid)
                    valid_losses.append(valid_loss.item())
                    # if len(valid_losses) > 1 and valid_losses[-1] > valid_losses[-2]:
                    #     print(f"Validation loss increased at epoch {epoch}, stopping training.")
                    #     break
            
            # testing
            with torch.no_grad():
                y_pred = model(x)
                y_pred_indices = torch.argmax(y_pred, dim=1)
                y_true_indices = torch.argmax(t, dim=1)
                
                percentErrors[i, j] = (y_pred_indices != y_true_indices).float().mean().item()
                
                testPercentErrors[i, j] = (y_pred_indices[test_index] != y_true_indices[test_index]).float().mean().item()
                trainPercentErrors[i, j] = (y_pred_indices[train_index] != y_true_indices[train_index]).float().mean().item()


    # Plot results
    x_vals = np.arange(1, NumOfPower+1)
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    for ax, errors, title in zip(axes, [percentErrors, testPercentErrors, trainPercentErrors],
                                ['All Percent Errors', 'Test Percent Errors', 'Train Percent Errors']):
        mean_errors = np.mean(errors, axis=1)
        min_errors = np.min(errors, axis=1)
        max_errors = np.max(errors, axis=1)
        
        ax.errorbar(x_vals, mean_errors, yerr=[mean_errors - min_errors, max_errors - mean_errors], fmt='o')
        ax.plot(x_vals, mean_errors, color='C1')
        ax.axhline(y=np.min(errors), color='r', linestyle='--')
        ax.set_xlim([0, 9])
        ax.set_ylim([0, 0.3])
        ax.set_title(title)
        
    axes[2].set_xlabel('2^x of hidden units')
    plt.tight_layout()
    plt.show()