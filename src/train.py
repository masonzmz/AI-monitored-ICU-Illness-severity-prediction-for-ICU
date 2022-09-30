import torch
import random
import numpy as np
from tqdm import tqdm

from load_daata import *
from model import *


def train_icu_mse(learning_rate, weight_decay, dropout, seed, nn_model, duration):
    torch.cuda.empty_cache()
    device = try_gpu(0)

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)

    x_val = x_val.astype(float)
    x_train = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    x_val = torch.Tensor(x_val).to(device)
    y_val = torch.Tensor(y_val).to(device)

    data_train = myData(x_train, y_train)
    data_test = myData(x_val, y_val)

    if nn_model == 'rnn':
        model = RNN(input_size=num_features, hidden_layer_size=hidden_layer_size, hidden_layers=hidden_layers,
                    output_size=duration, batch_size=128, dropout=dropout, device=device)
    elif nn_model == 'gru':
        model = GRU(input_size=num_features, hidden_layer_size=hidden_layer_size, hidden_layers=3,
                    output_size=duration, batch_size=128, dropout=dropout, device=device)
    elif nn_model == 'lstm':
        model = LSTM(input_size=num_features, hidden_layer_size=hidden_layer_size, hidden_layers=hidden_layers,
                    output_size=duration, batch_size=128, dropout=dropout, device=device)
        
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10**(-learning_rate), weight_decay=5*10**(-weight_decay))

    trainloader = torch.utils.data.DataLoader(data_train,
                                            batch_size=128,
                                            shuffle=False,
                                            drop_last=True)
    
    testloader = torch.utils.data.DataLoader(data_test,
                                         batch_size=128,
                                         shuffle=False,
                                         drop_last=True)

    print(x_train.shape)
    print(y_train.shape)

    testing_mse_losses = []
    testing_mae_losses = []
    testing_mape_losses = []
    testing_rmse_losses = []
    testing_rs_losses = []

    training_losses = []


    for i in tqdm(range(num_epochs)):
        training_loss = []
        for (input_x, input_y) in trainloader:
            
            optimizer.zero_grad()
            model = model.train()
            if nn_model == 'lstm':
                model.hidden_cell = (torch.zeros(hidden_layers, batch_size,hidden_layer_size).to(device),
                                     torch.zeros(hidden_layers, batch_size,hidden_layer_size).to(device))
            else:
                model.hidden_cell = torch.zeros((hidden_layers, batch_size, hidden_layer_size), device=device)

            input_x = input_x.view(input_x.shape[0], 48, duration)
            input_x = input_x.to(device)
            output = model(input_x)
            output = output.to(device)
            input_y = input_y.to(device)
            single_loss = mse_train(output, input_y)
            single_loss.backward()

            optimizer.step()
            training_loss.append(single_loss.item())
            input_seqs.append(output)

        training_losses.append(np.mean(training_loss))

        with torch.no_grad():
            model.eval()
            model.to(device)

            test_mse_loss = []
            test_mae_loss = []
            test_mape_loss = []
            test_rmse_loss = []
            test_rs_loss = []
            
            for (x_val_, y_val_) in testloader:
                x_val_ = x_val_.view(x_val_.shape[0], 48, duration)
                x_val_ = x_val_.to(device)
                y_val_ = y_val_.to(device)
                test_y_val_ = model(x_val_)

                # test losses
                test_mse_loss.append(mse_test(test_y_val_, y_val_).item())
                test_mae_loss.append(mae_test(test_y_val_, y_val_).item())
                test_mape_loss.append(mape_test(test_y_val_, y_val_).item())
                test_rmse_loss.append(rmse_test(test_y_val_, y_val_).item())
                test_rs_loss.append(rs_test(test_y_val_, y_val_).item())
                   
            testing_mse_losses.append(np.mean(test_mse_loss))
            testing_mae_losses.append(np.mean(test_mae_loss))
            testing_mape_losses.append(np.mean(test_mape_loss))
            testing_rmse_losses.append(np.mean(test_rmse_loss))
            testing_rs_losses.append(np.mean(test_rs_loss))

    print('\n training_loss: ', training_losses[-1])
    print('testing_mse_loss: ', testing_mse_losses[-1])
    print('testing_mae_loss: ', testing_mae_losses[-1])
    print('testing_mape_loss: ', testing_mape_losses[-1])
    print('testing_rmse_loss: ', testing_rmse_losses[-1])
    print('testing_rs_loss: ', testing_rs_losses[-1])
    
    with open(f'new_results/{nn_model}_lr{lr}_dropout{dropout}_seed{seed}.txt', 'a', encoding='utf-8') as fp:
        fp.write("testing_mse_losses:")
        fp.write('\n')
        fp.write(json.dumps(testing_mse_losses))
        fp.write('\n')
        fp.write("testing_mae_losses:")
        fp.write('\n')
        fp.write(json.dumps(testing_mae_losses))
        fp.write('\n')
        fp.write("testing_mape_losses:")
        fp.write('\n')
        fp.write(json.dumps(testing_mape_losses))
        fp.write('\n')
        fp.write("testing_rmse_losses:")
        fp.write('\n')
        fp.write(json.dumps(testing_rmse_losses))
        fp.write('\n')
        fp.write("testing_rs_losses:")
        fp.write('\n')
        fp.write(json.dumps(testing_rs_losses))
        fp.write('\n')
        fp.write("training_losses:")
        fp.write('\n')
        fp.write(json.dumps(training_losses))
        fp.write('\n')

    torch.save(model.state_dict(), f"New_results/models/{nn_model}_lr{lr}_dropout{dropout}_seed{seed}.pth")


 
