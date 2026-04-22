import time as timelib
import torch
import numpy as np
import pandas as pd
from neural_networks import multiModalRepresentation_diff, multiModalRepresentation, encoderDecoder, LateFusionNet, OKNetV4
from dataloader import gestureBlobDataset, gestureBlobBatchDataset, gestureBlobMultiDataset, size_collate_fn
from unsupervise_blob_dataset import UnsupervisedBlobDatasetProbabilistic, UnsupervisedBlobMultiDatasetProbabilistic
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from datetime import datetime
from typing import Tuple, List

# Use subset sampler for train test split

def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def calc_labels(y: torch.Tensor) -> torch.Tensor:
    # out = torch.ones(y.size()[0], 15, dtype = torch.float32)*(0.01/14)
    out = torch.zeros(y.size()[0], 15, dtype = torch.long)
    #print(out.size())
    for i in range(out.size()[0]):
        # out[i, int(y[i].item()) - 1] = 0.99
        out[i, int(y[i].item()) - 1] = 1
    return(out)

def train_multimodal_embeddings(lr: float, num_epochs: int, blobs_folder_path:str, weights_save_path: str, weight_decay: float) -> None:
    device = get_device()
    print(f'Using device: {device}')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)
    
    # Extract dataset name from path
    if 'knot' in blobs_folder_path.lower() or 'tying' in blobs_folder_path.lower():
        dataset_name = 'Knot_Tying'
    elif 'needle' in blobs_folder_path.lower() or 'passing' in blobs_folder_path.lower():
        dataset_name = 'Needle_Passing'
    elif 'suturing' in blobs_folder_path.lower():
        dataset_name = 'Suturing'
    else:
        dataset_name = 'dataset'

    gesture_dataset = gestureBlobDataset(blobs_folder_path = blobs_folder_path)
    dataset = gestureBlobBatchDataset(gesture_dataset = gesture_dataset, random_tensor = 'random')
    dataloader = DataLoader(dataset = dataset, batch_size = 2, shuffle = True)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    # net = multiModalRepresentation_diff(out_features = 2048, lstm_num_layers= 2, parser = 'cnn')
    net = multiModalRepresentation(out_features = 512, lstm_num_layers= 2, parser = 'cnn')
    net = net.train()
    net = net.to(device)

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        # for idx in range(len(dataset)):
        for data in dataloader:
            current_tensor, random_tensor, y_match, y_rand = data
            curr_opt, curr_kin = current_tensor
            _, rand_kin = random_tensor
            # kin = torch.cat([kinematics, kinematics_rand], dim = 1)
            curr_opt = curr_opt.to(device)
            curr_kin = curr_kin.to(device)
            y_match = y_match.to(device)
            rand_kin = rand_kin.to(device)
            y_rand = y_rand.to(device)
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            out1 = net((curr_opt, curr_kin))
            out2 = net((curr_opt, rand_kin))
            loss = loss_function(out2, y_rand) + loss_function(out1, y_match)
            # loss = loss_function(out2.log(), y_rand) + loss_function(out1.log(), y_match) # Use for KLDivLoss
            loss.backward()
            optimizer.step()
            print('Out1: {}'.format(out1[0, :]))
            print('Out2: {}'.format(out2[0, :]))
            print('Current loss2 = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print(out1[0, :,])
        print(out2[0, :])
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_' + dataset_name + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))

def train_encoder_decoder_embeddings(lr: float, num_epochs: int, blobs_folder_path: str, weights_save_path: str, weight_decay: float, dataset_name: str, time: bool = False) -> None:
    # set batch size
    bs = 128
    
    device = get_device()
    print(f'Using device: {device}')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset(blobs_folder_path = blobs_folder_path, time = time) # time is for unstacked temporal vector [2, 25, 240, 320]
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = bs, shuffle = False, collate_fn = size_collate_fn)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    
    # net = encoderDecoder(embedding_dim = 512)
    net = LateFusionNet(embedding_dim=512, batch_size=bs)
    
    net = net.train()
    net = net.to(device)

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for data in dataloader:
            curr_opt, curr_kin = data
            curr_opt = curr_opt.to(device)
            curr_kin = curr_kin.to(device)
            optimizer.zero_grad()
            out1 = net(curr_opt)
            loss = loss_function(out1, curr_kin)
            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_' + dataset_name + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))

def train_encoder_decoder_multidata_embeddings(lr: float, num_epochs: int, blobs_folder_paths_list: List[str], weights_save_path: str, weight_decay: float) -> None:
    device = get_device()
    print(f'Using device: {device}')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobMultiDataset(blobs_folder_paths_list = blobs_folder_paths_list)
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = 128, shuffle = False, collate_fn = size_collate_fn)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    net = encoderDecoder(embedding_dim = 2048)
    net = net.train()
    net = net.to(device)

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for data in dataloader:
            curr_opt, curr_kin = data
            curr_opt = curr_opt.to(device)
            curr_kin = curr_kin.to(device)
            optimizer.zero_grad()
            out1 = net(curr_opt)
            loss = loss_function(out1, curr_kin)
            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_multidata_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))


def train_ok_network_embeddings(lr: float, num_epochs: int, blobs_folder_path: str, weights_save_path: str, weight_decay: float, dataset_name: str, time: bool = False) -> None:
    """
    Training loop for the OKNet for one action
    """
    
    
    # set batch size
    bs = 1
    
    device = get_device()
    print(f'Using device: {device}')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)
        
    gesture_dataset = UnsupervisedBlobDatasetProbabilistic(blobs_folder_path = blobs_folder_path, time = time)
    #dataloader = DataLoader(dataset = gesture_dataset, batch_size = bs, shuffle = False, collate_fn = size_collate_fn)
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = bs, shuffle = False)

    loss_function = torch.nn.BCEWithLogitsLoss()
    
    net = OKNetV4(embedding_dim=2048)
    
    net = net.train()
    net = net.to(device)

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)
    
    # loss and accuracy logger 
    loss_logger = pd.DataFrame(columns=['epoch', 'loss', 'time_elapsed'])
    acc_logger = pd.DataFrame(columns=['epoch', 'accuracy'])
    

    for epoch in range(num_epochs):
        time1 = timelib.time()
        print_epoch = epoch + 1
        running_loss = 0
        count = 0

        print('Epoch {}'.format(epoch + 1))
        for batch_idx, (x, y) in enumerate(dataloader):
            curr_opt, curr_kin = x
            #print(count) # sanity check
            #print(curr_opt.size(), curr_kin.size())
            curr_opt = curr_opt.to(device)
            curr_kin = curr_kin.to(device)
            y = y.to(device)
                
            optimizer.zero_grad()
            
            out1 = net((curr_opt, curr_kin))
            
            loss = loss_function(out1, y)
            
            try:
                loss.backward()
                
                optimizer.step()
                
            except Exception as e:
                print((curr_opt.size(), curr_kin.size()))
                print(e)
                pass
            
            # print('Current loss = {}'.format(loss.item()))

            running_loss += loss.item() * y.shape[0]
            count += y.shape[0]
            
            #count += 1
        
        time2 = timelib.time()
        # print epoch information
        
        print(f'\n Epoch: {print_epoch}, Loss: {(running_loss/count):.5f}')
        print(f'Elapsed time for epoch: { (time2 - time1):.4f} s')
        loss_logger = pd.concat([loss_logger, pd.DataFrame({'epoch':[print_epoch], 'loss':[running_loss/count], 'time_elapsed':[time2-time1]})])
        
        # save state dict and compute accuracy every 10 epochs
        if print_epoch % 10 == 0:
        
            now = datetime.now()
            now = '_'.join((str(now).split('.')[0]).split(' ')).replace(':','_')
            file_name = 'multimodal_' + dataset_name + '_epoch_' + str(print_epoch) + '_' + now + '.pth'
            file_name = os.path.join(weights_save_path, file_name)
            torch.save(net.state_dict(), file_name)
            
            with torch.no_grad():
                #y_true = []
                #y_pred = []
                
                acc_sum = 0
                pos_samples = 0
                total = 0
                for batch_idx, (x, y) in enumerate(dataloader):
                    opt = x[0]
                    kin = x[1]
                    opt = opt.to(device)
                    kin = kin.to(device)
                    y = y.to(device)
    
                    outputs = net((opt, kin))
                    predictions = (outputs > 0.5).float()
                    pos_samples += torch.sum(y).data.item()
                    acc_sum += torch.sum(predictions == y)
                    total += y.shape[0]
                    
                    # are these next 2 lines needed?
                    #y_pred += predictions.cpu().detach().numpy().squeeze().tolist()
                    #y_true += y.cpu().detach().numpy().squeeze().tolist()
    
                acc = acc_sum / total
                acc = acc.cpu().detach().numpy()
    
                acc_logger = pd.concat([acc_logger, pd.DataFrame({'epoch':[print_epoch], 'acc':[acc]})])
                print(f'\n Epoch: {print_epoch}, Accuracy: {acc:.4f}')
        

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' ')).replace(':','_')
    file_name = 'multimodal_' + dataset_name + '_epoch_' + str(print_epoch) + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))
    
    # save loggers
    loss_logger.to_csv(os.path.join(weights_save_path, 'losses.csv'))
    acc_logger.to_csv(os.path.join(weights_save_path, 'accuracies.csv'))
    
    
def train_ok_network_multidata_embeddings(lr: float, num_epochs: int, blobs_folder_paths_list: List[str], weights_save_path: str, weight_decay: float, dataset_name: str, time: bool = False) -> None:
    """
    Training loop for the OKNet for one action
    """
    
    
    # set batch size
    bs = 128
    
    device = get_device()
    print(f'Using device: {device}')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)
        
    # ============================================================================================================
    gesture_dataset = UnsupervisedBlobMultiDatasetProbabilistic(blobs_folder_paths_list = blobs_folder_paths_list, time = time)
    # ============================================================================================================
    
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = bs, shuffle = False, collate_fn = size_collate_fn)

    loss_function = torch.nn.BCEWithLogitsLoss()
    
    net = OKNetV4(batch_size=bs, embedding_dim=512)
    
    net = net.train()
    net = net.to(device)

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)
    
    # loss and accuracy logger 
    loss_logger = pd.DataFrame(columns=['epoch', 'loss', 'time_elapsed'])
    acc_logger = pd.DataFrame(columns=['epoch', 'accuracy'])
    

    for epoch in range(num_epochs):
        time1 = time.time()
        print_epoch = epoch + 1
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for batch_idx, (x, y) in enumerate(dataloader):
            curr_opt, curr_kin = x
            curr_opt = curr_opt.to(device)
            curr_kin = curr_kin.to(device)
            y = y.to(device)
                
            optimizer.zero_grad()
            
            out1 = net((curr_opt, curr_kin))
            
            loss = loss_function(out1, y)
            loss.backward()
            
            optimizer.step()
            
            # print('Current loss = {}'.format(loss.item()))

            running_loss += loss.item() * y.shape[0]
            count += y.shape[0]
            
            count += 1
        
        time2 = time.time()
        # print epoch information
        
        print(f'\n Epoch: {print_epoch}, Loss: {(running_loss/count):.5f}')
        print(f'Elapsed time for epoch: { time2 - time1:0.04f} s')
        loss_logger = pd.concat([loss_logger, pd.DataFrame({'epoch':[print_epoch], 'loss':[running_loss/count], 'time_elapsed':[time2-time1]})])
        
        # save state dict and compute accuracy every 10 epochs
        if print_epoch % 10 == 0:
        
            now = datetime.now()
            now = '_'.join((str(now).split('.')[0]).split(' '))
            file_name = 'multimodal_' + dataset_name + '_' + print_epoch + '_' + now + '.pth'
            file_name = os.path.join(weights_save_path, file_name)
            torch.save(net.state_dict(), file_name)
            
            with torch.no_grad():
                y_true = []
                y_pred = []
                
                acc_sum = 0
                pos_samples = 0
                total = 0
                for batch_idx, (x, y) in enumerate(dataloader):
                    opt = x[0]
                    kin = x[1]
                    opt = opt.to(device)
                    kin = kin.to(device)
                    y = y.to(device)
    
                    outputs = net((opt, kin))
                    predictions = (outputs > 0.5).float()
                    pos_samples += torch.sum(y).data.item()
                    acc_sum += torch.sum(predictions == y)
                    total += y.shape[0]
                    
                    # are these next 2 lines needed?
                    y_pred += predictions.cpu().detach().numpy().squeeze().tolist()
                    y_true += y.cpu().detach().numpy().squeeze().tolist()
    
                acc = acc_sum / total
                acc_logger = pd.concat([acc_logger, pd.DataFrame({'epoch':[print_epoch], 'acc':[acc]})])
                print(f'\n Epoch: {print_epoch}, Accuracy: {acc:.4f}')
        

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_' + dataset_name + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))
    
    # save loggers
    loss_logger.to_csv(os.path.join(weights_save_path, 'losses.csv'))
    acc_logger.to_csv(os.path.join(weights_save_path, 'accuracies.csv'))

def main():
    blobs_folder_path = '../jigsaw_dataset/Knot_Tying/blobs'
    lr = 1e-3
    num_epochs = 1000
    weights_save_path = './weights_save'
    weight_decay = 1e-8
    dataset_name = 'Knot_Tying'

    blobs_folder_paths_list = ['../jigsaw_dataset/Knot_Tying/blobs', '../jigsaw_dataset/Needle_Passing/blobs', '../jigsaw_dataset/Suturing/blobs']

    # train_multimodal_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay)
    # train_encoder_decoder_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay, dataset_name = dataset_name)
    train_encoder_decoder_multidata_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_paths_list = blobs_folder_paths_list, weights_save_path = weights_save_path, weight_decay = weight_decay)

if __name__ == '__main__':
    main()
