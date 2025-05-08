import os
import datetime
import logging
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch_geometric.seed import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from models import GAE_hetero_link_pred, Bipartite_link_pred
from datasets import get_sumo_dataloader

def train_bipartite_link_pred(args): 
    def train_one_epoch(model: nn.Module, train_data: HeteroData, optimizer):
        ''' train one epoch ''' 
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, 
                    train_data.edge_index_dict, 
                    train_data['demand', 'measurement'].edge_label_index, 
                    train_data['demand', 'measurement'].edge_attr)
        target = train_data['demand', 'measurement'].edge_label
        assert pred.shape == target.shape
        loss = F.mse_loss(pred, target).sqrt()
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test_one_epoch(model: nn.Module, test_data: HeteroData):
        ''' test one epoch '''
        model.eval()
        pred = model(test_data.x_dict, 
                    test_data.edge_index_dict, 
                    test_data['demand', 'measurement'].edge_label_index)
        target = test_data['demand', 'measurement'].edge_label
        assert pred.shape == target.shape
        loss = F.mse_loss(pred, target).sqrt()
        return float(loss)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    hidden_channels = args.hidden_channels # 64
    out_channels = args.out_channels # 64
    learning_rate = args.lr # 5e-4
    weight_decay = args.weight_decay # 5e-4
    epoch_n = args.epoch # 200
    period = args.period # 20
    train_set = args.data 
    model_save = args.save
    model_name = args.model
    resume = args.resume
    seed = args.seed
    assert os.path.exists(train_set), f'Invalid directory {train_set}'
    assert os.path.exists(model_save), f'Invalid directory {model_save}'

    seed_everything(seed)
    log_file_name = '{}-{:%Y-%m-%d-%H}'.format(model_name, datetime.datetime.now())
    logging.basicConfig(filename=os.path.join('logs', log_file_name), 
                        format="%(asctime)s | %(levelname)s | %(message)s", 
                        level=logging.INFO, 
                        filemode='w')
    if resume: 
        resume_checkpoint = torch.load(resume)
        model.load_state_dict(resume_checkpoint['model'])
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        start_epoch, end_epoch = resume_checkpoint['epoch'], resume_checkpoint['epoch'] + epoch_n
    else: 
        model = Bipartite_link_pred(hidden_channels=hidden_channels, out_channels=out_channels).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        start_epoch, end_epoch = 0, epoch_n

    train_loader = get_sumo_dataloader(data_dir=train_set)
    train_loss_list = []
    test_loss_list = []
    val_loss_list = []
    best_epoch = float('inf')
    for epoch in range(start_epoch, end_epoch):
        avg_train_loss = 0.0
        avg_test_loss = 0.0
        avg_val_loss = 0.0
        batch_n = len(train_loader)
        bar = tqdm(train_loader, desc='Training...'.ljust(20))
        for data in bar:
            # unidirected graph
            data = T.ToUndirected()(data)
            del data['measurement', 'rev_contributes_to', 'demand'].edge_label
            # link-level split
            train_data, val_data, test_data = T.RandomLinkSplit(
                num_val  = 0.1, # 10% validation
                num_test = 0.1, # 10% test
                neg_sampling_ratio=0.0,
                edge_types=[('demand', 'contributes_to', 'measurement')],
                rev_edge_types=[('measurement', 'rev_contributes_to', 'demand')],
            )(data)
            # Generate the co-occurence matrix of movies<>movies:
            metapath = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]
            train_data = T.AddMetaPaths(metapaths=[metapath])(train_data)

            # Apply normalization to filter the metapath:
            _, edge_weight = gcn_norm(
                train_data['measurement', 'measurement'].edge_index,
                num_nodes=train_data['measurement'].num_nodes,
                add_self_loops=False,
            )
            edge_index = train_data['measurement', 'measurement'].edge_index[:, edge_weight > 0.01]

            train_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
            val_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
            test_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index

            # train one epoch / test one epoch
            loss = train_one_epoch(model, train_data, optimizer)
            train_loss = test_one_epoch(model, train_data)
            val_loss = test_one_epoch(model, val_data)
            test_loss = test_one_epoch(model, test_data)
            bar.set_description(f'Epoch [{epoch+1}/{end_epoch}]')
            bar.set_postfix(loss=loss, train=train_loss, val=val_loss, test=test_loss)
            avg_train_loss += train_loss
            avg_test_loss += test_loss
            avg_val_loss += val_loss
        avg_train_loss /= batch_n
        avg_test_loss /= batch_n
        avg_val_loss /= batch_n
        train_loss_list.append(avg_train_loss)
        test_loss_list.append(avg_test_loss)
        val_loss_list.append(avg_val_loss)
        logging.info('[{}/{}] train loss={:.4f} | val loss = {:.4f} | test loss = {:.4f}'.format(epoch+1, end_epoch, avg_train_loss, avg_val_loss, avg_test_loss))
        if avg_test_loss < best_epoch: 
            best_epoch = avg_test_loss
            # save model
            checkpoint = {
                'epoch': epoch+1, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(model_save, f'{model_name}-epoch-{epoch+1}-loss-{best_epoch}.pth'))
        elif (epoch + 1) % period == 0: 
            # save model
            checkpoint = {
                'epoch': epoch+1, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(model_save, f'{model_name}-epoch-{epoch+1}-loss-{best_epoch}.pth'))

        

