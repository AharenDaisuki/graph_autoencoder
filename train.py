import os
import datetime
import logging
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch_geometric.seed import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.utils import spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from loss import rmse_loss, huber_loss, pinball_loss, node_u_recon_loss, node_v_recon_loss
from models import Bipartite_link_pred, Bipartite_LinkQuantileRegression_GAE
from datasets import get_sumo_trainloader, get_sumo_dataloaders

def seed_torch(seed: int): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TrainerBase(object):
    """
    Trainer base class. 
    """
    def __init__(self):
        pass 

    def _prepare_data(self, data: HeteroData): 
        raise NotImplementedError
    
    def _train_one_epoch(self, model: nn.Module, train_data: HeteroData, optimizer, edge_weight = None) -> float: 
        raise NotImplementedError
    
    @torch.no_grad
    def _test_one_epoch(self, model: nn.Module, test_data: HeteroData) -> float: 
        raise NotImplementedError
    
    def _save_model(self, cur_epoch: int, loss: float, best_epoch: float, model: nn.Module, model_save_path: str, model_name: str, optimizer):
        raise NotImplementedError
    
    def train(args): 
        raise NotImplementedError
    
## blp
class Trainer_blp(TrainerBase): 
    def __init__(self):
        super(Trainer_blp, self).__init__()

    def _prepare_data(self, data):
        # add reverse edges 
        data = T.ToUndirected()(data)
        del data['measurement', 'rev_contributes_to', 'demand'].edge_label

        # link-level split (8-1-1)
        train_data, val_data, test_data = T.RandomLinkSplit(
            num_val  = 0.1, # 10% validation
            num_test = 0.1, # 10% test
            neg_sampling_ratio=0.0,
            edge_types=[('demand', 'contributes_to', 'measurement')],
            rev_edge_types=[('measurement', 'rev_contributes_to', 'demand')],
        )(data)

        # Generate metapaths (v->u->v / u->v->u)
        metapath_0 = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]     
        # metapath_1 = [('demand', 'contributes_to', 'measurement'), ('measurement', 'rev_contributes_to', 'demand')]     
        train_data = T.AddMetaPaths(metapaths=[metapath_0])(train_data)

        # Apply normalization to filter the metapath:
        _, edge_weight = gcn_norm(
            train_data['measurement', 'measurement'].edge_index,
            num_nodes=train_data['measurement'].num_nodes,
            add_self_loops=False,
        )
        edge_index = train_data['measurement', 'measurement'].edge_index[:, edge_weight > 0.006]  

        train_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        val_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        test_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        return train_data, val_data, test_data
    
    def _train_one_epoch(self, model, train_data, optimizer, edge_weight=None):
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, 
                    train_data.edge_index_dict, 
                    train_data['demand', 'measurement'].edge_label_index)
        target = train_data['demand', 'measurement'].edge_label
        loss = huber_loss(pred=pred, target=target)
        # loss = rmse_loss(pred=pred, target=target)
        loss.backward()
        optimizer.step()
        return float(loss)

    def _test_one_epoch(self, model, test_data):
        model.eval()
        pred = model(test_data.x_dict, 
                    test_data.edge_index_dict, 
                    test_data['demand', 'measurement'].edge_label_index)
        target = test_data['demand', 'measurement'].edge_label
        loss = rmse_loss(pred=pred, target=target)
        return float(loss)
        
    def _save_model(self, cur_epoch, loss, best_epoch, model, model_save_path, model_name, optimizer):
        checkpoint = {
            'epoch': cur_epoch+1, 
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'best': best_epoch
        }
        if model_save_path: 
            torch.save(checkpoint, os.path.join(model_save_path, f'{model_name}-epoch-{cur_epoch+1}-loss-{loss}.pth'))
    
    def train(self, args):
        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        hidden_channels = args.hidden_channels # 64
        out_channels = args.out_channels # 64
        learning_rate = args.lr # 1e-3
        weight_decay = args.weight_decay # 5e-4
        epoch_n = args.epoch # 200
        period = args.period # 20
        train_set = args.data 
        model_save = args.save
        model_name = args.model
        resume = args.resume
        seed = args.seed
        swap = args.swap
        log_file_name = args.log
        assert os.path.exists(train_set), f'Invalid directory {train_set}'
        # assert os.path.exists(model_save), f'Invalid directory {model_save}'
        if model_save is None: 
            model_save = os.path.join('checkpoints', model_name)

        if not os.path.exists(model_save): 
            os.makedirs(model_save)

        # seed_everything(seed)
        seed_torch(seed)
        if log_file_name is None: 
            log_file_name = '{}-{:%Y-%m-%d-%H}'.format(model_name, datetime.datetime.now())

        logging.basicConfig(filename=os.path.join('logs', log_file_name), 
                            format="%(asctime)s | %(levelname)s | %(message)s", 
                            level=logging.INFO, 
                            filemode='a')
        model = Bipartite_link_pred(hidden_channels=hidden_channels, out_channels=out_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if resume: 
            resume_checkpoint = torch.load(resume)
            model.load_state_dict(resume_checkpoint['model'])
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            start_epoch, end_epoch = resume_checkpoint['epoch'], resume_checkpoint['epoch'] + epoch_n
            best_epoch = resume_checkpoint['best'] 
        else: 
            start_epoch, end_epoch = 0, epoch_n
            best_epoch = float('inf')

        train_loader = get_sumo_trainloader(data_dir=train_set, swap_prob=swap)
        train_loss_list = []
        test_loss_list = []
        val_loss_list = []
        # node_loss_list = []
        for epoch in range(start_epoch, end_epoch):
            avg_train_loss = 0.0
            avg_test_loss = 0.0
            avg_val_loss = 0.0
            # avg_node_loss = 0.0
            batch_n = len(train_loader)
            bar = tqdm(train_loader, desc='Training...'.ljust(20))
            for data in bar:
                train_data, val_data, test_data = self._prepare_data(data)

                # train one epoch / test one epoch
                loss = self._train_one_epoch(model, train_data, optimizer)
                train_loss = self._test_one_epoch(model, train_data)
                val_loss = self._test_one_epoch(model, val_data)
                test_loss = self._test_one_epoch(model, test_data)

                # le, ln, l = self._train_one_epoch_v2(model, train_data, data, optimizer)
                # le_train, le_test, le_val, ln_ = self._test_one_epoch_v2(model, train_data, test_data, val_data, data)
                
                bar.set_description(f'Epoch [{epoch+1}/{end_epoch}]')
                
                bar.set_postfix(loss=loss, train=train_loss, val=val_loss, test=test_loss)
                avg_train_loss += train_loss
                avg_test_loss += test_loss
                avg_val_loss += val_loss
                
                # bar.set_postfix(loss=l, edge=le, node=ln)
                # avg_train_loss += le_train
                # avg_test_loss += le_test
                # avg_val_loss += le_val
                # avg_node_loss += ln_

            avg_train_loss /= batch_n
            avg_test_loss /= batch_n
            avg_val_loss /= batch_n
            # avg_node_loss /= batch_n
            train_loss_list.append(avg_train_loss)
            test_loss_list.append(avg_test_loss)
            val_loss_list.append(avg_val_loss)
            # node_loss_list.append(avg_node_loss)
            logging.info('[{}/{}] | train loss={:.4f} | val loss={:.4f} | test loss={:.4f}'.format(epoch+1, end_epoch, avg_train_loss, avg_val_loss, avg_test_loss))
            # logging.info('[{}/{}] | train loss={:.4f} | val loss={:.4f} | test loss={:.4f} | node loss = {:.4f}'.format(epoch+1, end_epoch, avg_train_loss, avg_val_loss, avg_test_loss, avg_node_loss))
            if avg_test_loss < best_epoch: 
                best_epoch = avg_test_loss
                # save model
                self._save_model(cur_epoch=epoch, loss=avg_test_loss, best_epoch=best_epoch, 
                                 model=model, model_save_path=model_save, model_name=model_name, optimizer=optimizer)
            elif (epoch + 1) % period == 0: 
                # save model
                self._save_model(cur_epoch=epoch, loss=avg_test_loss, best_epoch=best_epoch, 
                             model=model, model_save_path=model_save, model_name=model_name, optimizer=optimizer)
        return train_loss_list, test_loss_list, val_loss_list
        # return train_loss_list, test_loss_list, val_loss_list, node_loss_list

class Trainer_blp_node_recon(Trainer_blp): 
    def __init__(self):
        super(Trainer_blp_node_recon, self).__init__()

    def _prepare_data(self, data):
        # add reverse edges 
        data = T.ToUndirected()(data)
        del data['measurement', 'rev_contributes_to', 'demand'].edge_label

        # link-level split (8-1-1)
        train_data, val_data, test_data = T.RandomLinkSplit(
            num_val  = 0.1, # 10% validation
            num_test = 0.1, # 10% test
            neg_sampling_ratio=0.0,
            edge_types=[('demand', 'contributes_to', 'measurement')],
            rev_edge_types=[('measurement', 'rev_contributes_to', 'demand')],
        )(data)

        # Generate metapaths (v->u->v / u->v->u)
        metapath_0 = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]     
        # metapath_1 = [('demand', 'contributes_to', 'measurement'), ('measurement', 'rev_contributes_to', 'demand')]     
        train_data = T.AddMetaPaths(metapaths=[metapath_0])(train_data)

        # Apply normalization to filter the metapath:
        _, edge_weight = gcn_norm(
            train_data['measurement', 'measurement'].edge_index,
            num_nodes=train_data['measurement'].num_nodes,
            add_self_loops=False,
        )
        edge_index = train_data['measurement', 'measurement'].edge_index[:, edge_weight > 0.006]  

        train_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        val_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        test_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        return train_data, val_data, test_data, data

    def _train_one_epoch(self, model, train_data, data, optimizer, gamma = 0.0): 
        model.train()
        optimizer.zero_grad()
        
        # train set
        pred_train = model(train_data.x_dict, train_data.edge_index_dict, train_data['demand', 'measurement'].edge_label_index)
        target_train = train_data['demand', 'measurement'].edge_label
        edge_loss = huber_loss(pred=pred_train, target=target_train) # 0.1
        # loss = rmse_loss(pred=pred, target=target)
        
        # complete set
        pred_complete = model(data.x_dict, data.edge_index_dict, data['demand', 'measurement'].edge_index)
        node_u_loss = node_u_recon_loss(data, pred_complete)
        node_v_loss = node_v_recon_loss(data, pred_complete)
        node_loss = node_u_loss + node_v_loss
        # node_loss = node_v_loss # 10
        loss = (1 - gamma) * edge_loss + gamma * node_loss
        loss.backward()
        optimizer.step()
        return float(edge_loss), float(node_loss), float(loss)
    
    def _test_one_epoch(self, model, train_data, test_data, val_data, data):
        model.eval()
        # train set
        pred_train = model(train_data.x_dict, train_data.edge_index_dict, train_data['demand', 'measurement'].edge_label_index)
        target_train = train_data['demand', 'measurement'].edge_label
        edge_loss_train = huber_loss(pred=pred_train, target=target_train)
        # test set
        pred_test = model(test_data.x_dict, test_data.edge_index_dict, test_data['demand', 'measurement'].edge_label_index)
        target_test = test_data['demand', 'measurement'].edge_label
        edge_loss_test = huber_loss(pred=pred_test, target=target_test)
        # validation set
        pred_val = model(val_data.x_dict, val_data.edge_index_dict, val_data['demand', 'measurement'].edge_label_index)
        target_val = val_data['demand', 'measurement'].edge_label
        edge_loss_val = huber_loss(pred=pred_val, target=target_val)
        # complete set
        pred_complete = model(data.x_dict, data.edge_index_dict, data['demand', 'measurement'].edge_index)
        node_u_loss = node_u_recon_loss(data, pred_complete)
        node_v_loss = node_v_recon_loss(data, pred_complete)
        # node_loss = node_u_loss + node_v_loss
        # node_loss = node_v_loss
        return float(edge_loss_train), float(edge_loss_test), float(edge_loss_val), float(node_u_loss), float(node_v_loss)
    
    def train(self, args):
        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        hidden_channels = args.hidden_channels # 64
        out_channels = args.out_channels # 64
        learning_rate = args.lr # 1e-3
        weight_decay = args.weight_decay # 5e-4
        epoch_n = args.epoch # 200
        period = args.period # 20
        train_set = args.data 
        model_save = args.save
        model_name = args.model
        resume = args.resume
        seed = args.seed
        # swap = args.swap
        log_file_name = args.log
        assert os.path.exists(train_set), f'Invalid directory {train_set}'
        if model_save is None: 
            model_save = os.path.join('checkpoints', model_name)

        if not os.path.exists(model_save): 
            os.makedirs(model_save)

        # seed_everything(seed)
        seed_torch(seed)
        if log_file_name is None: 
            log_file_name = '{}-{:%Y-%m-%d-%H}'.format(model_name, datetime.datetime.now())

        logging.basicConfig(filename=os.path.join('logs', log_file_name), 
                            format="%(asctime)s | %(levelname)s | %(message)s", 
                            level=logging.INFO, 
                            filemode='a')
        model = Bipartite_link_pred(hidden_channels=hidden_channels, out_channels=out_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if resume: 
            resume_checkpoint = torch.load(resume)
            model.load_state_dict(resume_checkpoint['model'])
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            start_epoch, end_epoch = resume_checkpoint['epoch'], resume_checkpoint['epoch'] + epoch_n
            best_epoch = resume_checkpoint['best'] 
        else: 
            start_epoch, end_epoch = 0, epoch_n
            best_epoch = float('inf')

        train_loader = get_sumo_trainloader(data_dir=train_set, swap_prob=0.0)
        train_loss_list = []
        test_loss_list = []
        val_loss_list = []
        node_loss_list = []
        for epoch in range(start_epoch, end_epoch):
            avg_train_loss = 0.0
            avg_test_loss = 0.0
            avg_val_loss = 0.0
            avg_node_u_loss = 0.0
            avg_node_v_loss = 0.0
            batch_n = len(train_loader)
            bar = tqdm(train_loader, desc='Training...'.ljust(20))
            for data in bar:
                train_data, val_data, test_data, data = self._prepare_data(data)
                le, ln, l = self._train_one_epoch(model, train_data, data, optimizer, gamma=0.5)
                le_train, le_test, le_val, ln_u, ln_v = self._test_one_epoch(model, train_data, test_data, val_data, data)
                bar.set_description(f'Epoch [{epoch+1}/{end_epoch}]')
                bar.set_postfix(loss=l, edge=le, node=ln)
                avg_train_loss += le_train
                avg_test_loss += le_test
                avg_val_loss += le_val
                avg_node_u_loss += ln_u
                avg_node_v_loss += ln_v
                # avg_node_loss += (ln_u + ln_v)

            avg_train_loss /= batch_n
            avg_test_loss /= batch_n
            avg_val_loss /= batch_n
            # avg_node_loss /= batch_n
            avg_node_u_loss /= batch_n
            avg_node_v_loss /= batch_n
            avg_node_loss = avg_node_u_loss + avg_node_v_loss
            train_loss_list.append(avg_train_loss)
            test_loss_list.append(avg_test_loss)
            val_loss_list.append(avg_val_loss)
            node_loss_list.append(avg_node_loss)
            logging.info(
                '[{}/{}] | train loss={:.4f} | val loss={:.4f} | test loss={:.4f} | node u loss={:.4f} | node v loss={:.4f} | node loss={:.4f}'.format(epoch+1, end_epoch, 
                        avg_train_loss, avg_val_loss, avg_test_loss, avg_node_u_loss, avg_node_v_loss, avg_node_loss)
                )
            if avg_node_loss < best_epoch: 
                best_epoch = avg_node_loss
                # save model
                self._save_model(cur_epoch=epoch, loss=avg_node_loss, best_epoch=best_epoch, 
                                 model=model, model_save_path=model_save, model_name=model_name, optimizer=optimizer)
            elif (epoch + 1) % period == 0: 
                # save model
                self._save_model(cur_epoch=epoch, loss=avg_node_loss, best_epoch=best_epoch, 
                             model=model, model_save_path=model_save, model_name=model_name, optimizer=optimizer)
        return train_loss_list, test_loss_list, val_loss_list, node_loss_list
  
class Trainer_blp_structured_node_recon(Trainer_blp_node_recon): 
    """
    Trainer for bipartite graph autoencoder.  
    """
    def __init__(self):
        super(Trainer_blp_node_recon, self).__init__()

    def _prepare_data(self, data: HeteroData, gcn_norm_threshold: float = 5e-3):
        # add reverse edges 
        data = T.ToUndirected()(data)
        del data['measurement', 'rev_contributes_to', 'demand'].edge_label
        # remove isolated nodes
        # data = T.RemoveIsolatedNodes()(data)

        # Generate metapaths (v->u->v / u->v->u)
        metapath_0 = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]     
        # metapath_1 = [('demand', 'contributes_to', 'measurement'), ('measurement', 'rev_contributes_to', 'demand')]     
        data = T.AddMetaPaths(metapaths=[metapath_0])(data)

        # Apply normalization to filter the metapath:
        _, edge_weight = gcn_norm(
            data['measurement', 'measurement'].edge_index,
            num_nodes=data['measurement'].num_nodes,
            add_self_loops=False,
        )
        edge_index = data['measurement', 'measurement'].edge_index[:, edge_weight > gcn_norm_threshold]  
        data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        return data
    
    def _train_one_epoch(self, model, train_data, optimizer, gamma: float = 1e-3): 
        model.train()
        optimizer.zero_grad()
        
        # train set
        pred = model(train_data.x_dict, train_data.edge_index_dict, train_data['demand', 'measurement'].edge_index)
        target = train_data['demand', 'measurement'].edge_label
        edge_loss = huber_loss(pred=pred, target=target, delta=0.1) # edge loss: 10^-1
        # edge_loss = rmse_loss(pred=pred, target=target)
        # node_u_loss = node_u_recon_loss(train_data, pred) # node u loss: 10
        # node_v_loss = node_v_recon_loss(train_data, pred) # node v loss: 
        # node_loss = node_u_loss + node_v_loss
        # node_loss = node_v_loss
        # loss = (1 - gamma) * edge_loss + gamma * node_loss
        # loss = edge_loss
        loss = edge_loss
        loss.backward()
        optimizer.step()
        return float(loss)
        # return float(edge_loss), float(node_u_loss), float(node_v_loss), float(loss)
    
    def _test_one_epoch(self, model, test_data, gamma: float = 0.1):
        model.eval()
        pred = model(test_data.x_dict, test_data.edge_index_dict, test_data['demand', 'measurement'].edge_index)
        target = test_data['demand', 'measurement'].edge_label
        # edge_loss = huber_loss(pred=pred, target=target, delta=0.23) # edge loss: 10^-1
        edge_loss = rmse_loss(pred=pred, target=target)
        # node_u_loss = node_u_recon_loss(test_data, pred) # node loss: 10
        # node_v_loss = node_v_recon_loss(test_data, pred)       
        # node_loss = node_u_loss + node_v_loss
        # node_loss = node_v_loss
        # loss = (1 - gamma) * edge_loss + gamma * node_loss
        loss = edge_loss
        return float(loss)
        # return float(edge_loss), float(node_u_loss), float(node_v_loss), float(loss)
    
    # def _init_train(self, args): 
    #     device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    #     hidden_channels = args.hidden_channels # 64
    #     out_channels = args.out_channels # 64
    #     learning_rate = args.lr # 1e-3
    #     weight_decay = args.weight_decay # 5e-4
    #     epoch_n = args.epoch # 200
    #     period = args.period # 20
    #     dataset = args.data 
    #     model_save = args.save
    #     model_name = args.model
    #     resume = args.resume
    #     seed = args.seed
    #     log_file_name = args.log
    #     gamma = args.gamma
    #     # if dataset exists
    #     assert os.path.exists(dataset), f'Invalid directory {dataset}'
    #     # model save path
    #     if model_save is None: 
    #         model_save = os.path.join('checkpoints', model_name)
    #     if not os.path.exists(model_save): 
    #         os.makedirs(model_save)
    #     self.gamma = gamma
    #     self.period = period
    #     self.model_name = model_name
    #     self.model_save_path = model_save
    #     # seed everything for reproduction
    #     # seed_everything(seed)
    #     seed_torch(seed)
    #     if log_file_name is None: 
    #         log_file_name = '{}-{:%Y-%m-%d-%H}'.format(model_name, datetime.datetime.now())
    #     # log 
    #     logging.basicConfig(filename=os.path.join('logs', log_file_name), 
    #                         format="%(asctime)s | %(levelname)s | %(message)s", 
    #                         level=logging.INFO, 
    #                         filemode='a')
    #     # model & optimizer & data loader
    #     model = Bipartite_link_pred(hidden_channels=hidden_channels, out_channels=out_channels).to(device)
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #     train_loader, test_loader = get_sumo_dataloaders(data_dir=dataset)
    #     # resume?
    #     if resume: 
    #         resume_checkpoint = torch.load(resume)
    #         model.load_state_dict(resume_checkpoint['model'])
    #         optimizer.load_state_dict(resume_checkpoint['optimizer'])
    #         self.start_epoch = resume_checkpoint['epoch'] 
    #         self.end_epoch = resume_checkpoint['epoch'] + epoch_n
    #         self.best = resume_checkpoint['best'] 
    #     else: 
    #         self.start_epoch = 0 
    #         self.end_epoch = epoch_n
    #         self.best = float('inf')
    #     return model, optimizer, train_loader, test_loader # parameters

    def train(self, args):
        # model, optimizer, train_loader, test_loader = self._init_train(args)
        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        hidden_channels = args.hidden_channels # 64
        out_channels = args.out_channels # 64
        learning_rate = args.lr # 1e-3
        weight_decay = args.weight_decay # 5e-4
        epoch_n = args.epoch # 200
        period = args.period # 20
        dataset = args.data 
        model_save = args.save
        model_name = args.model
        resume = args.resume
        seed = args.seed
        log_file_name = args.log
        # gamma = args.gamma
        # if dataset exists
        assert os.path.exists(dataset), f'Invalid directory {dataset}'
        # model save path
        if model_save is None: 
            model_save = os.path.join('checkpoints', model_name)
        if not os.path.exists(model_save): 
            os.makedirs(model_save)
        # self.gamma = gamma
        self.period = period
        self.model_name = model_name
        self.model_save_path = model_save
        # seed everything for reproduction
        # seed_everything(seed)
        seed_torch(seed)
        if log_file_name is None: 
            log_file_name = '{}-{:%Y-%m-%d-%H}'.format(model_name, datetime.datetime.now())
        # log 
        logging.basicConfig(filename=os.path.join('logs', log_file_name), 
                            format="%(asctime)s | %(levelname)s | %(message)s", 
                            level=logging.INFO, 
                            filemode='a')
        # model & optimizer & data loader
        model = Bipartite_link_pred(hidden_channels=hidden_channels, out_channels=out_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_loader, test_loader, _, _ = get_sumo_dataloaders(data_dir=dataset)
        # resume?
        if resume: 
            resume_checkpoint = torch.load(resume)
            model.load_state_dict(resume_checkpoint['model'])
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            self.start_epoch = resume_checkpoint['epoch'] 
            self.end_epoch = resume_checkpoint['epoch'] + epoch_n
            self.best = resume_checkpoint['best'] 
        else: 
            self.start_epoch = 0 
            self.end_epoch = epoch_n
            self.best = float('inf')

        train_loss_list = []
        test_loss_list = []
        for epoch in range(self.start_epoch, self.end_epoch): 
            # train
            # avg_train_edge_loss = 0.0
            # avg_train_node_u_loss = 0.0
            # avg_train_node_v_loss = 0.0
            avg_train_loss = 0.0
            train_sample_n = len(train_loader)
            bar = tqdm(train_loader, desc='Training...'.ljust(20))
            for data in bar:
                train_data = self._prepare_data(data, gcn_norm_threshold=0.02) # NOTE: adjust threshold
                # le, ln_u, ln_v, l = self._train_one_epoch(model, train_data, optimizer, gamma=self.gamma) # NOTE: adjust gamma
                loss = self._train_one_epoch(model, train_data, optimizer)
                bar.set_description(f'Epoch [{epoch+1}/{self.end_epoch}]')
                # bar.set_postfix(edge=le, node_u=ln_u, node_v=ln_v, loss=l)
                bar.set_postfix(loss=loss)
                # avg_train_node_u_loss += ln_u
                # avg_train_node_v_loss += ln_v
                # avg_train_edge_loss += le
                avg_train_loss += loss
            # avg_train_node_u_loss /= train_sample_n
            # avg_train_node_v_loss /= train_sample_n
            # avg_train_edge_loss /= train_sample_n
            avg_train_loss /= train_sample_n
            # test
            # avg_test_edge_loss = 0.0
            # avg_test_node_u_loss = 0.0
            # avg_test_node_v_loss = 0.0
            avg_test_loss = 0.0
            test_sample_n = len(test_loader)
            bar = tqdm(test_loader, desc='Testing...'.ljust(20))
            for data in bar:
                test_data = self._prepare_data(data)
                # le, ln_u, ln_v, l = self._test_one_epoch(model, test_data, gamma=self.gamma)
                loss = self._test_one_epoch(model, test_data)
                bar.set_description(f'Epoch [{epoch+1}/{self.end_epoch}]')
                # bar.set_postfix(edge=le, node_u=ln_u, node_v=ln_v)
                bar.set_postfix(edge=loss)
                # avg_test_node_u_loss += ln_u
                # avg_test_node_v_loss += ln_v
                # avg_test_edge_loss += le
                avg_test_loss += loss
            # avg_test_node_u_loss /= test_sample_n
            # avg_test_node_v_loss /= test_sample_n
            # avg_test_edge_loss /= test_sample_n
            avg_test_loss /= test_sample_n

            train_loss_list.append(avg_train_loss)
            test_loss_list.append(avg_test_loss)
            # logging.info(
            #     '[{}/{}] | train loss={:.4f} | test loss={:.4f} | test edge={:.4f} | test node u={:.4f} | test node v={:.4f}'.format(
            #         epoch+1, self.end_epoch, avg_train_loss, avg_test_loss, avg_test_edge_loss, avg_test_node_u_loss, avg_test_node_v_loss)
            #     )
            logging.info(
                '[{}/{}] | train loss={:.4f} | test loss={:.4f}'.format(
                    epoch+1, self.end_epoch, avg_train_loss, avg_test_loss)
                )
            # node_recon_loss = self.gamma * (avg_test_node_u_loss + avg_test_node_v_loss)
            if avg_test_loss < self.best: 
                self.best = avg_test_loss
                # save model
                self._save_model(cur_epoch=epoch, loss=avg_test_loss, best_epoch=self.best, model=model, 
                                 model_save_path=self.model_save_path, model_name=self.model_name, optimizer=optimizer)
            elif (epoch + 1) % self.period == 0: 
                # save model
                self._save_model(cur_epoch=epoch, loss=avg_test_loss, best_epoch=self.best, model=model, 
                                 model_save_path=self.model_save_path, model_name=self.model_name, optimizer=optimizer)
            # self._save_model(cur_epoch=epoch, loss=avg_test_loss, best_epoch=self.best, model=model, 
            #                     model_save_path=self.model_save_path, model_name=self.model_name, optimizer=optimizer)
        return train_loss_list, test_loss_list

## bigi
# class Trainer_bigi(Trainer_blp):
#     def __init__(self):
#         super(Trainer_bigi, self).__init__()

#     def _prepare_data(self, data):
#         # add reverse edges 
#         data = T.ToUndirected()(data)
#         del data['measurement', 'rev_contributes_to', 'demand'].edge_label

#         # link-level split (8-1-1)
#         train_data, val_data, test_data = T.RandomLinkSplit(
#             num_val  = 0.1, # 10% validation
#             num_test = 0.1, # 10% test
#             neg_sampling_ratio=0.0,
#             edge_types=[('demand', 'contributes_to', 'measurement')],
#             rev_edge_types=[('measurement', 'rev_contributes_to', 'demand')],
#         )(data)
#         return train_data, val_data, test_data

#     def train(self, args):
#         device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
#         hidden_channels = args.hidden_channels # 64
#         out_channels = args.out_channels # 64
#         learning_rate = args.lr # 1e-3
#         weight_decay = args.weight_decay # 5e-4
#         epoch_n = args.epoch # 200
#         period = args.period # 20
#         train_set = args.data 
#         model_save = args.save
#         model_name = args.model
#         resume = args.resume
#         seed = args.seed
#         swap = args.swap
#         log_file_name = args.log
#         assert os.path.exists(train_set), f'Invalid directory {train_set}'
#         # assert os.path.exists(model_save), f'Invalid directory {model_save}'
#         if not os.path.exists(model_save): 
#             os.makedirs(model_save)

#         seed_everything(seed)
#         if log_file_name is None: 
#             log_file_name = '{}-{:%Y-%m-%d-%H}'.format(model_name, datetime.datetime.now())

#         logging.basicConfig(filename=os.path.join('logs', log_file_name), 
#                             format="%(asctime)s | %(levelname)s | %(message)s", 
#                             level=logging.INFO, 
#                             filemode='a')
#         model = Bipartite_link_pred_2hop(hidden_channels=hidden_channels, out_channels=out_channels, layer_n=3).to(device)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#         if resume: 
#             resume_checkpoint = torch.load(resume)
#             model.load_state_dict(resume_checkpoint['model'])
#             optimizer.load_state_dict(resume_checkpoint['optimizer'])
#             start_epoch, end_epoch = resume_checkpoint['epoch'], resume_checkpoint['epoch'] + epoch_n
#             best_epoch = resume_checkpoint['best'] 
#         else: 
#             start_epoch, end_epoch = 0, epoch_n
#             best_epoch = float('inf')

#         train_loader = get_sumo_trainloader(data_dir=train_set, swap_prob=swap)
#         train_loss_list = []
#         test_loss_list = []
#         val_loss_list = []
#         for epoch in range(start_epoch, end_epoch):
#             avg_train_loss = 0.0
#             avg_test_loss = 0.0
#             avg_val_loss = 0.0
#             batch_n = len(train_loader)
#             bar = tqdm(train_loader, desc='Training...'.ljust(20))
#             for data in bar:
#                 train_data, val_data, test_data = self._prepare_data(data)

#                 # train one epoch / test one epoch
#                 loss = self._train_one_epoch(model, train_data, optimizer)
#                 train_loss = self._test_one_epoch(model, train_data)
#                 val_loss = self._test_one_epoch(model, val_data)
#                 test_loss = self._test_one_epoch(model, test_data)
#                 bar.set_description(f'Epoch [{epoch+1}/{end_epoch}]')
#                 bar.set_postfix(loss=loss, train=train_loss, val=val_loss, test=test_loss)
#                 avg_train_loss += train_loss
#                 avg_test_loss += test_loss
#                 avg_val_loss += val_loss
#             avg_train_loss /= batch_n
#             avg_test_loss /= batch_n
#             avg_val_loss /= batch_n
#             train_loss_list.append(avg_train_loss)
#             test_loss_list.append(avg_test_loss)
#             val_loss_list.append(avg_val_loss)
#             logging.info('[{}/{}] train loss={:.4f} | val loss = {:.4f} | test loss = {:.4f}'.format(epoch+1, end_epoch, avg_train_loss, avg_val_loss, avg_test_loss))
#             if avg_test_loss < best_epoch: 
#                 best_epoch = avg_test_loss
#                 # save model
#                 self._save_model(cur_epoch=epoch, test_loss=avg_test_loss, best_epoch=best_epoch, 
#                                  model=model, model_save_path=model_save, model_name=model_name, optimizer=optimizer)
#             elif (epoch + 1) % period == 0: 
#                 # save model
#                 self._save_model(cur_epoch=epoch, test_loss=avg_test_loss, best_epoch=best_epoch, 
#                              model=model, model_save_path=model_save, model_name=model_name, optimizer=optimizer)

class Trainer_blg(Trainer_blp): 
    def __init__(self):
        super(Trainer_blg, self).__init__()
    
    def _train_one_epoch(self, model, train_data, optimizer, edge_weight=None):
        model.train()
        optimizer.zero_grad()
        lower, mid, upper = model(train_data.x_dict, 
                                  train_data.edge_index_dict, 
                                  train_data['demand', 'measurement'].edge_label_index)
        target = train_data['demand', 'measurement'].edge_label
        # assert pred.shape == target.shape
        # loss_1 = rmse_loss(pred=mid, target=target)
        loss_1 = huber_loss(pred=mid, target=target)
        loss_2 = pinball_loss(pred=lower, target=target, tau=0.05)
        loss_3 = pinball_loss(pred=upper, target=target, tau=0.95)
        loss = loss_1 + 0.1 * (loss_2 + loss_3)
        # print(f'huber loss: {float(loss_1)} | lower loss: {float(loss_2)} | upper loss: {float(loss_3)}')
        loss.backward()
        optimizer.step()
        return float(loss)
    
    def _test_one_epoch(self, model, test_data):
        model.eval()
        lower, mid, upper = model(test_data.x_dict, 
                    test_data.edge_index_dict, 
                    test_data['demand', 'measurement'].edge_label_index)
        target = test_data['demand', 'measurement'].edge_label
        loss = rmse_loss(pred=mid, target=target)
        # TODO: check quantile info
        return float(loss)
    
    def train(self, args): 
        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        hidden_channels = args.hidden_channels # 64
        # u_dim = args.u_dim
        # v_dim = args.v_dim
        out_channels = args.out_channels # 64
        learning_rate = args.lr # 1e-3
        weight_decay = args.weight_decay # 5e-4
        epoch_n = args.epoch # 200
        period = args.period # 20
        train_set = args.data 
        model_save = args.save
        model_name = args.model
        resume = args.resume
        seed = args.seed
        swap = args.swap
        log_file_name = args.log
        assert os.path.exists(train_set), f'Invalid directory {train_set}'
        # assert os.path.exists(model_save), f'Invalid directory {model_save}'
        if model_save is None: 
            model_save = os.path.join('checkpoints', model_name)

        if not os.path.exists(model_save): 
            os.makedirs(model_save)

        # seed_everything(seed)
        seed_torch(seed)
        if log_file_name is None: 
            log_file_name = '{}-{:%Y-%m-%d-%H}'.format(model_name, datetime.datetime.now())

        logging.basicConfig(filename=os.path.join('logs', log_file_name), 
                            format="%(asctime)s | %(levelname)s | %(message)s", 
                            level=logging.INFO, 
                            filemode='a')
        # TODO: hard code |u|
        model = Bipartite_LinkQuantileRegression_GAE(num_node_u=2000, hidden_channels=hidden_channels, out_channels=out_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if resume: 
            resume_checkpoint = torch.load(resume)
            model.load_state_dict(resume_checkpoint['model'])
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            start_epoch, end_epoch = resume_checkpoint['epoch'], resume_checkpoint['epoch'] + epoch_n
            best_epoch = resume_checkpoint['best'] 
        else: 
            start_epoch, end_epoch = 0, epoch_n
            best_epoch = float('inf')

        train_loader = get_sumo_trainloader(data_dir=train_set, swap_prob=swap)
        train_loss_list = []
        test_loss_list = []
        val_loss_list = []
        for epoch in range(start_epoch, end_epoch):
            avg_train_loss = 0.0
            avg_test_loss = 0.0
            avg_val_loss = 0.0
            batch_n = len(train_loader)
            bar = tqdm(train_loader, desc='Training...'.ljust(20))
            for data in bar:
                train_data, val_data, test_data = self._prepare_data(data)

                # train one epoch / test one epoch
                loss = self._train_one_epoch(model, train_data, optimizer)
                train_loss = self._test_one_epoch(model, train_data)
                val_loss = self._test_one_epoch(model, val_data)
                test_loss = self._test_one_epoch(model, test_data)
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
            logging.info('[{}/{}] | train loss={:.4f} | val loss={:.4f} | test loss={:.4f}'.format(epoch+1, end_epoch, avg_train_loss, avg_val_loss, avg_test_loss))
            if avg_test_loss < best_epoch: 
                best_epoch = avg_test_loss
                # save model
                self._save_model(cur_epoch=epoch, test_loss=avg_test_loss, best_epoch=best_epoch, 
                                 model=model, model_save_path=model_save, model_name=model_name, optimizer=optimizer)
            elif (epoch + 1) % period == 0: 
                # save model
                self._save_model(cur_epoch=epoch, test_loss=avg_test_loss, best_epoch=best_epoch, 
                             model=model, model_save_path=model_save, model_name=model_name, optimizer=optimizer)
                
# [deprecated]
# def train_bipartite_link_pred(args): 
#     def prepare_data(data: HeteroData):
#         # add reverse edges 
#         data = T.ToUndirected()(data)
#         del data['measurement', 'rev_contributes_to', 'demand'].edge_label

#         # link-level split (8-1-1)
#         train_data, val_data, test_data = T.RandomLinkSplit(
#             num_val  = 0.1, # 10% validation
#             num_test = 0.1, # 10% test
#             neg_sampling_ratio=0.0,
#             edge_types=[('demand', 'contributes_to', 'measurement')],
#             rev_edge_types=[('measurement', 'rev_contributes_to', 'demand')],
#         )(data)

#         # Generate metapaths (v->u->v / u->v->u)
#         metapath_0 = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]     
#         metapath_1 = [('demand', 'contributes_to', 'measurement'), ('measurement', 'rev_contributes_to', 'demand')]     
#         train_data = T.AddMetaPaths(metapaths=[metapath_0])(train_data)

#         # Apply normalization to filter the metapath:
#         _, edge_weight = gcn_norm(
#             train_data['measurement', 'measurement'].edge_index,
#             num_nodes=train_data['measurement'].num_nodes,
#             add_self_loops=False,
#         )
#         edge_index = train_data['measurement', 'measurement'].edge_index[:, edge_weight > 0.01]

#         train_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
#         val_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
#         test_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index

#         # _, edge_weight = gcn_norm(
#         #     train_data['demand', 'demand'].edge_index,
#         #     num_nodes=train_data['demand'].num_nodes,
#         #     add_self_loops=False,
#         # )
#         # edge_index = train_data['demand', 'demand'].edge_index[:, edge_weight > 0.01]

#         # train_data['demand', 'metapath_1', 'demand'].edge_index = edge_index
#         # val_data['demand', 'metapath_1', 'demand'].edge_index = edge_index
#         # test_data['demand', 'metapath_1', 'demand'].edge_index = edge_index

#         # calculate mse weight
#         # edge_labels = train_data[('demand', 'contributes_to', 'measurement')].edge_label
#         # values, counts = np.unique(edge_labels.cpu(), return_counts=True)
#         # weight = max(counts) / counts
#         # weight_dict = {k: v for k, v in zip(values, weight)}
#         weight_dict = None
#         return train_data, val_data, test_data, weight_dict
    
#     def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weight_dict = None): 
#         device_, data_type_ = pred.device, pred.dtype
#         tmp_target_arr = target.cpu().numpy()
#         weight = torch.Tensor([weight_dict[t] for t in tmp_target_arr]).to(device=device_, dtype=data_type_)
#         return (weight * (pred - target).pow(2)).mean().sqrt()

#     def train_one_epoch(model: nn.Module, train_data: HeteroData, optimizer, loss_weight = None):
#         ''' train one epoch ''' 
#         model.train()
#         optimizer.zero_grad()
#         # pred = model(train_data.x_dict, 
#         #             train_data.edge_index_dict, 
#         #             train_data['demand', 'measurement'].edge_label_index, 
#         #             edge_weight = train_data['demand', 'measurement'].edge_attr)
#         pred = model(train_data.x_dict, 
#                     train_data.edge_index_dict, 
#                     train_data['demand', 'measurement'].edge_label_index)
#         target = train_data['demand', 'measurement'].edge_label
#         assert pred.shape == target.shape
#         if loss_weight is None: 
#             loss = F.mse_loss(pred, target).sqrt()
#         else: 
#             loss = weighted_mse_loss(pred=pred, target=target, weight_dict=loss_weight) 
#         loss.backward()
#         optimizer.step()
#         return float(loss)

#     @torch.no_grad()
#     def test_one_epoch(model: nn.Module, test_data: HeteroData):
#         ''' test one epoch '''
#         model.eval()
#         pred = model(test_data.x_dict, 
#                     test_data.edge_index_dict, 
#                     test_data['demand', 'measurement'].edge_label_index)
#         target = test_data['demand', 'measurement'].edge_label
#         assert pred.shape == target.shape 
#         loss = F.mse_loss(pred, target).sqrt()
#         return float(loss)
    
#     device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
#     hidden_channels = args.hidden_channels # 64
#     out_channels = args.out_channels # 64
#     learning_rate = args.lr # 5e-4
#     weight_decay = args.weight_decay # 5e-4
#     epoch_n = args.epoch # 200
#     period = args.period # 20
#     train_set = args.data 
#     model_save = args.save
#     model_name = args.model
#     resume = args.resume
#     seed = args.seed
#     swap = args.swap
#     log_file_name = args.log
#     assert os.path.exists(train_set), f'Invalid directory {train_set}'
#     assert os.path.exists(model_save), f'Invalid directory {model_save}'

#     seed_everything(seed)
#     if log_file_name is None: 
#         log_file_name = '{}-{:%Y-%m-%d-%H}'.format(model_name, datetime.datetime.now())

#     logging.basicConfig(filename=os.path.join('logs', log_file_name), 
#                         format="%(asctime)s | %(levelname)s | %(message)s", 
#                         level=logging.INFO, 
#                         filemode='a')
#     model = Bipartite_link_pred(hidden_channels=hidden_channels, out_channels=out_channels).to(device)
#     # model = Bipartite_link_pred_2hop(hidden_channels=hidden_channels, 
#     #                                  out_channels=out_channels, 
#     #                                  layer_n=2, 
#     #                                  alpha=0.1, 
#     #                                  dropout=0.1).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     if resume: 
#         resume_checkpoint = torch.load(resume)
#         model.load_state_dict(resume_checkpoint['model'])
#         optimizer.load_state_dict(resume_checkpoint['optimizer'])
#         start_epoch, end_epoch = resume_checkpoint['epoch'], resume_checkpoint['epoch'] + epoch_n
#         best_epoch = resume_checkpoint['best'] 
#     else: 
#         start_epoch, end_epoch = 0, epoch_n
#         best_epoch = float('inf')

#     train_loader = get_sumo_trainloader(data_dir=train_set, swap_prob=swap)
#     train_loss_list = []
#     test_loss_list = []
#     val_loss_list = []
#     for epoch in range(start_epoch, end_epoch):
#         avg_train_loss = 0.0
#         avg_test_loss = 0.0
#         avg_val_loss = 0.0
#         batch_n = len(train_loader)
#         bar = tqdm(train_loader, desc='Training...'.ljust(20))
#         for data in bar:
#             # # unidirected graph
#             # data = T.ToUndirected()(data)
#             # del data['measurement', 'rev_contributes_to', 'demand'].edge_label
#             # # link-level split
#             # train_data, val_data, test_data = T.RandomLinkSplit(
#             #     num_val  = 0.1, # 10% validation
#             #     num_test = 0.1, # 10% test
#             #     neg_sampling_ratio=0.0,
#             #     edge_types=[('demand', 'contributes_to', 'measurement')],
#             #     rev_edge_types=[('measurement', 'rev_contributes_to', 'demand')],
#             # )(data)
#             # # Generate the co-occurence matrix of movies<>movies:
#             # metapath = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]
#             # train_data = T.AddMetaPaths(metapaths=[metapath])(train_data)

#             # # Apply normalization to filter the metapath:
#             # _, edge_weight = gcn_norm(
#             #     train_data['measurement', 'measurement'].edge_index,
#             #     num_nodes=train_data['measurement'].num_nodes,
#             #     add_self_loops=False,
#             # )
#             # edge_index = train_data['measurement', 'measurement'].edge_index[:, edge_weight > 0.01]

#             # train_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
#             # val_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
#             # test_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
#             train_data, val_data, test_data, _ = prepare_data(data)

#             # train one epoch / test one epoch
#             loss = train_one_epoch(model, train_data, optimizer)
#             train_loss = test_one_epoch(model, train_data)
#             val_loss = test_one_epoch(model, val_data)
#             test_loss = test_one_epoch(model, test_data)
#             bar.set_description(f'Epoch [{epoch+1}/{end_epoch}]')
#             bar.set_postfix(loss=loss, train=train_loss, val=val_loss, test=test_loss)
#             avg_train_loss += train_loss
#             avg_test_loss += test_loss
#             avg_val_loss += val_loss
#         avg_train_loss /= batch_n
#         avg_test_loss /= batch_n
#         avg_val_loss /= batch_n
#         train_loss_list.append(avg_train_loss)
#         test_loss_list.append(avg_test_loss)
#         val_loss_list.append(avg_val_loss)
#         logging.info('[{}/{}] train loss={:.4f} | val loss = {:.4f} | test loss = {:.4f}'.format(epoch+1, end_epoch, avg_train_loss, avg_val_loss, avg_test_loss))
#         if avg_test_loss < best_epoch: 
#             best_epoch = avg_test_loss
#             # save model
#             checkpoint = {
#                 'epoch': epoch+1, 
#                 'model': model.state_dict(), 
#                 'optimizer': optimizer.state_dict(), 
#                 'best': best_epoch
#             }
#             if model_save: 
#                 torch.save(checkpoint, os.path.join(model_save, f'{model_name}-epoch-{epoch+1}-loss-{best_epoch}.pth'))
#         elif (epoch + 1) % period == 0: 
#             # save model
#             checkpoint = {
#                 'epoch': epoch+1, 
#                 'model': model.state_dict(), 
#                 'optimizer': optimizer.state_dict(),
#                 'best': best_epoch
#             }
#             if model_save: 
#                 torch.save(checkpoint, os.path.join(model_save, f'{model_name}-epoch-{epoch+1}-loss-{avg_test_loss}.pth'))



        

