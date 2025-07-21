import argparse
from visual import plot_historical_loss
from train import Trainer_blp, Trainer_blg, Trainer_blp_node_recon, Trainer_blp_structured_node_recon

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type = int, default = 64, help = 'number of hidden channels (encoder)')
    parser.add_argument('--out_channels', type = int, default = 128, help = 'number of output channels (encoder)')
    parser.add_argument('--lr', type = float, default = 5e-3, help = 'learning rate') # test 5e-3
    parser.add_argument('--weight_decay', type = float, default = 1e-5, help = 'adam weight decay')
    parser.add_argument('--epoch', type = int, default = 200, help = 'number of training epochs')
    parser.add_argument('--period', type = int, default = 100, help = 'save model every n epochs')
    parser.add_argument('--data',  type = str, default = 'sim_dataset_v2', help = 'training data folder')
    parser.add_argument('--seed',  type = int, default = 2025, help = 'training reproductivity')    
    parser.add_argument('--model', type = str, default = 'BipartiteLinkPred', help = 'model name')        
    # parser.add_argument('--swap', type = float, default = 0.0, help = 'probability of feature swaping')
    # parser.add_argument('--gamma', type = float, default = 1e-4, help = 'loss weight')
    parser.add_argument('--resume', type = str, help = 'load model from checkpoints')
    parser.add_argument('--save',  type = str, help = 'checkpoint save path') 
    parser.add_argument('--log', type = str, help = 'log file name') 
    args = parser.parse_args()
    # trainer = Trainer_blg()
    # trainer = Trainer_blp_node_recon()
    # train_loss, val_loss, test_loss, node_loss = trainer.train(args)
    # plot_historical_loss(x=range(args.epoch), y_list=[train_loss, val_loss, test_loss], label_list=['train', 'val', 'test'], savefig="loss.png")
    # plot_historical_loss(x=range(args.epoch), y_list=[node_loss], label_list=['node'], savefig="node.png")
    trainer = Trainer_blp_structured_node_recon()
    loss_train, loss_test = trainer.train(args)
    plot_historical_loss(x=range(args.epoch), y_list=[loss_train, loss_test], label_list=['train', 'test'], savefig="loss.png")