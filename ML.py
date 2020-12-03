import torch
import numpy as np
import os, argparse, json
import wandb
from sklearn.metrics import precision_recall_fscore_support

from models import TheModel
from datasets import data_loader

def train(config, model, data, results, outputs):
        
    model.train()
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduling
    def lr_lambda(e):
        if e < 4*config['epochs']/10:
            return config['learning_rate']
        elif 4*config['epochs']/10 <= e < config['epochs']/2:
            return config['learning_rate'] * 0.1
        elif config['epochs']/2 <= e < 9*config['epochs']/10:
            return config['learning_rate']
        elif e >= 9*config['epochs']/10:
            return config['learning_rate'] * 0.1
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(config['epochs']):
        results[epoch] = {}
        # do the training
        results[epoch]['train'], outputs['train'] = training(config, model, data, 'train', optimizer)


        # evaluation on valid and possibly test        
        for portion in list(results['best'].keys()):
            if portion != 'train':
                results[epoch][portion], outputs[portion] = evaluate(config, model, data, portion)

        # update valid and possibly test best if valid is the best
        if results[epoch]['valid']['f1'] > results['best']['valid']['best f1']:
            torch.save(model.state_dict(), config['train_results_dir']+'model_state.pt')
            for portion in list(results['best'].keys()):
                if portion != 'train':
                    results['best'][portion].update(dict(zip(results['best'][portion].keys(), results[epoch][portion].values())))
                    analyze(config, outputs, portion)
                    
        
        wandb.log(results[epoch])
        wandb.run.summary.update(results['best'])
        json.dump(results, open(config['results_dir']+'config.json', 'w'), indent=4)
    
    return results
        



def training(config, model, dataset, portion, optimizer):
    """
    performs one epoch training loop over all data
    """
    counter = 0
    for data in dataset[portion]:
        optimizer.zero_grad()
        outputs = model(data)
        
        loss = torch.tensor(0.0)
        loss.backward()
        optimizer.step()
        print("batch: %d loss: %.4f\r" % (counter,loss), end="")
        counter += 1
    
    f1, p, r, tp, tn, fp, fn = compute_metrics()
    results = {'f1': f1, 'precision': p, 'recall': r, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    return results, outputs



def evaluate(config, model, data, portion):
    """
    This function runs the model over valid and/or test set
    Returns f1, precision, accuracy, and the model outputs
    """
    model_eval = model
    model_eval.eval()
    with torch.no_grad():
        outputs = model(data[portion])
        results = compute_metrics()
        return results, outputs




def test(config, model, data, results, outputs, portion):
    """
    Loads the best trained model and evaluates it on test set.
    Also analyzes the outputs of the model.
    This function can be also written in the if inside main.
    """
    results['test-only'] = {}
    # load the model
    results['test-only'][portion], outputs[portion] = evaluate(config, model, data, 'test')
    analyze(config, outputs, 'test')



def analyze(config, outputs, portion):
    """
    Save some plots and csv files to config['results_dir']+'/'+portion+'/'
    Does not return anything.
    """

def compute_metrics():
    """
    Returns f1 score, precision, recall, TP, TN, FP, and FN
    """
    metrics = {'f1': f1, 'precision': p, 'recall': r, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    return metrics


if __name__ == "__main__":
    
    config = load_configs()
    device = setup_gpu(config['gpu_num'])

    # load data
    train_loader, valid_loader, test_loader = data_loader(config['batch_size'], config['label_dim'], config['random_seed'])
    data = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    config.update({'train split':len(train_loader.dataset), 'valid split':len(valid_loader.dataset), 'test split':len(test_loader.dataset)})

    model = TheModel(
                    image_embedding_size=1024, latent_dim=config['feature_dim'], label_dim=config['label_dim'],
                    activation=config['activation_func'], ablation=config['ablation'],
                    device=device, attention=config['attention'])

    if config['wandb_track'] == 1:
        import wandb
        from torch.utils.tensorboard import SummaryWriter
        wandb.init(project='ML Template', name=config['results_dir'], sync_tensorboard=True)
        wandb.config.update(config)
        wandb.config.codedir = os.path.basename(os.getcwd())
        tb_writer = SummaryWriter(log_dir=wandb.run.dir)
        wandb.watch(model, log="all")

    print('--------- Summary of the data ---------')
    print('train data: ', len(train_loader.dataset))
    print('valid data: ', len(valid_loader.dataset))
    print('test data: ', len(test_loader.dataset))
    print('all data: ', len(train_loader.dataset)+len(valid_loader.dataset)+len(test_loader.dataset))
    print('--------- End of Summary of the data ---------')

    # If pre-trained model exist, load it to continue training 
    if os.path.exists(config['train_results_dir']+'model_state.pt'):
        print('Loading pretrained networks ...')
        model.load_state_dict(torch.load(config['train_results_dir']+'model_state.pt'))
    else:         
        print('Starting from scratch to train networks.')

    model.to(device)

    results, outputs = initialize_result_keeper(config)

    if config['eval_mode'] == 'train' or config['eval_mode'] == 'train-test':
        # train-test: evaluate on test set on each epoch
        # train: only evaluate on valid set. Test once at the end
        print('Training the model!')
        results = train(config, model, data, results, outputs)
        # TODO: if best result of valid is better than other experiments then perform test: test()
    elif config['eval_mode'] == 'test':
        print('Evaluating the model on test data!')
        test(config, model, data, results, outputs, 'test')



def initialize_result_keeper(config):
    results = {
        'best': {
            'train': {'best f1': -1.0, 'best precision': -1.0, 'best recall': -1.0},
            'valid': {'best f1': -1.0, 'best precision': -1.0, 'best recall': -1.0},
        },              
    } # stores the metircs in the form of: results[epoch][portion][metric]

    outputs = {
        'train': {},
        'valid': {},
    } # stores the outputs of the model in the form of: outputs[portion] 
    
    # TODO: load the results if exists

    if config['eval_mode'] == 'train-test' or config['eval_mode'] == 'test':
        results['best'].update({'test': {'best f1': -1.0, 'best precision': -1.0, 'best recall': -1.0}})
        outputs.update({'test': {}})
    
    return results, outputs


def setup_gpu(gpu_num=0):
    device_name = 'cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    return device


def load_configs():
    # returns a dictionary of configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_track', default=1, type=int)
    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--experiment_name', default='Random', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--label_dim', default=300, type=int)
    parser.add_argument('--feature_dim', default=100, type=int)
    parser.add_argument('--prediction_thresh', default=0.50, type=float)
    parser.add_argument('--ablation', default='Ours', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--attention', default='attention', type=str)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--eval_mode', default='train-test', type=str, help='whether to test or just train. train-test, train, test')
    parser.add_argument('--results_dir', default='./results/', type=str)

    args = parser.parse_args()
    
    return vars(args)