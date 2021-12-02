# File Name : trainer.py
# Purpose :  Base class for training any model using pytorch
# Creation Date : 11-29-2021
# Last Modified : 
# Created By : vamshi

import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from pytorch_trainer.metric.metric import Metric

def stringify_dict(d):
    """
    If a dict contains callable (functions or classes) values, stringify_dict replaces them with their __name__ attributes.
    Useful for logging the dictionary.
    """
    str_d = {}
    for k,v in d.items():
        if isinstance(v, dict):
            str_d[k] = stringify_dict(v)
        else:
            str_d[k] = v.__name__ if callable(v) else v 
    return str_d

class Trainer:
    def __init__(
        self,
        experiment_name,
        general_options,
        experiment_summary='',
    ):
        self.experiment_name = experiment_name
        self.device = 'cuda:0' if torch.cuda.is_available() and general_options['use_cuda'] else 'cpu'
        ## instantiate network in build_model()
        self.model = None
        ## instantiate optimizer and loss function in train()
        self.optimizer = None
        self.criterion = None
        ## dataloaders
        self.trainloader, self.valloader = None, None

        ## logger
        self.use_tensorboard = False
        if general_options['use_tensorboard'] == True:
            self.use_tensorboard = True
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter('runs/'+self.experiment_name)
            self.logger.add_text('summary', experiment_summary)
    
    def initialize_dataloaders(self, dataloader_fn, **dataloader_kwargs):
        self.trainloader, self.valloader = dataloader_fn(**dataloader_kwargs)

    def build_model(self, network, **network_kwargs):
        """
        Builds the model and load to gpu (if needed). 
        """
        self.model = network(**network_kwargs)
        self.model.to(self.device)
    
    def single_step(self, is_train_step, metric_fn=None):
        """
        Perform one step of training/validation (one iteration of the corresponding dataloader)
        Args:
            is_train_step (Boolean): whether train step or validation
            metric_fn(Function): Metric function that could be run during training or validation on the model predictions. Predictions and target tensors are passed as arguments to the function.
        """
        if is_train_step:
            self.model.train()
            dataloader = self.trainloader
        else:
            self.model.eval()
            dataloader = self.valloader

        running_loss = 0

        for images, labels in tqdm(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if is_train_step:
                self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            if is_train_step:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            if metric_fn is not None:
                metric_fn(outputs.detach(), labels.detach())

            del loss, outputs, images, labels
        running_loss = running_loss/len(dataloader)

        return running_loss
    
    def train(self, 
                epochs,
                loss_fn,
                optimizer,
                loss_fn_kwargs = {},
                optimizer_kwargs = {}, 
                lr_scheduler = None, 
                lr_scheduler_kwargs={}, 
                metric = None,
                metric_kwargs = {},
                save_best=True,
                save_location='',
                save_name='',
                continue_training_saved_model=None,
            ):
        if self.trainloader is None or self.valloader is None:
            print("Initialize the dataloders before training!")
            return

        if save_best:
            if save_location == '' or save_name == '':
                print("Provide the name and location of the model to be saved!")
                return
            ## create directory if it doesn't exist
            if not os.path.exists(save_location):
                os.makedirs(save_location)
            save_model_path = os.path.join(save_location, save_name+'.pth')
            
            best_val_loss = torch.finfo(torch.float32).max  ## to save best model
        
        ## initialize the loss function
        self.criterion = loss_fn(**loss_fn_kwargs)

        ## optimize only unfrozen layers (i.e requires_grad == True)
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), **optimizer_kwargs)
        if lr_scheduler is not None:
            scheduler = lr_scheduler(self.optimizer, **lr_scheduler_kwargs)

        start_epoch = 0
        if continue_training_saved_model is not None:
            saved_model = torch.load(continue_training_saved_model)
            self.model.load_state_dict(saved_model['state_dict'])
            self.optimizer.load_state_dict(saved_model['optimizer'])
            start_epoch = saved_model['epoch']

        train_metric_fn, val_metric_fn = None, None
        if metric is not None:
            train_metric = metric(**metric_kwargs)
            val_metric = metric(**metric_kwargs)
            train_metric_fn = train_metric.add
            val_metric_fn = val_metric.add

        for e in tqdm(range(start_epoch, start_epoch+epochs)):
            if metric is not None:
                train_metric.reset()
                val_metric.reset()

            train_loss = self.single_step(is_train_step=True, metric_fn=train_metric_fn)
            print("{}. Training loss = {:.3f}".format(e, train_loss))
            if metric is not None:
                train_metric_value = train_metric.value()
                print("{}. Train Metric - {}: {}".format(e, metric.__name__, train_metric_value))
            val_loss = self.single_step(is_train_step=False, metric_fn=val_metric_fn)
            print("{}. Validation loss = {:.3f}".format(e, val_loss))
            if metric is not None:
                val_metric_value = val_metric.value()
                print("{}. Val Metric - {}: {}".format(e, metric.__name__, val_metric_value))

            if lr_scheduler is not None:
                scheduler.step()

            ## logging
            if self.use_tensorboard:
                self.logger.add_scalars("losses", {'train': train_loss,
                                                   'val': val_loss}, e)
                if metric is not None:
                    if isinstance(train_metric_value, tuple):
                        ## @TODO make this pretty
                        train_metric_value = train_metric_value[0]
                        val_metric_value = val_metric_value[0]

                    self.logger.add_scalars(metric.__name__, {'train':train_metric_value ,
                                                   'val': val_metric_value}, e)

                for param_name, param_w in self.model.named_parameters():
                    self.logger.add_histogram(param_name, param_w, e)

            ## save best model
            if save_best == True:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save( { 
                                    'epoch': e,
                                    'state_dict': self.model.to('cpu').state_dict(),
                                    'train_loss': train_loss,
                                    'val_loss': val_loss,
                                    'optimizer': self.optimizer.state_dict()
                                }, save_model_path )
                    self.model.to(self.device)
        ## end epoch for

        ## logging
        if self.use_tensorboard:
            images, labels = next(iter(self.trainloader))
            images = images.to(self.device)
            self.logger.add_graph(self.model, images)

            training_hparams = {
                'epochs': start_epoch+epochs,
                'loss_fn': loss_fn,
                'optimizer': optimizer,
                **loss_fn_kwargs,
                **optimizer_kwargs
            }
            if lr_scheduler is not None:
                training_hparams.update({'lr_scheduler':lr_scheduler}, **lr_scheduler_kwargs)

            metrics = {
                    'Train_loss' : train_loss,
                    'Val_loss' : val_loss,
            }
            if metric is not None:
                metrics.update({ 'Train_'+metric.__name__ : train_metric_value, 'Val_'+metric.__name__ : val_metric_value })

            self.logger.add_hparams(hparam_dict = stringify_dict(training_hparams), metric_dict = metrics)
    ## end train()

def test():
    class DummyNet(nn.Module):
        def __init__(self, n_channels):
            super().__init__()
            self.d = nn.Linear(n_channels, 10)
        def forward(self, x):
            return self.d(x)
    
    def get_dataloaders(**kwargs):
        """
        Sample pytorch dataloader
        """
        trainloader = torch.utils.data.DataLoader(**kwargs)
        valloader = torch.utils.data.DataLoader(**kwargs)
        
        return trainloader,valloader

    ### General hyper-parameters
    general_options = {
        'use_cuda' :          True,         # use GPU ?
        'use_tensorboard' :   True          # Use Tensorboard for saving hparams and metrics ?
    }

    ### Training hyper-parameters
    trainer_args = {
        'epochs' : 50, 
        'loss_fn' : nn.CrossEntropyLoss, ## must be of type nn.Module
        'optimizer' : optim.SGD, 
        'loss_fn_kwargs': {},
        'optimizer_kwargs' : {'lr' : 0.001, 'momentum' : 0.9, 'weight_decay' : 5e-4},
        'lr_scheduler' : torch.optim.lr_scheduler.CosineAnnealingLR, 
        'lr_scheduler_kwargs' : {'T_max' : 200},
        'metric': Metric,  ## must be of type metric.Metric or its derived
        'metric_kwargs': {},
        'save_best' : True,
        'save_location' : './saved_models',
        'save_name' : 'test_trainer_base',
        'continue_training_saved_model' : None,
    }

    dataloader_args = {
        'batch_size' : 32,
        'num_workers': 12
    }

    network_args = {
        'n_channels': 3
    }

    experiment_summary = 'test'
            
    trainer = Trainer('test', general_options, experiment_summary=experiment_summary)
    trainer.initialize_dataloaders(get_dataloaders, **dataloader_args)
    trainer.build_model(DummyNet, **network_args)
    trainer.train(**trainer_args)