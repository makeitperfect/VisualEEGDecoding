import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os,sys
import time
import json
import random
import logging
import argparse
import copy
root_dir =  os.path.dirname(os.path.abspath(__file__))
print(root_dir)
sys.path.append(root_dir)
import models

import tqdm
import torch.nn.functional as F
from einops import rearrange
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import scipy.signal as signal

def butter_bandpass(lowcut, highcut, fs, order=5):
  
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

set_seed(2025)


def create_logger(logger_file_path):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) 

    file_handler = logging.FileHandler(final_log_file,encoding="utf-8")  
    console_handler = logging.StreamHandler() 


    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  
    console_handler.setFormatter(formatter)  
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class Pair_dataset(Dataset):
    def __init__(self,eeg_data,blur_features,index_dict,use_filter = False,low_freq = 0.1, high_freq = 50.0,select_period = [0,250]):
        super().__init__()

        self.blur_features = blur_features
        self.eeg_data = eeg_data
        self.index_dict = index_dict
        self.selected_period = select_period

        if use_filter:
            self.eeg_data = butter_bandpass_filter(self.eeg_data, low_freq, high_freq, fs=250, order=4)
            self.eeg_data = torch.from_numpy(self.eeg_data).float()


    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, index):
            x       = self.eeg_data[index][:,self.selected_period[0]:self.selected_period[1]].float()
            x_key   = self.index_dict[index]
  
            sample  = {
                'eeg': x,
                'x_key':  x_key,
                'l_1':    self.blur_features['1'][x_key].float(),
                'l_3':    self.blur_features['3'][x_key].float(),
                'l_9':    self.blur_features['9'][x_key].float(),
                'l_15':   self.blur_features['15'][x_key].float(),
                'l_21':   self.blur_features['21'][x_key].float(),
                'l_27':   self.blur_features['27'][x_key].float(),
                'l_33':   self.blur_features['33'][x_key].float(),
                'l_39':   self.blur_features['39'][x_key].float(),
                'l_45':   self.blur_features['45'][x_key].float(),
                'l_51':   self.blur_features['51'][x_key].float(),
                'l_57':   self.blur_features['57'][x_key].float(),
                'l_63':   self.blur_features['63'][x_key].float(),
            }

            return sample
    

def get_dataset(base_path,sub,cross_sub,use_filter = False,low_freq = 0.1, high_freq = 50.0,select_period = [0,250]):

    # load blurring img feature
    train_blur_feature_path = os.path.join(base_path,'Image_feature','UniformBlur',"UniBlur_RN50_train.pt")
    train_blur_feature = torch.load(train_blur_feature_path,weights_only=False)

    test_blur_feature_path = os.path.join(base_path,'Image_feature','UniformBlur',"UniBlur_RN50_test.pt")
    test_blur_feature = torch.load(test_blur_feature_path,weights_only=False)

    eeg_data = []


    if cross_sub == False:
        train_data_path = os.path.join(base_path,'Preprocessed_data','sub-{:02}'.format(sub),'train.pt')   
        loaded_data = torch.load(train_data_path,weights_only=False)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])

        eeg_data    = loaded_data['eeg'].mean(axis=1)
        index_dict  = np.array(loaded_data['img'])

        # random split validation data
        random_index     = np.random.permutation(eeg_data.shape[0])
        print(len(index_dict),len(random_index))

        train_eeg_data   = eeg_data[random_index[:int(eeg_data.shape[0]*0.95)]]
        train_index_dict = index_dict[random_index[:int(eeg_data.shape[0]*0.95)]]


        val_eeg_data   = eeg_data[random_index[int(eeg_data.shape[0]*0.95):]]
        val_index_dict = index_dict[random_index[int(eeg_data.shape[0]*0.95):]]

        train_dataset  =  Pair_dataset(train_eeg_data,train_blur_feature,train_index_dict,use_filter,low_freq,high_freq,select_period)
        val_dataset    =  Pair_dataset(val_eeg_data,train_blur_feature,val_index_dict,use_filter,low_freq,high_freq,select_period)

        test_data_path = os.path.join(base_path,'Preprocessed_data','sub-{:02}'.format(sub),'test.pt')   
        loaded_data = torch.load(test_data_path,weights_only=False)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])
        test_eeg_data    = loaded_data['eeg'].mean(axis=1)
        test_index_dict  = loaded_data['img']

        test_dataset = Pair_dataset(test_eeg_data,test_blur_feature,test_index_dict,use_filter,low_freq,high_freq,select_period)

    else:
   
        train_eeg_data = []
        val_eeg_data   = []
        test_eeg_data  = []

        train_index_dict = []
        val_index_dict     = []
        test_index_dict  = []

        for i in range(1,5):
            if i == sub:
                data_path   = os.path.join(base_path,'Preprocessed_data','sub-{:02}'.format(i),'test.pt')   
                loaded_data = torch.load(data_path,weights_only=False)
                loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])

                test_eeg_data    = loaded_data['eeg'].mean(axis=1)
                test_index_dict  = np.array(loaded_data['img'])

            else:
                data_path = os.path.join(base_path,'Preprocessed_data','sub-{:02}'.format(i),'train.pt')   
                loaded_data = torch.load(data_path,weights_only=False)
                loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])
                eeg_data    = loaded_data['eeg'].mean(axis=1)

                random_index = np.random.permutation(eeg_data.shape[0])
                train_eeg_data.append(eeg_data[random_index[:int(eeg_data.shape[0]*0.95)]])
                train_index_dict.extend(np.array(loaded_data['img'])[random_index[:int(eeg_data.shape[0]*0.95)]])


                val_eeg_data.append(eeg_data[random_index[int(eeg_data.shape[0]*0.95):]])
                val_index_dict.extend(np.array(loaded_data['img'])[random_index[int(eeg_data.shape[0]*0.95):]])

        train_eeg_data = torch.cat(train_eeg_data,dim=0)
        val_eeg_data   = torch.cat(val_eeg_data,dim=0)

        train_dataset =  Pair_dataset(train_eeg_data,train_blur_feature,train_index_dict,use_filter,low_freq,high_freq,select_period)
        val_dataset   = Pair_dataset(val_eeg_data,train_blur_feature,val_index_dict,use_filter,low_freq,high_freq,select_period)
        test_dataset  = Pair_dataset(test_eeg_data,test_blur_feature,test_index_dict,use_filter,low_freq,high_freq,select_period)

    return  train_dataset,val_dataset,test_dataset



class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
       
    def compute_ranking_weights(self,loss_list):
        sorted_indices = torch.argsort(loss_list)
        weights = torch.zeros_like(loss_list)
        for i, idx in enumerate(sorted_indices):
            weights[idx] = 1 / (i + 1)
        return weights


    def forward(self, eeg_features, img_features, logit_scale):
        device = eeg_features.device
        logits_per_eeg = logit_scale * eeg_features @ img_features.T

        num_logits = logits_per_eeg.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        eeg_loss = F.cross_entropy(logits_per_eeg, labels, reduction='none')

        logits_per_img = logit_scale * img_features @ eeg_features.T
        num_logits = logits_per_img.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        image_loss = F.cross_entropy(logits_per_img, labels, reduction='none')
        return eeg_loss,image_loss, None

def get_test_accu(model,prarams,test_dataloader,device):

    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            teeg = data['eeg'].to(device)
            # img_list = [data[k].to(device) for k in params['blur_level']] 
            img_list = torch.cat([data[k][:,None].to(device) for k in params['blur_level']],1)

            tfea= model(teeg)
            tfea  = tfea/tfea.norm(dim=-1, keepdim=True)
            embed = model.get_image_feature(img_list)
            embed  = embed    / embed.norm(dim=-1, keepdim=True)
            similarity = tfea @ embed.transpose(-1,-2)
            _, indices = similarity.topk(5)
            indices = indices.cpu().detach()
            label = torch.arange(0,teeg.shape[0])

            top1 += (label[:,None] == indices[:, :1]).sum()
            top3 += (label[:,None].expand(-1,3) == indices[:, :3]).any(1).sum()
            top5 += (label[:,None].expand(-1,5) == indices).any(1).sum()
            total += teeg.shape[0]

        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)
        print(top1,total)
    return top1_acc, top3_acc, top5_acc




def train(params,logger): 
    set_seed(params['seed'])

    base_path = params['data_path']
    print(base_path)

    train_dataset,val_dataset,test_dataset = get_dataset(base_path,params['sub'],params['cross_subject'],params['use_filter'],params['low_freq'], params['high_freq'],params['select_period'])

    print('Dataset size',len(train_dataset),len(val_dataset),len(test_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = params['train_batch_size'],shuffle=True,drop_last=True)
    val_loader   = torch.utils.data.DataLoader(dataset=val_dataset, batch_size = len(val_dataset), shuffle=False)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = len(test_dataset), shuffle=False)


    model = models.__dict__[str(params['net_name'])](271,1024,params['select_period'][1]-params['select_period'][0]).to(device)

    optimizier   = torch.optim.AdamW(model.parameters(),lr = args.lr) 
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizier, T_max = CFG.epoch, eta_min = 1e-6)
    criterion = ClipLoss()

    best_test_accu = 0
    saved_metirc = {}

    loss_points = []
    accu_points_top1 = []
    accu_points_top3 = []
    accu_points_top5 = []

    for e in range(params['epoch']):
        model.train()
        step = 0
        all_loss = 0
        for data in tqdm.tqdm(train_loader):
            optimizier.zero_grad()
            x = data['eeg'].to(device)
            img_list = torch.cat([data[k][:,None].to(device) for k in params['blur_level']],1)

            if params['mixup'] == True:
                # Mixup
                rand_index = np.random.permutation(x.shape[0])
                lam   = np.random.beta(0.2, 0.2)
                x   = lam * x + (1 - lam) * x[rand_index]
                img_list = lam * img_list + (1 - lam)*img_list[rand_index]

            eeg_f  = model(x)
            # eeg_f  = eeg_f    / eeg_f.norm(dim=-1, keepdim=True)

            img_f  = model.get_image_feature(img_list)
            img_f  = img_f    / img_f.norm(dim=-1, keepdim=True)

            logit_scale = 1

            eeg_loss, img_loss, external_loss = criterion(eeg_f, img_f,logit_scale)
            loss = (eeg_loss.mean() + img_loss.mean() ) / 2

            loss.backward()
            optimizier.step()

            step += 1
            all_loss += loss.detach().cpu()

        # scheduler.step()
        train_loss = all_loss/step
        loss_points.append(train_loss.detach().cpu().numpy())

        model.eval()
        # top1_acc, top3_acc, top5_acc = get_test_accu(model,test_loader,test_center,device)
        # test_top1_acc, test_top3_acc, test_top5_acc = get_test_accu(model,test_loader,test_embed,device)
        val_top1_acc, val_top3_acc, val_top5_acc = get_test_accu(model,params,val_loader,device)

        test_top1_acc, test_top3_acc, test_top5_acc = get_test_accu(model,params,test_loader,device)
        accu_points_top1.append(test_top1_acc)
        accu_points_top3.append(test_top3_acc)
        accu_points_top5.append(test_top5_acc)
        
        if best_test_accu < test_top1_acc:
            best_model = copy.deepcopy(model.state_dict())
            best_test_accu = val_top1_acc
            saved_metirc['epoch'] = e
            saved_metirc['train_loss'] = train_loss.item()
            saved_metirc['test_top1_acc'] = test_top1_acc
            saved_metirc['test_top3_acc'] = test_top3_acc
            saved_metirc['test_top5_acc'] = test_top5_acc

            # torch.save(model.state_dict(), os.path.join(base_path, f'{CFG.net_name}_best.pth'))
            # print('TEST CENTER: epoch:{},train_loss:{:.3},top_1_acc:{:.3},top_3_acc:{:.3},top_5_acc:{:.3}'.format(e,train_loss,top1_acc, top3_acc, top5_acc))  

        logger.info('VAL  EMBED:  epoch:{},train_loss:{:.3},top_1_acc:{:.3},top_3_acc:{:.3},top_5_acc:{:.3}'.format(e,train_loss,val_top1_acc,  val_top3_acc, val_top5_acc))  
        logger.info('TEST EMBED:  epoch:{},train_loss:{:.3},top_1_acc:{:.3},top_3_acc:{:.3},top_5_acc:{:.3}'.format(e,train_loss,test_top1_acc, test_top3_acc, test_top5_acc))  


    saved_metirc['train_loss_points'] = loss_points
    saved_metirc['test_top1_acc_points'] = accu_points_top1
    saved_metirc['test_top3_acc_points'] = accu_points_top3
    saved_metirc['test_top5_acc_points'] = accu_points_top5

    if params['save_feature']:
        # 保存特征用于图片生成
        torch.save(best_model, os.path.join(params['save_path'], '{}_subject{}_best.pth'.format(params["net_name"],params['sub'])))
        logger.info('保存特征用于图片生成')
        model.load_state_dict(best_model)
        model.eval()


        model_output_features_train = {}
        image_list_feature_train    = {}
        for data in tqdm.tqdm(train_dataset):
      
            x   = data['eeg'][None].to(device)
            img_list = torch.cat([data[k][None,None].to(device) for k in params['blur_level']],1)

            key = data['x_key']
            model_output = model(x)
            img_f  = model.get_image_feature(img_list)

            model_output_features_train[key] = model_output[0].detach().cpu()
            image_list_feature_train[key]  = img_f[0].detach().cpu()

        if not os.path.exists(os.path.join(root_dir,'saved_files', 'model_feature',os.path.basename(__file__),params['net_name'])): 
            os.makedirs(os.path.join(root_dir,'saved_files', 'model_feature',os.path.basename(__file__),params['net_name']))

        torch.save({'eeg_output':model_output_features_train,'img_output':image_list_feature_train}, os.path.join(root_dir,'saved_files', 'model_feature',os.path.basename(__file__),params['net_name'],'RN50_features_train_sub_{}_2.pt'.format(params['sub'])))


        model_output_features_test = {}
        image_list_feature_test    = {}

        for data in test_dataset:
            x = data['eeg'][None].to(device)
            img_list = torch.cat([data[k][None,None].to(device) for k in params['blur_level']],1)

            key = data['x_key']
            model_output = model(x)
            img_f  = model.get_image_feature(img_list)

            model_output_features_test[key] = model_output[0].detach().cpu()
            image_list_feature_test[key]  = img_f[0].detach().cpu()
            

        torch.save({'eeg_output':model_output_features_test,'img_output':image_list_feature_test}, os.path.join(root_dir,'saved_files', 'model_feature',os.path.basename(__file__),params['net_name'],'RN50_features_test_sub_{}_2.pt'.format(params['sub'])))

    return saved_metirc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default = 'Brain_Visual_Encoder_MEG')
    parser.add_argument('--epoch', type=int, default=100)
    # parser.add_argument('--epoch', type=int,  default=50)
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size',  type=int, default=200)
    parser.add_argument('--lr',    type=float, default=0.001)
    parser.add_argument('--mixup', type=bool,  default=False)

    parser.add_argument('--blur_level', type=list, default = ['l_1','l_3','l_9','l_15','l_21','l_27','l_33','l_39','l_45','l_51','l_57','l_63'])

    parser.add_argument('--use_filter', type=bool,  default=False)
    parser.add_argument('--low_freq',   type=float, default=0.1)
    parser.add_argument('--high_freq',  type=float, default=50.0)
    
    parser.add_argument('--save_feature',  type=bool, default=False)
    parser.add_argument('--cross_subject', type=bool, default=False)


    args = parser.parse_args()
    params = {
        'net_name': args.net_name,
        'epoch': args.epoch,
        'train_batch_size': args.train_batch_size,
        'test_batch_size':  args.test_batch_size,
        'lr':    args.lr,
        'mixup': args.mixup,
        'blur_level': args.blur_level,
        'use_filter': args.use_filter,
        'high_freq': args.high_freq,
        'save_feature':args.save_feature,

        'select_period': [0,201],
        'save_path':os.path.join(root_dir,'logs',os.path.basename(__file__),args.net_name,time.strftime('%Y-%m-%d-%H-%M')),
        'data_path': f"D:\\Dataset\\things-meg",
        'cross_subject':args.cross_subject
    }

    


    if not os.path.exists(params['save_path']):
        os.makedirs(params['save_path'])

    with open(os.path.join(params['save_path'],"config.json"), "w") as outfiles:  
        json.dump(params, outfiles, indent = 4)

    logger = create_logger(params['save_path'])


    all_metrics = []


    for sub in range(1,5):
        for seed in range(21,26):
            params['seed'] = seed
            params['sub'] = sub
            all_metrics.append(train(params,logger))

    import matplotlib.pyplot as plt
    
    figs, axs = plt.subplots(len(all_metrics), 4, figsize=(24,4*len(all_metrics)))

    for i, metrics in enumerate(all_metrics):
        axs[i, 0].plot(metrics['train_loss_points'], label='train_loss')
        axs[i, 0].set_title('Subject {}, Train Loss'.format(i + 1))
        axs[i, 0].legend()

        axs[i, 1].plot(metrics['test_top1_acc_points'], label='test_top1_acc')
        axs[i, 1].set_title('Subject {}, Test Top1 Acc'.format(i + 1))
        axs[i, 1].legend()

        axs[i, 2].plot(metrics['test_top3_acc_points'], label='test_top3_acc')
        axs[i, 2].set_title('Subject {}, Test Top3 Acc'.format(i + 1))
        axs[i, 2].legend()

        axs[i, 3].plot(metrics['test_top5_acc_points'], label='test_top5_acc')
        axs[i, 3].set_title('Subject {}, Test Top5 Acc'.format(i + 1))
        axs[i, 3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(params['save_path'], 'train_result.png'))

    logger.info('所有被试的训练结果：')
    top_1_accs = [m['test_top1_acc'] for m in all_metrics]
    top_3_accs = [m['test_top3_acc'] for m in all_metrics]
    top_5_accs = [m['test_top5_acc'] for m in all_metrics]
    logger.info('top_1_accs: {}'.format(top_1_accs))
    logger.info('top_3_accs: {}'.format(top_3_accs))
    logger.info('top_5_accs: {}'.format(top_5_accs))
    logger.info('平均 top_1_acc: {:.3f}'.format(np.mean(top_1_accs)))
    logger.info('平均 top_3_acc: {:.3f}'.format(np.mean(top_3_accs)))
    logger.info('平均 top_5_acc: {:.3f}'.format(np.mean(top_5_accs)))

    all_metrics = pd.DataFrame(all_metrics)
    all_metrics.to_csv(os.path.join(params['save_path'],"all_metrics.csv"),index=False)
