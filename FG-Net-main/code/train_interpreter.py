import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' # see issue #152

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

import json
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from data_utils import heatmap2au, MyDataset
from model_utils import load_psp_standalone, latent_to_image, prepare_model
from models.interpreter import pyramid_interpreter


def heatmap2au_new(heatmap, alpha=7):
    #logits = logits.clamp(-4,4)
    #x = torch.tanh(heatmap)
    x =  heatmap
    #print("heatmap.shape",heatmap.shape)
    x_flat = x.view(x.size(0),x.size(1),-1)
    lse = (1/alpha) * torch.log(torch.mean(torch.exp(alpha * x_flat), dim=2))
    logits = 4 * lse
    #logits = logits.clamp(-4,4)

    return logits

def au_logits2pred(au_logits, threshold = 0.5):
    binary_pred = (au_logits > threshold).astype(int)
    return binary_pred

def prepare(args):
    pspencoder = load_psp_standalone(args['style_encoder_path'], 'cuda')
    g_all, upsamplers = prepare_model(args)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    train_list = pd.read_csv(args['train_csv'])
    val_list = pd.read_csv(args['test_csv'])
    test_list = pd.read_csv(args['test_csv'])

    return train_list, val_list, test_list, g_all, upsamplers, pspencoder, transform


def val(interpreter, val_loader, pspencoder, g_all, upsamplers, args):
    interpreter.eval()
    pspencoder.eval()
    g_all.eval()


    with torch.no_grad():
        pred_list = []
        gt_list = []
        f1_list = []
        acc_list, per_list, rec_list = [], [], []


        for i, (images, labels) in enumerate(tqdm(val_loader)):
            # get latent code
            latent_codes = pspencoder(images)
            latent_codes = g_all.style(latent_codes.reshape(latent_codes.shape[0]*latent_codes.shape[1], latent_codes.shape[2])).reshape(latent_codes.shape)
            
            # get stylegan features
            features = latent_to_image(g_all, upsamplers, latent_codes, upsample=False, args=args)

            heatmaps_pred = interpreter(features)
            heatmaps_pred = heatmaps_pred[:,:labels.size(1),:,:]
            
            #heatmaps_pred = torch.clamp(heatmaps_pred, min=-1., max=1.)
            #labels_pred = torch.mean(heatmaps_pred,dim=(2,3))
            # AU_logit between [-4,4]
            #labels_pred = heatmap2au_new(heatmaps_pred)
            labels_pred = torch.mean(heatmaps_pred,dim=(2,3))
            labels_pred = torch.sigmoid(labels_pred)
            #sigmoid to [0,1]
            pred_list.append(labels_pred.detach().cpu())
            gt_list.append(labels.detach().cpu())

        pred_list = torch.cat(pred_list, dim=0).numpy()
        gt_list = torch.cat(gt_list, dim=0).numpy()

        binary_pred = au_logits2pred(pred_list)
        #binary to {0,1}, calculate performance

        for j in range(args['num_labels']):
            f1_list.append(100.0*f1_score(gt_list[:, j], binary_pred[:, j]))
            acc_list.append(100.0*accuracy_score(gt_list[:, j], binary_pred[:, j]))
            per_list.append(100.0*precision_score(gt_list[:, j], binary_pred[:, j]))
            rec_list.append(100.0*recall_score(gt_list[:, j], binary_pred[:, j]))

        return sum(f1_list)/len(f1_list), sum(acc_list)/len(acc_list), sum(per_list)/len(per_list), sum(rec_list)/len(rec_list), f1_list , acc_list, per_list, rec_list


def main(args):
    print(args['c_rate'])
    print('Prepare model')
    train_list, val_list, test_list, g_all, upsamplers, pspencoder, transform = prepare(args)
    #print(torch.cuda.memory_allocated() / 1024**3, "G allocated")
    #print(torch.cuda.memory_reserved() / 1024**3, "G reserved")

    print('Prepare data')
    #print(len(train_list))
    train_data = MyDataset(train_list, transform, 'train', args)
    #print(len(train_data))
    val_data = MyDataset(val_list, transform, 'val', args)
    val_loader = DataLoader(dataset=val_data, batch_size=args['batch_size'],
                            shuffle=False, collate_fn=val_data.collate_fn)
    test_data = MyDataset(test_list, transform, 'val', args)
    test_loader = DataLoader(dataset=test_data, batch_size=2*args['batch_size'],
                            shuffle=False, collate_fn=test_data.collate_fn)

    print('Start training')

    if args['dataset'] == 'BP4D':
        aus = [1,2,4,6,7,10,12,14,15,17,23,24]
    elif args['dataset'] == 'DISFA':
        aus = [1,2,4,6,9,12,25,26]
    
    interpreter = pyramid_interpreter(32, args['dropout']).cuda()
    interpreter.init_weights()
    criterion_r = nn.MSELoss()
    if args['dataset'] == 'DISFA':
        #criterion_c = nn.BCELoss()
        criterion_c = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([19.10667077, 22.17753366, 5.56268499, 11.66718311, 22.90169925,  6.76298143, 2.60896074, 10.34258216]).cuda())
    elif args['dataset'] == 'BP4D':
        criterion_c = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([3.73368584, 4.85313347, 3.93861213, 1.16948652, 0.82129433, 0.68288573, 0.77909724, 1.14730746, 4.90397298, 1.91280627, 5.04496047, 5.60518265]).cuda())
        
    optimizer = optim.AdamW(list(interpreter.parameters())
                            +list(g_all.parameters())
                            +list(pspencoder.parameters()),
                            lr=args['learning_rate'], weight_decay=args['weight_decay'])

    total_loss, total_sample = 0., 0
    total_loss_r, total_loss_c = 0.,0.
    best_f1, best_f1_list = 0., []

    for epoch in range(args['num_epochs']):
        interpreter.train()
        g_all.train()
        pspencoder.train()
        train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, collate_fn=train_data.collate_fn)
        args['interval'] = min(args['interval'], len(train_loader))
        for i, (images, labels, heatmaps) in enumerate(tqdm(train_loader, total=args['interval'])):
            #print('label',labels)
            #print(torch.max(heatmaps), torch.min(heatmaps))
            #logit_label = heatmap2au_new(heatmaps)
            #print(logit_label)
            #print(torch.sigmoid(logit_label))
            if i >= args['interval']:
                break
            #print(labels.size())
            batch_size = images.shape[0]
            # get latent code
            latent_codes = pspencoder(images)
            '''
            find_nan = torch.isnan(latent_codes).any()
            if find_nan:
                print('psp encoder nan')
                exit(1)
            '''
            
            latent_codes = g_all.style(latent_codes.reshape(latent_codes.shape[0]*latent_codes.shape[1], latent_codes.shape[2])).reshape(latent_codes.shape)
            '''
            find_nan = torch.isnan(latent_codes).any()
            if find_nan:
                print('g_all nan')
                exit(1)
            '''
            # get stylegan features
            features = latent_to_image(g_all, upsamplers, latent_codes, upsample=False, args=args)
            '''
            for i, feat in enumerate(features):
                #when no upsampling, feature is a list
                if torch.isnan(feat).any():
                    print(f"Batch {i}: features[{i}] contains NaN")
                    print(f"  shape: {feat.shape}, min: {feat.min().item()}, max: {feat.max().item()}")
                    exit(1)
            '''
            
            heatmaps_pred = interpreter(features)
            #heatmaps_pred = torch.clamp(heatmaps_pred, min=-1., max=1.)
            
            '''
            find_nan = torch.isnan(heatmaps_pred).any()
            if find_nan:
                print('heatmaps_pred nan')
                exit(1)
            '''

            #heatmaps_pred = torch.clamp(heatmaps_pred, min=-4., max=4.)

            loss = 0.
            loss_c, loss_r = 0., 0.
            #print(torch.max(heatmaps_pred),torch.min(heatmaps_pred))
            '''
            for j in range(args['num_labels']):
                loss_r += 0.8*criterion_r(heatmaps_pred[:,j,:,:], heatmaps[:,j,:,:])
                if torch.isnan(loss_r):
                    print('regression bad')
                    print(torch.max(heatmaps_pred),torch.min(heatmaps_pred))
                    exit(1)
                loss_r /= len(aus)
            '''
            heatmaps_pred = heatmaps_pred[:,:args['num_labels'],:,:]
            loss_r = criterion_r(heatmaps_pred,heatmaps)
            logit_pred = torch.mean(heatmaps_pred, dim=(2,3))
            #logit_pred = heatmap2au_new(heatmaps_pred)
            loss_c = criterion_c(logit_pred, labels)

            #print(heatmap2au(heatmaps_pred).dtype)
            #print(criterion_c(heatmap2au_new(heatmaps_pred), labels.float()))


            #logit_pred = torch.mean(heatmaps_pred, dim=(2,3))
            #loss_c +=  criterion_c(logit_pred, labels.float())
            #print(loss_c)
            #loss_c += 0.2 * criterion_c(heatmap2au_new(heatmaps_pred),labels.float())
            #print(heatmap2au_new(heatmaps_pred))
            #if torch.isnan(loss_c):
            #    print('classification bad')
            #    print(heatmap2au_new(heatmaps_pred))
            #    exit(1)
            loss = args['c_rate'] * loss_c + (1-args['c_rate']) * loss_r
            #print('regression loss',loss)

            #print(criterion_c(heatmap2au_new(heatmaps_pred), labels))(torch.sigmoid(heatmap2au(heatmaps_pred)), labels)
            #print('sum loss',loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(interpreter.parameters())
                                            +list(g_all.parameters())
                                            +list(pspencoder.parameters()), 0.1)
            optimizer.step()

            total_loss += loss.item()*batch_size
            total_loss_c += loss_c.item()*batch_size
            total_loss_r += loss_r.item()*batch_size
            total_sample += batch_size

        avg_loss = total_loss / total_sample
        print('** Epoch {}/{} loss {:.6f} **'.format(epoch+1, args['num_epochs'], avg_loss))
        print('loss c',total_loss_c/total_sample)
        print('loss r',total_loss_r/total_sample)

        val_f1, val_acc, val_per, ver_rec, val_f1_list, val_acc_list, val_per_list, val_rec_list = val(interpreter, val_loader, pspencoder, g_all, upsamplers, args)
        print('Val avg F1: {:.2f}'.format(val_f1))
        print('Val avg accuracy: {:.2f}'.format(val_acc))
        print('Val avg percision: {:.2f}'.format(val_per))
        print('Val avg recall: {:.2f}'.format(ver_rec))
        for j in range(args['num_labels']):
            print('AU {}: {:.2f}'.format(aus[j], val_f1_list[j]), end=' ')
        print('')
        for j in range(args['num_labels']):
            print('AU {}: {:.2f}'.format(aus[j], val_acc_list[j]), end=' ')
        print('')  
        for j in range(args['num_labels']):
            print('AU {}: {:.2f}'.format(aus[j], val_per_list[j]), end=' ')
        print('')
        for j in range(args['num_labels']):
            print('AU {}: {:.2f}'.format(aus[j], val_rec_list[j]), end=' ')
        print('')

        if best_f1 < val_f1:
            best_f1 = val_f1
            best_f1_list = val_f1_list
            model_path = os.path.join(args['exp_dir'], 'model.pth')
            print('save to:', model_path)
            torch.save({'interpreter': interpreter.state_dict(),
                        'g_all': g_all.state_dict(),
                        'pspencoder': pspencoder.state_dict()}, model_path)

    checkpoint = torch.load(os.path.join(args['exp_dir'], 'model.pth'))
    interpreter.load_state_dict(checkpoint['interpreter'])
    g_all.load_state_dict(checkpoint['g_all'])
    pspencoder.load_state_dict(checkpoint['pspencoder'])
    val_f1, val_acc, val_per, ver_rec, val_f1_list, val_acc_list, val_per_list, val_rec_list = val(interpreter, test_loader, pspencoder, g_all, upsamplers, args)
    print('Val avg F1: {:.2f}'.format(val_f1))
    print('Val avg accuracy: {:.2f}'.format(val_acc))
    print('Val avg percision: {:.2f}'.format(val_per))
    print('Val avg recall: {:.2f}'.format(ver_rec))
    for j in range(args['num_labels']):
        print('AU {}: {:.2f}'.format(aus[j], val_f1_list[j]), end=' ')
    print('')
    for j in range(args['num_labels']):
        print('AU {}: {:.2f}'.format(aus[j], val_acc_list[j]), end=' ')
    print('')  
    for j in range(args['num_labels']):
        print('AU {}: {:.2f}'.format(aus[j], val_per_list[j]), end=' ')
    print('')
    for j in range(args['num_labels']):
        print('AU {}: {:.2f}'.format(aus[j], val_rec_list[j]), end=' ')
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #print(torch.cuda.memory_allocated() / 1024**3, "G allocated")
    #print(torch.cuda.memory_reserved() / 1024**3, "G reserved")
    parser.add_argument('--exp', type=str)
    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))
    print('Opt', opts)
    os.makedirs(opts['exp_dir'], exist_ok=True)
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    torch.manual_seed(opts['seed'])
    main(opts)
 
