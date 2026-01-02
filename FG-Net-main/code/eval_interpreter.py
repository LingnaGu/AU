import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' # see issue #152

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import json
import argparse
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from data_utils import heatmap2au, MyDataset
from model_utils import latent_to_image, prepare_model, load_psp_standalone
from models.interpreter import pyramid_interpreter
import numpy as np


def heatmap2au_new(heatmap, alpha=7):
    x = heatmap
    #print("heatmap.shape",heatmap.shape)
    x_flat = x.view(x.size(0),x.size(1),-1)
    lse = (1/alpha) * torch.log(torch.mean(torch.exp(alpha * x_flat), dim=2))
    logits = 5 * lse
    logits = logits.clamp(-4,4)

    return logits

def au_logits2pred(au_logits, threshold = 0.5):
    binary_pred = (au_logits > threshold).astype(int)
    return binary_pred

def prepare(args):
	g_all, upsamplers = prepare_model(args)

	pspencoder = load_psp_standalone(args['style_encoder_path'], 'cuda')

	transform = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
	])

	test_list = pd.read_csv(args['test_csv'])

	return test_list, g_all, upsamplers, pspencoder, transform


def val(interpreter, val_loader, pspencoder, g_all, upsamplers, args):
    interpreter.eval()
    pspencoder.eval()
    g_all.eval()

    with torch.no_grad():
        pred_list = []
        gt_list = []
        raw_logit_list = []
        f1_list, acc_list, per_list, rec_list = [], [], [], []

        for (images, labels) in tqdm(val_loader):
            images = images.cuda()
            # get latent code
            latent_codes = pspencoder(images)
            latent_codes = g_all.style(latent_codes.reshape(latent_codes.shape[0]*latent_codes.shape[1], latent_codes.shape[2])).reshape(latent_codes.shape)
            # get stylegan features
            features = latent_to_image(g_all, upsamplers, latent_codes, upsample=False, args=args)

            heatmaps_pred = interpreter(features)[:,:labels.size(1),:,:]
            
            labels_pred = torch.mean(heatmaps_pred, dim=(2,3))
            raw_logits = labels_pred
            labels_pred = torch.sigmoid(labels_pred)
            
            # 直接收集数据，不需要复杂的转换
            pred_list.append(labels_pred.detach().cpu())
            raw_logit_list.append(raw_logits.detach().cpu()) 
            gt_list.append(labels.detach().cpu())

        # 统一使用torch.cat处理，保持维度一致性
        pred_tensor = torch.cat(pred_list, dim=0)
        raw_logit_tensor = torch.cat(raw_logit_list, dim=0)
        gt_tensor = torch.cat(gt_list, dim=0)
        
        # 转换为numpy
        pred_array = pred_tensor.numpy()
        raw_logits_array = raw_logit_tensor.numpy()
        gt_array = gt_tensor.numpy()
        
        # 计算binary predictions
        binary_pred = au_logits2pred(pred_array)
        
        # 保存数据
        predictions_data = {
            'raw_logits': raw_logits_array,        # shape: (n_samples, n_aus)
            'probabilities': pred_array,           # shape: (n_samples, n_aus)
            'ground_truth': gt_array               # shape: (n_samples, n_aus)
        }
        
        # Save to npy file
        save_path = args['checkpoint_path'].replace('model.pth', 'predictions.npy')
        np.save(save_path, predictions_data)
        print(f"Predictions saved to {save_path}")
        
        # 打印形状检查
        print(f"Raw logits shape: {raw_logits_array.shape}")
        print(f"Probabilities shape: {pred_array.shape}")
        print(f"Ground truth shape: {gt_array.shape}")
        print(f"Binary pred shape: {binary_pred.shape}")
        
        # 计算每个AU的指标
        n_aus = gt_array.shape[1]
        for j in range(n_aus):
            # 确保使用zero_division参数避免除零错误
            f1 = f1_score(gt_array[:, j], binary_pred[:, j], zero_division=0)
            acc = accuracy_score(gt_array[:, j], binary_pred[:, j])
            per = precision_score(gt_array[:, j], binary_pred[:, j], zero_division=0)
            rec = recall_score(gt_array[:, j], binary_pred[:, j], zero_division=0)
            
            f1_list.append(100.0 * f1)
            acc_list.append(100.0 * acc)
            per_list.append(100.0 * per)
            rec_list.append(100.0 * rec)
            
            # 打印每个AU的结果
            print(f"AU{j+1}: F1={f1_list[-1]:.2f}, Acc={acc_list[-1]:.2f}, "
                  f"Precision={per_list[-1]:.2f}, Recall={rec_list[-1]:.2f}")
        
        # 计算平均指标
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
        avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0
        avg_per = sum(per_list) / len(per_list) if per_list else 0
        avg_rec = sum(rec_list) / len(rec_list) if rec_list else 0
        
        return avg_f1, avg_acc, avg_per, avg_rec, f1_list, acc_list, per_list, rec_list


def main(args):
	print('Prepare model')
	val_list, g_all, upsamplers, pspencoder, transform = prepare(args)

	'''
	if 'bp4d' in args['checkpoint_path']:
		num_labels = 12
	else:
		num_labels = 8
	'''
	num_labels = 32 # bug in model checkpoint

	interpreter = pyramid_interpreter(num_labels, 0.1).cuda()

	checkpoint = torch.load(args['checkpoint_path'])
	interpreter.load_state_dict(checkpoint['interpreter'])
	g_all.load_state_dict(checkpoint['g_all'])
	pspencoder.load_state_dict(checkpoint['pspencoder'])

	print('Prepare data')
	val_data = MyDataset(val_list, transform, 'test', args)
	val_loader = DataLoader(dataset=val_data, batch_size=args['batch_size'],
							shuffle=False, collate_fn=val_data.collate_fn)

	print('Start evaluation')
	#aus = [1,2,4,6,12]
	#aus = [1,2,4,6,9,12,25,26]
	aus = [1,2,4,6,7,10,12,14,15,17,23,24]
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--exp', type=str)
	args = parser.parse_args()
	opts = json.load(open(args.exp, 'r'))
	print('Opt', opts)

	main(opts)
