# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from data_loader import AKSDataset
import ResNet
import time


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
                        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = [0, 1]
    best_all = np.zeros([config.split_num, 4])
    for split in range(config.split_num):
        if config.model_name == 'ResNet':
            print('The current model is ' + config.model_name)
            model = ResNet.resnet50(pretrained=True)

            model = model.to(device)

            model = nn.DataParallel(model, device_ids=gpu_ids)
        else:
            print('Model name not recognized. Using ResNet as default.')
            model = ResNet.resnet50(pretrained=True)
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=gpu_ids)
    

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
        criterion = nn.MSELoss().to(device)

        

        print(
            '*******************************************************************************************************************************************************')
        print('Using ' + str(split + 1) + '-th split.')

        transformations_train = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor(), \
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), \
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
        if config.database == 'SJTU':
            images_dir = config.key_frame_path
            datainfo_train = 'database/sjtu_data_info/train_' + str(split + 1) + '.csv'
            datainfo_test = 'database/sjtu_data_info/test_' + str(split + 1) + '.csv'
        
            trainset = AKSDataset(images_dir, datainfo_train, transformations_train,
                                  crop_size=config.crop_size,
                                  key_frame_num=config.key_frame_num)
            testset = AKSDataset(images_dir, datainfo_test, transformations_test,
                                 crop_size=config.crop_size,
                                 key_frame_num=config.key_frame_num)
        elif config.database == 'WPC':
            images_dir = config.key_frame_path
            datainfo_train = 'database/wpc_data_info/train_' + str(split + 1) + '.csv'
            datainfo_test = 'database/wpc_data_info/test_' + str(split + 1) + '.csv'
       
            trainset = AKSDataset(images_dir, datainfo_train, transformations_train,
                                  crop_size=config.crop_size,
                                  key_frame_num=config.key_frame_num)
            testset = AKSDataset(images_dir, datainfo_test, transformations_test,
                                 crop_size=config.crop_size,
                                 key_frame_num=config.key_frame_num)

        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                  shuffle=False, num_workers=config.num_workers)

        best_test_criterion = -1  # SROCC min
        best = np.zeros(4)

        n_test = len(testset)


        print('Starting training:')

        for epoch in range(config.epochs):

            model.train()
            start = time.time()
            batch_losses = []
            batch_losses_each_disp = []
            for i, (video, labels, _) in enumerate(train_loader):
                video = video.to(device)
                labels = labels.to(device)
                outputs = model(video)
                optimizer.zero_grad()
                loss = criterion(outputs.float(), labels.float())
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())

                loss.backward()
                optimizer.step()

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))
            end = time.time()
            print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end - start))


            model.eval()
            y_output_train = []
            y_train = []
          
            
            # Test
            model.eval()
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)
            # do validation after each epoch
            with torch.no_grad():
                for i, (video, labels, _) in enumerate(test_loader):
                    video = video.to(device)
                    labels = labels.to(device)
                    y_test[i] = labels.item()
                    outputs = model(video)
                    y_output[i] = outputs.item()

                    loss = criterion(outputs.float(), labels.float())
                    batch_losses.append(loss.item())
                    batch_losses_each_disp.append(loss.item())

                avg_loss = sum(batch_losses) / (len(testset))
                print('Epoch %d averaged testing loss: %.4f' % (epoch + 1, avg_loss))
                y_output_logistic = fit_function(y_test, y_output)
                test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
                test_SROCC = stats.spearmanr(y_output, y_test)[0]
                test_RMSE = np.sqrt(((y_output_logistic - y_test) ** 2).mean())
                test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]
                print(
                    "Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC,
                                                                                                test_PLCC, test_RMSE))

                if test_SROCC > best_test_criterion:
                    print("Update best model using best_val_criterion ")
                    # Save model with the requested naming format
                    save_filename = f"{config.database}_fold_{split}_best.pth"
                    save_path = os.path.join(config.ckpt_path, save_filename)
                    torch.save(model.state_dict(), save_path)
                    best[0:4] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
                    best_test_criterion = test_SROCC  # update best val SROCC
                    # best_all[split, :] = best
                    print("Update the best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(
                        test_SROCC, test_KROCC, test_PLCC, test_RMSE))

                
                print(
                    '-------------------------------------------------------------------------------------------------------------------')
        best_all[split, :] = best
        print('Training completed.')
        print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1],
                                                                                                   best[2], best[3]))
        print(
            '*************************************************************************************************************************')

    performance = np.mean(best_all, 0)
    print(
        '*************************************************************************************************************************')
    print("The mean performance: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(performance[0],
                                                                                              performance[1],
                                                                                              performance[2],
                                                                                              performance[3]))
    print(
        '*************************************************************************************************************************')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--conv_base_lr', type=float, default=3e-7)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--split_num', type=int, default=9)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--key_frame_num', type=int, default=5)
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_ids', type=list, default=[0,1])
    parser.add_argument('--key_frame_path', type=str, default='')
    config = parser.parse_args()

    main(config)
