import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from dataset import load, loadPositive, loadNegative, getlabel
import func as func
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import  pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from tsne_plot import  plot
from similarity_score import  SimilarityScore

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')


# generator
class Gen(nn.Module):
    def __init__(self, nz=config.BERT_FEATURES, map_dim=config.HIDDEN_LAYER, hideen_layer=config.HIDDEN_LAYER, leak=0.2, num_gpu=1):
        super(Gen, self).__init__()
        self.num_gpu = num_gpu
        self.fc = nn.Sequential(
            nn.Linear(nz, hideen_layer), nn.LeakyReLU(leak),
            nn.Linear(hideen_layer , map_dim ),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim , map_dim ),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim , nz), nn.Tanh())

    def forward(self, z):
        if isinstance(z.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            out = nn.parallel.data_parallel(self.fc, z, range(self.num_gpu))
        else:
            out = self.fc(z)
        return out

# class label generator
class GenY(nn.Module):
    def __init__(self, nz=config.BERT_FEATURES, num_gpu=1):
        super(GenY, self).__init__()
        self.num_gpu = num_gpu
        map_dim = config.HIDDEN_LAYER
        hideen_layer = config.HIDDEN_LAYER
        leak = 0.2

        self.fc = nn.Sequential(
            nn.Linear(nz, hideen_layer), nn.LeakyReLU(leak),
            nn.Linear(hideen_layer , map_dim ),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim , map_dim ),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim , 1), nn.Sigmoid())

    def forward(self, z):
        if isinstance(z.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            out = nn.parallel.data_parallel(self.fc, z, range(self.num_gpu))
        else:
            out = self.fc(z)
        return out.squeeze()


# positive discriminator
class PDis(nn.Module):
    def __init__(self, nz=config.BERT_FEATURES, num_gpu=1):
        super(PDis, self).__init__()
        self.num_gpu = num_gpu
        map_dim = config.HIDDEN_LAYER
        hideen_layer = config.HIDDEN_LAYER
        leak = 0.2
        self.fc = nn.Sequential(
            nn.Linear(nz, hideen_layer), nn.LeakyReLU(leak),
            nn.Linear(hideen_layer, map_dim),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim, map_dim),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim, 1), nn.Sigmoid())
        #self.fc = nn.Sequential(nn.Linear(nz, 1), nn.Sigmoid())

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            out = nn.parallel.data_parallel(self.fc, x, range(self.num_gpu))
        else:
            out = self.fc(x)
        return out.squeeze()


# negative discriminator
class NDis(nn.Module):
    def __init__(self, nz=config.BERT_FEATURES, num_gpu=1):
        super(NDis, self).__init__()
        self.num_gpu = num_gpu
        map_dim = config.HIDDEN_LAYER
        hideen_layer = config.HIDDEN_LAYER
        leak = 0.2
        self.fc = nn.Sequential(
            nn.Linear(nz, hideen_layer), nn.LeakyReLU(leak),
            nn.Linear(hideen_layer, map_dim),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim, map_dim),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim, 1), nn.Sigmoid())

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            out = nn.parallel.data_parallel(self.fc, x, range(self.num_gpu))
        else:
            out = self.fc(x)
        return out.squeeze()
        
# unlabelled discriminator
class UDis(nn.Module):
    def __init__(self, nz=config.BERT_FEATURES, num_gpu=1):
        super(UDis, self).__init__()
        self.num_gpu = num_gpu
        map_dim = config.HIDDEN_LAYER
        hideen_layer = config.HIDDEN_LAYER
        leak = 0.2
        self.fc = nn.Sequential(
            nn.Linear(nz, hideen_layer), nn.LeakyReLU(leak),
            nn.Linear(hideen_layer, map_dim),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim, map_dim),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim, 1), nn.Sigmoid())

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            out = nn.parallel.data_parallel(self.fc, x, range(self.num_gpu))
        else:
            out = self.fc(x)
        return out.squeeze()
        
# class label generator
class YDis(nn.Module):
    def __init__(self, nz=config.BERT_FEATURES, num_gpu=1):
        super(YDis, self).__init__()
        self.num_gpu = num_gpu
        map_dim = config.HIDDEN_LAYER
        hideen_layer = config.HIDDEN_LAYER
        leak = 0.2
        self.fc = nn.Sequential(
            nn.Linear(nz, hideen_layer), nn.LeakyReLU(leak),
            nn.Linear(hideen_layer, map_dim),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim, map_dim),
            nn.LeakyReLU(leak),
            nn.Linear(map_dim, 1), nn.Sigmoid())

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            out = nn.parallel.data_parallel(self.fc, x, range(self.num_gpu))
        else:
            out = self.fc(x)
        return out.squeeze()


# ArrayGAN class 
class ArrayGAN(object):
    def __init__(self, config):
        self.P_CLASS = config.P_CLASS
        self.N_CLASS = config.N_CLASS
        self.P_RATIO = config.P_RATIO
        self.CHANEL_DIM = config.CHANEL_DIM
        self.Z_DIM = config.Z_DIM
        self.MAP_DIM = config.MAP_DIM
        self.IMAGE_SIZE = config.IMAGE_SIZE
        self.LEAK = config.LEAK

        self.NUM_GPU = config.NUM_GPU
        self.EPOCHS = config.EPOCHS
        self.P_WEIGHT = config.P_WEIGHT
        self.PU_WEIGHT = config.PU_WEIGHT
        self.N_WEIGHT = config.N_WEIGHT
        self.NU_WEIGHT = config.NU_WEIGHT

        self.LR = config.LR
        self.BETA1 = config.BETA1
        self.BETA2 = config.BETA2
        self.SOFT_LABEL = config.SOFT_LABEL

        self.IMAGE_SET = config.IMAGE_SET
        self.IMAGE_PATH = config.IMAGE_PATH
        self.NUM_IMAGES = config.NUM_IMAGES
        self.BATCH_SIZE = config.BATCH_SIZE
        self.WORKERS = config.WORKERS

        self.VERBOSE = config.VERBOSE
        self.LOG_STEP = config.LOG_STEP
        self.LOG_PATH = config.LOG_PATH
        self.SAMPLE_STEP = config.SAMPLE_STEP
        self.SAMPLE_SIZE = config.SAMPLE_SIZE
        self.SAMPLE_PATH = config.SAMPLE_PATH
        self.MODEL_PATH = config.MODEL_PATH

        ## Introducing BERT
        self.BERT_FEATURES = config.BERT_FEATURES
        self.HIDDEN_LAYER = config.HIDDEN_LAYER
        #self.make_sub_dir()

        self.train_positive = None
        self.train_negative = None
        self.train = None
        self.label = None



        self.get_loader()

        self.p_gen = None
        self.n_gen = None
        self.p_dis = None
        self.n_dis = None
        self.u_dis = None
        self.y_gen = None
        self.y_dis = None


        self.p_gen_optim = None
        self.n_gen_optim = None
        self.p_dis_optim = None
        self.n_dis_optim = None
        self.u_dis_optim = None
        self.y_gen_optim = None
        self.y_dis_optim = None

        self.y_gen_loss_list = []
        self.n_gen_loss_list = []
        self.p_gen_loss_list = []

        self.pd_r_loss_list = []
        self.pd_pf_loss_list = []

        self.nd_r_loss_list = []
        self.nd_nf_loss_list = []

        self.ud_r_loss_list = []
        self.ud_pf_loss_list = []
        self.ud_nf_loss_list = []

        self.pd_pf_loss_avg_list = []
        self.pd_r_loss_avg_list = []
        self.nd_r_loss_avg_list = []
        self.nd_nf_loss_loss_list = []
        self.y_gen_loss_avg_list = []
        self.n_gen_loss_avg_list = []
        self.p_gen_loss_avg_list = []
        self.ud_r_loss_avg_list = []
        self.ud_pf_loss_avg_list = []
        self.ud_nf_loss_avg_list = []

        self.precisionList = []
        self.recallList = []
        self.f1scoreList = []

        self.build_model()

    def save_model(self, epoch):
        p_gen_path = os.path.join(self.MODEL_PATH, 'p_gen-%d.pkl' % epoch)
        torch.save(self.p_gen.state_dict(), p_gen_path)

        n_gen_path = os.path.join(self.MODEL_PATH, 'n_gen-%d.pkl' % epoch)
        torch.save(self.n_gen.state_dict(), n_gen_path)

        y_gen_path = os.path.join(self.MODEL_PATH, 'y_gen-%d.pkl' % epoch)
        torch.save(self.y_gen.state_dict(), y_gen_path)

    
    # Creating noise
    def create_noise(self, batch_size, features):
        return torch.rand(batch_size, features)

    # Building the model
    def build_model(self):
        self.p_gen = Gen(nz=self.BERT_FEATURES, map_dim=self.MAP_DIM, hideen_layer=self.HIDDEN_LAYER,
                         leak=self.LEAK, num_gpu=self.NUM_GPU)
        self.n_gen = Gen(nz=self.BERT_FEATURES, map_dim=self.MAP_DIM, hideen_layer=self.HIDDEN_LAYER,
                         leak=self.LEAK, num_gpu=self.NUM_GPU)

        self.p_dis = PDis(nz=self.BERT_FEATURES, num_gpu=self.NUM_GPU)
        self.n_dis = NDis(nz=self.BERT_FEATURES, num_gpu=self.NUM_GPU)
        self.u_dis = UDis(nz=self.BERT_FEATURES, num_gpu=self.NUM_GPU)

        self.y_gen = GenY(nz=self.BERT_FEATURES, num_gpu=self.NUM_GPU)
        self.y_dis = YDis(nz=self.BERT_FEATURES, num_gpu=self.NUM_GPU)


        if torch.cuda.is_available():
            print('in cuda')
            self.p_gen.cuda()
            self.n_gen.cuda()

            self.p_dis.cuda()
            self.n_dis.cuda()
            self.u_dis.cuda()

            self.y_gen.cuda()
            self.y_dis.cuda()
    ## Adam optimizer
        self.p_gen_optim = optim.Adam(self.p_gen.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2))
        self.n_gen_optim = optim.Adam(self.n_gen.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2))
        self.p_dis_optim = optim.Adam(self.p_dis.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2))
        self.n_dis_optim = optim.Adam(self.n_dis.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2))
        self.u_dis_optim = optim.Adam(self.u_dis.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2))

        self.y_gen_optim = optim.Adam(self.y_gen.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2))
        self.y_dis_optim = optim.Adam(self.y_dis.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2))

        
## Reseting grad
    def reset_grad(self):
        self.p_gen.zero_grad()
        self.n_gen.zero_grad()
        self.p_dis.zero_grad()
        self.n_dis.zero_grad()
        self.u_dis.zero_grad()
        self.y_gen.zero_grad()
        self.y_dis.zero_grad()

    def get_loader(self):
        self.train_positive = loadPositive()
        self.train_negative = loadNegative()
        self.train = load()
        self.label = getlabel()


## Training model
    def train_model(self):
        fz = torch.randn(self.SAMPLE_SIZE, self.Z_DIM)
        fz_var = func.to_variable(fz, volatile=True)
        self.SAMPLE_STEP = 1
        real_label = func.to_variable(torch.ones(self.BATCH_SIZE))
        fake_label = func.to_variable(torch.zeros(self.BATCH_SIZE))

        psum = 0
        nsum = 0
        simScore = SimilarityScore()
        for epoch in range(self.EPOCHS):
            if (epoch + 1) % self.SAMPLE_STEP == 0:
                print('EPOCH [%d]' % (epoch + 1))





            for bi, data in tqdm(enumerate(zip(self.train_positive, self.train_negative, self.train, self.label)),
                                 total=int(len(self.train_positive) / self.train_positive.batch_size)):

                pos_data, neg_data, gt_data, label = data

                label = func.to_variable(label)

                pos_data = model.encode(pos_data)
                gt_data = model.encode(gt_data)
                neg_data = model.encode(neg_data)

                pos_data = torch.from_numpy(pos_data)
                gt_data = torch.from_numpy(gt_data)
                neg_data = torch.from_numpy(neg_data)

                reals_p = func.to_variable(pos_data)
                reals_n = func.to_variable(neg_data)
                u_reals = func.to_variable(gt_data)

                # ========== update D networks ==========
                # ===== p_dis_net =====
                # ===== p_dis_net =====

                self.reset_grad()
                pz = torch.randn(self.BATCH_SIZE, self.BERT_FEATURES)
                pz_var = func.to_variable(pz, volatile=True)
                p_fakes = func.to_variable(self.p_gen(pz_var).data)

                # train p_dis with real data
                pd_r = self.p_dis(reals_p)
                ## pd_r_loss = functional.binary_cross_entropy(pd_r, real_label * self.SOFT_LABEL)
                pd_r_loss = functional.binary_cross_entropy(pd_r, real_label)
                self.pd_r_loss_list.append(pd_r_loss)
                pd_r_loss.backward()

                # train p_dis with fake data
                pd_pf = self.p_dis(p_fakes.detach())
                pd_pf_loss = functional.binary_cross_entropy(pd_pf, fake_label)
                self.pd_pf_loss_list.append(pd_pf_loss)
                pd_pf_loss.backward()
                self.p_dis_optim.step()

                # ========== n_dis_net ==========
                # ========== n_dis_net ==========
                self.reset_grad()
                nz = torch.randn(self.BATCH_SIZE, self.BERT_FEATURES)
                nz_var = func.to_variable(nz, volatile=True)
                n_fakes = func.to_variable(self.n_gen(nz_var).data)

                # train n_dis with real data
                nd_r = self.n_dis(reals_n)
                nd_r_loss = functional.binary_cross_entropy(nd_r, real_label)
                self.nd_r_loss_list.append(nd_r_loss)
                nd_r_loss.backward()

                # train n_dis with fake data
                nd_nf = self.n_dis(n_fakes.detach())
                nd_nf_loss = functional.binary_cross_entropy(nd_nf, fake_label)
                self.nd_nf_loss_list.append(nd_nf_loss)
                nd_nf_loss.backward()
                self.n_dis_optim.step()

                # ===== u_dis_net =====
                # ===== u_dis_net =====
                self.reset_grad()

                # train u_dis with u_real data
                ud_r = self.u_dis(u_reals)
                label = label.squeeze()
                ud_r_loss = functional.binary_cross_entropy(ud_r, label)
                self.ud_r_loss_list.append(ud_r_loss)
                ud_r_loss.backward()

                # train u_dis with p_fake data
                ud_pf = self.u_dis(p_fakes.detach())
                ud_pf_loss = functional.binary_cross_entropy(ud_pf, fake_label)
                self.ud_pf_loss_list.append(ud_pf_loss)
                ud_pf_loss.backward()

                # train u_dis with n_fake images
                ud_nf = self.u_dis(n_fakes.detach())
                ud_nf_loss = functional.binary_cross_entropy(ud_nf, real_label)
                self.ud_nf_loss_list.append(ud_nf_loss)
                ud_nf_loss.backward()
                self.u_dis_optim.step()

                # =========================================
                # ========== update G network ===========
                # =========================================

                # ===== p_gen_net =====
                self.reset_grad()
                pz = torch.randn(self.BATCH_SIZE, self.BERT_FEATURES)
                pz_var = func.to_variable(pz)
                p_fakes = self.p_gen(pz_var)

                pg = self.p_dis(p_fakes)
                pg_loss = functional.binary_cross_entropy(pg, fake_label)
                pg_loss = pg_loss * self.P_WEIGHT

                pug = self.u_dis(p_fakes)
                pug_loss = functional.binary_cross_entropy(pug, real_label)
                pug_loss = pug_loss * self.PU_WEIGHT
                p_gen_loss = pg_loss + pug_loss
                self.p_gen_loss_list.append(p_gen_loss)
                p_gen_loss.backward()
                self.p_gen_optim.step()

                # ===== n_gen_net =====
                self.reset_grad()
                nz = torch.randn(self.BATCH_SIZE, self.BERT_FEATURES)
                nz_var = func.to_variable(nz)
                n_fakes = self.n_gen(nz_var)

                ng = self.n_dis(n_fakes)
                ng_loss = functional.binary_cross_entropy(ng, fake_label)
                ng_loss = ng_loss * self.N_WEIGHT

                nug = self.u_dis(n_fakes)
                nug_loss = functional.binary_cross_entropy(nug, fake_label)
                nug_loss = nug_loss * self.NU_WEIGHT
                n_gen_loss = ng_loss + nug_loss
                self.n_gen_loss_list.append(n_gen_loss)
                n_gen_loss.backward()
                self.n_gen_optim.step()

                # ===== y_gen_net =====
                self.reset_grad()
                yp_fakes = self.y_gen(p_fakes.detach())
                ygp = self.u_dis(p_fakes)
                ygp = func.to_variable(ygp)
                ygp_loss = functional.binary_cross_entropy(yp_fakes, real_label)
                ygp_loss = ygp_loss * self.P_WEIGHT

                yn_fakes = self.y_gen(n_fakes.detach())
                ygn = self.u_dis(n_fakes)
                ygn = func.to_variable(ygn)
                ygn_loss = functional.binary_cross_entropy(yn_fakes, fake_label)
                ygn_loss = ygn_loss * self.N_WEIGHT

                y_gen_loss = ygp_loss + ygn_loss
                self.y_gen_loss_list.append(y_gen_loss)
                y_gen_loss.backward()
                self.y_gen_optim.step()

            ## Saving model after each epoch
            self.save_model(epoch + 1)

            ## Testing data
            df = pd.read_csv('fever2_2_Class.csv')
            claims_test = df['claims']
            class_test = df['classes']
            batch_size = 100

            claims_test = DataLoader(claims_test, batch_size, shuffle=False)
            outPred = []

            score_P = simScore.score(p_fakes, pos_data)

            print(score_P)

            score_N = simScore.score(n_fakes, neg_data)

            print(score_N)


            for bi, data in tqdm(enumerate(claims_test), total=int(len(claims_test) / claims_test.batch_size)):

                if torch.cuda.is_available():
                    device = "cuda:0"
                else:
                    device = "cpu"


                claims_encode = model.encode(data)
                claims_encode = torch.from_numpy(claims_encode)
                claims_encode = claims_encode.to(device)
                prediction = self.y_gen(claims_encode)
                prediction = prediction.to('cpu')
                out = np.where(prediction >= 0.50, 1, 0)
                outPred.extend(out)

            target_names = ['Supported', 'Refuted']
            # report = classification_report(class_test, outPred, target_names=target_names)
            precision = precision_score(class_test, outPred)
            recall = recall_score(class_test, outPred)
            f1 = f1_score(class_test, outPred)
            print(f"Precision: {precision:.8f}, Recall: {recall:.8f}, F1 Score: {f1:.8f}")
            self.precisionList.append(precision)
            self.recallList.append(recall)
            self.f1scoreList.append(f1)

            ## Visualizatiom
            # pos_data = model.encode(pos_data)
            # gt_data = model.encode(gt_data)
            # neg_data
            plot(pos_data, p_fakes, neg_data, n_fakes, gt_data, epoch)

            pd_pf_loss_avg = sum(self.pd_pf_loss_list)/len(self.pd_pf_loss_list)
            self.pd_pf_loss_avg_list.append(pd_pf_loss_avg)
            pd_r_loss_avg = sum(self.pd_r_loss_list)/len(self.pd_r_loss_list)
            self.pd_r_loss_avg_list.append(pd_r_loss_avg)

            nd_r_loss_avg = sum(self.nd_r_loss_list)/len(self.nd_r_loss_list)
            self.nd_r_loss_avg_list.append(nd_r_loss_avg)
            nd_nf_loss_avg = sum(self.nd_nf_loss_list)/len(self.nd_nf_loss_list)
            self.nd_nf_loss_loss_list.append(nd_nf_loss_avg)

            y_gen_loss_avg = sum(self.y_gen_loss_list)/len(self.y_gen_loss_list)
            self.y_gen_loss_avg_list.append(y_gen_loss_avg)
            n_gen_loss_avg = sum(self.n_gen_loss_list)/len(self.n_gen_loss_list)
            self.n_gen_loss_avg_list.append(n_gen_loss_avg)
            p_gen_loss_avg = sum(self.p_gen_loss_list)/len(self.p_gen_loss_list)
            self.p_gen_loss_avg_list.append(p_gen_loss_avg)

            ud_r_loss_avg = sum(self.ud_r_loss_list)/len(self.ud_r_loss_list)
            self.ud_r_loss_avg_list.append(ud_r_loss_avg)
            ud_pf_loss_avg = sum(self.ud_pf_loss_list)/len(self.ud_pf_loss_list)
            self.ud_pf_loss_avg_list.append(ud_pf_loss_avg)
            ud_nf_loss_avg = sum(self.ud_nf_loss_list)/len(self.ud_nf_loss_list)
            self.ud_nf_loss_avg_list.append(ud_nf_loss_avg)

            print(f"Positive Generator loss: {p_gen_loss_avg:.8f}, Positive Discriminator loss: {pd_pf_loss_avg:.8f},")
            print(f"Negative Generator loss: {n_gen_loss_avg:.8f}, Negative Discriminator loss: {nd_nf_loss_avg:.8f},")

            print(f"Positive Discriminator real loss loss: {pd_r_loss_avg:.8f}, Negative Discriminator real loss: {nd_r_loss_avg:.8f},")

            print(f"Class label generator loss for positive samples: {ud_pf_loss_avg:.8f}")
            print(f"Class label generator loss for negative samples: {ud_nf_loss_avg:.8f}")
            print(f"Class label generator loss  for real samples: {ud_r_loss_avg:.8f}")

            print(f"Class label generator loss: {y_gen_loss_avg:.8f}")
    def plot_graphs(self):
        # plot and save the generator and discriminator loss for positive

        # ud_pf_loss_avg = sum(self.ud_pf_loss_list) / len(self.ud_pf_loss_list)
        # ud_nf_loss_avg = sum(self.ud_nf_loss_list)/len(self.ud_nf_loss_list)
        # ud_r_loss_avg = sum(self.ud_r_loss_list) / len(self.ud_r_loss_list)

        plt.figure()
        plt.plot(self.ud_pf_loss_list, label='Y Discriminator positive loss')
        plt.plot(self.ud_nf_loss_list, label='Y Discriminator negative loss')
        plt.plot(self.ud_r_loss_list, label='Y Discriminator final loss')
        plt.legend()
        plt.savefig('plots/Y-Dis_Losses_new.png')


        #y_gen_loss_avg = sum(self.y_gen_loss_list) / len(self.y_gen_loss_list)


        plt.figure()
        plt.plot(self.y_gen_loss_list, label='Generator loss')
        plt.plot(self.ud_r_loss_list, label='Discriminator loss')
        plt.legend()
        plt.savefig('plots/Y-Gen_Dis_Loss_new.png')


        plt.figure()
        plt.plot(self.p_gen_loss_list, label='Positve Generator loss')
        plt.plot(self.pd_pf_loss_list, label='Positve Discriminator loss (fake)')
        plt.plot(self.pd_r_loss_list, label='Positve Discriminator loss (real)')
        plt.legend()
        plt.savefig('plots/P-Gen_Dis_Loss_new.png')

        plt.figure()
        plt.plot(self.nd_r_loss_list, label='Negative Discriminator loss (real)')
        plt.plot(self.nd_nf_loss_list, label='Negative Discriminator loss (fake)')
        plt.plot(self.n_gen_loss_list, label='Negative Generator loss')
        plt.legend()
        plt.savefig('plots/N-Gen_Dis_Loss_new.png')

        plt.figure()
        plt.plot(self.precisionList, label = 'Precision')
        plt.plot(self.recallList, label='Recall')
        plt.plot(self.f1scoreList, label='F1')
        plt.legend()
        plt.savefig('plots/Precision_Recall_F1_new.png')


    def test_result(self):
        df = pd.read_csv('test_data.csv')
        claims_test = df['claims']
        class_test = df['classes']
        batch_size = 100

        claims_test = DataLoader(claims_test, batch_size, shuffle=False)
        outPred = []
        precisionList = []
        recallList = []
        f1scoreList = []
        for bi, data in tqdm(enumerate(claims_test), total=int(len(claims_test) / claims_test.batch_size)):
            print("______________________________HERE_____________________________________________")
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"

            claims_encode = model.encode(data)
            claims_encode = torch.from_numpy(claims_encode)
            claims_encode = claims_encode.to(device)
            prediction = self.y_gen(claims_encode)
            prediction = prediction.to('cpu')
            out = np.where(prediction >= 0.50, 1, 0)
            outPred.extend(out)
            print("___________________________________________________________________________")
        target_names = ['Supported', 'Refuted']
        precision = precision_score(class_test, outPred)
        recall = recall_score(class_test, outPred)
        f1 = f1_score(class_test, outPred)

        print(f"Precision: {precision:.8f}, Recall: {recall:.8f}, F1 Score: {f1:.8f}")

        self.precisionList.append(precision)
        self.recallList.append(recall)
        self.f1scoreList.append(f1)


def arraygan_main(config):
    # create pun gan model
    array_gan = ArrayGAN(config)
    print("Model Done")
    ## Training model
    array_gan.train_model()
    print("Training Done")
    ## Testing
    print("== Final Testing ==")
    array_gan.test_result()

    ## Genrate plots
    array_gan.plot_graphs()

if __name__ == '__main__':
    # positive class
    pc_list = [3]
    # negative class
    nc_list = [5]
    # percentage of positive samples as P data
    pr_list = [0.002]
    # positive weight
    pw_list = [0.01]
    puw_list = [1.0]
    # negative weight
    nw_list = [100.0]
    nuw_list = [1.0]
    for pr in pr_list:
        for pc, nc in zip(pc_list, nc_list):
            for pw, puw, nw, nuw in zip(pw_list, puw_list, nw_list, nuw_list):
                print('pr[%.4f], pc[%d]-nc[%d], pw[%.2f]-puw[%.2f]-nw[%.2f]-nuw[%.2f]'
                      % (pr, pc, nc, pw, puw, nw, nuw))
                # set parameters for generative PU model
                parser = argparse.ArgumentParser()
                # model hyper-params
                parser.add_argument('--P_CLASS', type=int, default=pc)
                parser.add_argument('--N_CLASS', type=int, default=nc)
                parser.add_argument('--P_RATIO', type=float, default=pr)
                parser.add_argument('--CHANEL_DIM', type=int, default=1)
                parser.add_argument('--Z_DIM', type=int, default=config.HIDDEN_LAYER)
                parser.add_argument('--MAP_DIM', type=int, default=config.HIDDEN_LAYER)
                parser.add_argument('--IMAGE_SIZE', type=int, default=28)
                parser.add_argument('--LEAK', type=float, default=0.2)
                # BERT Features
                parser.add_argument('--BERT_FEATURES', type=int, default=config.BERT_FEATURES)
                parser.add_argument('--HIDDEN_LAYER', type=int, default=config.HIDDEN_LAYER)
                # training hyper-params
                parser.add_argument('--NUM_GPU', type=int, default=1)
                parser.add_argument('--EPOCHS', type=int, default=200)
                parser.add_argument('--P_WEIGHT', type=float, default=pw)
                parser.add_argument('--PU_WEIGHT', type=float, default=puw)
                parser.add_argument('--N_WEIGHT', type=float, default=nw)
                parser.add_argument('--NU_WEIGHT', type=float, default=nuw)
                parser.add_argument('--LR', type=float, default=0.0003)
                parser.add_argument('--BETA1', type=float, default=0.9)
                parser.add_argument('--BETA2', type=float, default=0.999)
                parser.add_argument('--SOFT_LABEL', type=float, default=1.0)
                # misc
                parser.add_argument('--IMAGE_SET', type=str, default='mnist')
                parser.add_argument('--IMAGE_PATH', type=str, default='./data')
                parser.add_argument('--NUM_IMAGES', type=int, default=5000)
                parser.add_argument('--WORKERS', type=int, default=2)
                parser.add_argument('--BATCH_SIZE', type=int, default=100)
                parser.add_argument('--VERBOSE', type=bool, default=True)
                parser.add_argument('--LOG_STEP', type=int, default=200)
                parser.add_argument('--LOG_PATH', type=str, default='./logs')
                parser.add_argument('--SAMPLE_STEP', type=int, default=20)
                parser.add_argument('--SAMPLE_SIZE', type=int, default=200)
                parser.add_argument('--SAMPLE_PATH', type=str, default='./samples')
                parser.add_argument('--MODEL_PATH', type=str, default='./models')
                args = parser.parse_args()
                # main
                arraygan_main(args)
