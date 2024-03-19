from Dataloader_publc_HD import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
import numpy as np
import sys
from torch.autograd import Variable
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
from m_spl import sp_cost
from torch.nn import functional as F

args = parse_args()

mBatchSize = args.batchsize
train_set = args.train_set
mEpochs = args.epoch
model_select = args.model_select
mLearningRate = args.lr
dim = args.band_number
num_workers = args.num_workers

mDevice=torch.device("cuda")
nora = args.nora
print('mBatchSize',mBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('model_select',model_select)
print('nora',nora)


model_save = "./" + "BS_Publc_HD_EM_select_10_" + "_model_select_" + str(model_select) + "_all_" + str(mBatchSize) + "_" + str(mLearningRate) + "/"


def Loss_EM(prob, select_band=20):
    band_number = prob.shape[1]
    prob = prob.squeeze(2)
    prob = (torch.cat((1-prob,prob),dim=2).permute(1,0,2)+0.00001).log()
    token = Variable(torch.ones(select_band)).cuda()
    sizes = Variable(torch.IntTensor(np.array([band_number]))).cuda()
    target_sizes = Variable(torch.IntTensor(np.array([select_band]))).cuda()
    cost = sp_cost(prob, token, sizes, target_sizes)
    return cost


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, model_num_outputs=3, loss_balance_weights=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )
        self.model_num_outputs = model_num_outputs
        self.loss_balance_weights = loss_balance_weights

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='nearest')
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        if self.model_num_outputs == 1:
            score = [score]
        weights = self.loss_balance_weights
        assert len(weights) == len(score)
        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


if __name__ == '__main__':
    dim = dim
    model_save = model_save + 'dim_' + str(dim) + train_set + '/'
    featureTrans = False
    trainDataset = MyDataset_whole(train_set, dim = dim, feature_extraction=featureTrans, dataType = 'train')
    testDataset = MyDataset_whole(train_set, dim = dim, feature_extraction=featureTrans, dataType = 'test')

    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True, num_workers = 10)#, pin_memory=False
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True, num_workers = 10)#, pin_memory=False

    dim = 144
    #
    class_nums = 15
    from Single_block_net import Single_basic_block_net
    model_cls = Single_basic_block_net(dim, class_nums+1).cuda()
    from band_selection import BS_Layer_with_r
    model_sel = BS_Layer_with_r(dim).cuda()


    criterion = CrossEntropy(ignore_label=0,
                                 weight=None,
                                 model_num_outputs=1,
                                 loss_balance_weights=[1])
    paras = list(model_cls.parameters()) + list(model_sel.parameters())
    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=mLearningRate, weight_decay = 0.0001)
    optimizer = torch.optim.Adam(paras, lr=mLearningRate, weight_decay = 0.0001)
    select_bands_num = 10

    model_cls.train()
    model_sel.train()
    for epoch in range(mEpochs):
        trainCorrect = 0
        trainTotal = 0
        trainLossTotal = 0.0
        random_tensor = (torch.rand(1, dim, 1, 1) * 0.9 + 0.1)#.cuda()
        for i, data in enumerate(tqdm(trainLoader)):
            torch.cuda.empty_cache()
            img, label = data
            img, label = Variable(img).float().cuda(), Variable(label).long().cuda()
            b,c,h,w = img.shape
            X, weights = model_sel(img)
            predict = model_cls(X)
            predictIndex = torch.argmax(predict, dim=1)
            
            loss_2 = Loss_EM(weights, select_bands_num)
            loss = criterion(predict, label)
            loss_2 = 0.1 * loss_2
            loss = loss + loss_2
            trainLossTotal += loss
            trainLossTotal += loss_2
            trainCorrect += ((predictIndex == label) & (label != 0)).sum()
            trainTotal += torch.sum(label != 0).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        weights = weights.squeeze().detach().cpu().numpy()
        band_indx = np.argsort(weights)[::-1][:select_bands_num]
        accuracy = trainCorrect.item() / trainTotal
        print('train epoch:', epoch, ': ', accuracy)
        print('total loss = %.5f' % float(trainLossTotal))
        print("the mean weight of every channel : ", np.sort(weights)[::-1][:select_bands_num])
        print('train epoch:', epoch, ', select best bands:',band_indx)
        print('\n')
        channel_weight_list = np.array(weights, dtype=np.float32)

        if (epoch+1) % 1 == 0:
            mkdir(model_save)
            torch.save(model_cls.state_dict(), model_save+ str(epoch) + 'test_acc_' + str(accuracy) + '.pkl', _use_new_zipfile_serialization=False)
            torch.save(model_sel.state_dict(), model_save+ str(epoch) + 'model_sel_test_acc_' + str(accuracy) + '.pkl', _use_new_zipfile_serialization=False)
            np.save(model_save + str(epoch) + 'test_acc_' + str(accuracy) + '_channel_weight_list.npy',channel_weight_list)


