from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, item):
        #print(item)
        image, label = item['image'], item['label']
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image.numpy(), (new_h, new_w))
        return {'image': img, 'label':label}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, item):
        image, label = item['image'], item['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        return {'image': image, 'label':label}

class ToTensor(object):
    def __call__(self, item):
        #print(item)
        image, label = item['image'], item['label']
        image = image.transpose((2, 0, 1))
        return {'image': image, 'label':label}

outputdict = {"0.8417296032023145": 0, "0.03327038291221672":1,  "1.0":2, 
                "0.84393310546875": 3, "0.03106689453125":4,
                "0.8499999999999999":5, "0.024999999999999998":6,
                "0.835":7, "0.04":8,
                "0.9062795643985768":9, "0.031220428876678877":10,
                "0.9125238040313137":11, "0.02497619163899199":12,
                "0.9175000000000003":13, "0.02":14,
                "0.9062643052602612":15, "0.031235693197039872":16}
outputdictMap = {"2.0": 0, "3.0":1,  "4.0":2, 
                "5.0": 3, "6.0":4,
                "7.0":5, "8.0":6,
                "9.0":7, "10.0":8,
                "11.0":9, "12.0":10,
                "13.0":11, "14.0":12,
                "15.0":13, "16.0":14,
                "17.0":15, "18.0":16,
		"19.0":17, "20.0":18,
		"21.0":19, "22.0":20,
		"23.0":21, "24.0":22,
		"25.0":23, "26.0":24,
		"27.0":25,"28.0":26,
		"29.0":27,"30.0":28}
class MyDatasetLoader(Dataset):

    def __init__(self, images, labels, root_dir, regression = False, transform=None):
        self.images = images
        if not regression:
            values = []
            #print(labels)
            for label in labels:
                l = [0]*len(outputdictMap)
                l[outputdictMap[str(float(label[0]))]] = 1
                values.append(l)
        else:
            values = labels
        self.labels = torch.from_numpy(np.array(values))
        self.root_dir = root_dir
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name =  os.path.join(self.root_dir, self.images[idx])
        image = torch.from_numpy(io.imread(img_name))#.double()
        label = self.labels[idx]
        item = {'image': image, 'label':label}
        if self.transform:
            item = self.transform(item)
        #print(item)
        return item

    def __len__(self):
        return len(self.images)
currentdir = os.getcwd()
gt = pd.read_csv(currentdir+'/datasetgen/groundtruth.txt', sep=" ", header=None)
gt = gt.dropna(axis=1, how='all')
real = pd.read_csv(currentdir+'/datasetgen/realgroundtruth.txt', sep=" ", header=None)
gt = gt.sample(frac=1)
trows = int(len(gt)*0.70)
vrows = int(len(gt)*0.90)
print("training:", trows, ", validation:", vrows)
training = gt[:trows]
validation = gt[trows:vrows]
testing = gt[vrows:]
training_X, training_Y = np.array(training[0]), np.array(training.iloc[:,2:3])#.int()
validation_X, validation_Y = np.array(validation[0]), np.array(validation.iloc[:,2:3])#.int()
testing_X, testing_Y = np.array(testing[0]), np.array(testing.iloc[:,2:3])
real_X, real_Y =  np.array(real[0]), np.array(real.iloc[:,2:3])
print(len(testing_X))
print("Training and Testing my model")
batch = 16
regression = True
numberofclasses = 1
input_size = 256
simulatedimagedirector=currentdir+"/datasetgen/dataset/"
realimagedirectory=currentdir+"/datasetgen/realimages/"
trainingSet = MyDatasetLoader(training_X, training_Y, simulatedimagedirector, regression, 
                                                                          transform=transforms.Compose([Rescale(input_size),
                                                                                                    ToTensor()]))
validationset = MyDatasetLoader(validation_X, validation_Y, simulatedimagedirector, regression, transform=transforms.Compose([Rescale(input_size), ToTensor()]))                                                                              
testingSet = MyDatasetLoader(testing_X, testing_Y, simulatedimagedirector, regression, 
                                                                         transform=transforms.Compose([Rescale(input_size),
                                                                                                    #RandomCrop(224),
                                                                                                   ToTensor()]))
realset = MyDatasetLoader(real_X, real_Y, realimagedirectory, regression, transform=transforms.Compose([Rescale(input_size), ToTensor()])) 
training_loader = DataLoader(trainingSet, batch_size=batch, shuffle=True, num_workers=0)
validation_loader = DataLoader(validationset, batch_size=batch, shuffle=True, num_workers=0)
testing_loader = DataLoader(testingSet, batch_size=batch, shuffle=True, num_workers=0)
real_loader = DataLoader(realset, batch_size=len(real_Y), shuffle=True, num_workers=0)
print(torch.__version__)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 128, kernel_size = 2, stride = 2, padding = 0)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2, padding=0) # in size: 256, out size: 128
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)
        #in size: 128, out size: 32
        self.conv2_bn = nn.BatchNorm2d(256)
        self.drop_out = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 32 * 32, 4096)
        self.fc1_bn = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, numberofclasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.drop_out(x)
        x = x.view(-1, 256 * 32 * 32)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net().double()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.MultiLabelSoftMarginLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001)
optimizer = optim.Adam(net.parameters(), lr=0.001)
#optimizer = optim.Adam([{'params': net.features.parameters()},
#    {'params': net.classifier.parameters(), 'weight_decay': 0.1}], lr=0.001)
print("Adam loss, Adam optimizer, lr = 0.001")
from sklearn.metrics import f1_score
from sklearn import metrics

def Fbetascore(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_t = []
    y_p = []
    for row in y_true:
        row = list(row)
        y_t.append(row.index(1))
    for row in y_pred:
        row = list(row)
        y_p.append(row.index(1))
    #print("y_t:",y_t)
    #print("y_p:",y_p)
    res = metrics.fbeta_score(y_t, y_p, 1.0, average='micro')
    return res

def Accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_t = []
    y_p = []
    for row in y_true:
        row = list(row)
        y_t.append(row.index(1))
    for row in y_pred:
        row = list(row)
        y_p.append(row.index(1))
    return np.sum(np.array(y_t) == np.array(y_p))

from scipy.stats import pearsonr
def SpearmanCor(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_t = []
    y_p = []
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    for row in y_true:
        y_t.append(row)
    for row in y_pred:
        row = list(row)
        y_p.append(row[0])
    print("y_t:", y_t)
    print("y_p:", y_p)
    cor, p = pearsonr(y_t, y_p)
    return cor, p

traininglossperepoch = []
validationlossperepoch = []
testinglossperepoch = []

trainingcor = []
validationcor = []
testingcor = []

trainingp = []
validationp = []
testingp = []

nepochs = 50
start = time.time()
for epoch in range(nepochs):
    running_loss = 0.0
    predicted = []
    truelabels = []
    net.train()
    for i, data in enumerate(training_loader, 0):
        inputs, labels = data['image'], data['label']
        inputs = inputs
        #print(inputs)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels.double())
        loss.backward()
        optimizer.step()
        output = outputs.detach().numpy()
        predicted = predicted + list(output)
        truelabels = truelabels + list(labels)
        #print(pred)
        running_loss += loss.item() #* batch
        #print("Loss:", loss)
        if (i) % 505 == 0:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / ((i+1))))
            #print(totalcorrect)
        #    running_loss = 0.0
    traininglossperepoch.append(running_loss * batch/len(training_Y))
    cor, p = SpearmanCor(truelabels, predicted)  
    trainingcor.append(cor)
    trainingp.append(p)  
    totloss = 0
    predicted = []
    truelabels = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader, 0):
            inputs, labels = data['image'], data['label']
            inputs = inputs.double()
            outputs = net(inputs)
            totloss += criterion(outputs.double(), labels.double()).item()# * batch
            output = outputs.detach().numpy()
            predicted = predicted + list(output)
            truelabels = truelabels + list(labels)
            #_, outputs = torch.max(outputs, 1)
    #print("Testing Loss:", 1.0*totloss/len(testing_Y))
    validationlossperepoch.append(1.0 * totloss * batch/len(validation_Y))
    cor, p = SpearmanCor(truelabels, predicted)
    validationcor.append(cor)
    validationp.append(p)
    #print('Finished Testing!')
#test()
end = time.time()
torch.save(net.state_dict(), "mynetregmseadampearson.pth")
totloss = 0
predicted = []
truelabels = []
net.eval()
with torch.no_grad():
    for i, data in enumerate(testing_loader, 0):
        inputs, labels = data['image'], data['label']
        inputs = inputs.double()
        outputs = net(inputs)
        labels = labels.double()
        #_, outputs = torch.max(outputs, 1)
        truelabels = truelabels + list(labels)
        output = outputs.detach().numpy()
        predicted = predicted + list(output)
        totloss += criterion(outputs.double(), labels.double()).item()# * batch
    testinglossperepoch.append(1.0*totloss*batch/len(testing_Y))
    cor, p = SpearmanCor(truelabels, predicted)
    testingcor.append(cor)
    testingp.append(p)
runtime = end - start
print('Training time: {:.0f}m {:.0f}s'.format(runtime // 60, runtime % 60))
print("Testing predicted:", predicted)
print("Testing true:", truelabels)

realloss = []
realp = []
realcor = []
totloss = 0
realtruth = []
realpred = []
net.eval()
with torch.no_grad():
    for i, data in enumerate(real_loader, 0):
        inputs, labels = data['image'], data['label']
        inputs = inputs.double()
        outputs = net(inputs)
        output = outputs.detach().numpy()
        #_, outputs = torch.max(outputs, 1)
        labels = labels.double()
        realtruth.append(labels)
        realpred.append(outputs)
        y_true = np.array(labels)
        y_pred = np.array(outputs)

    realloss.append(1.0*totloss*batch/len(real_Y))
cor, p = SpearmanCor(y_true, y_pred)
print("real true y:", y_true)
print("real pred y:", y_pred)
print("real cor:", cor)
print("real p:", p)


print("training loss", traininglossperepoch)
print("validation loss:", validationlossperepoch)
print("testing loss:", testinglossperepoch)

print("training cor:", trainingcor)
print("validation cor:", validationcor)
print("testing cor:", testingcor)

print("training p:", trainingp)
print("validation p:", validationp)
print("testing p:", testingp)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
t = range(nepochs)
plt.plot(t, np.array(traininglossperepoch), 'r--', t, np.array(validationlossperepoch), 'g--',linewidth=2)
#plt.plot(t, np.array(traininglossperepoch), 'r--', linewidth=2)
plt.xlabel('# Epochs') 
plt.ylabel('Loss')
plt.legend(('Training loss', 'Validation loss'),loc='upper right')
plt.title('Loss vs. # epochs')
#plt.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('lossRegadam-mse.png')
#plt.show()

plt.clf()
plt.cla()
plt.close()

#import numpy as np
#import matplotlib.pyplot as plt

t = range(nepochs)
plt.plot(t, np.array(trainingcor), 'r--', t, np.array(validationcor), 'g--',linewidth=2)
plt.xlabel('# Epochs') 
plt.ylabel('Spearman correlation')
plt.legend(('Training', 'Validation'),
           loc='upper right')
plt.title('Spearman correlation vs. # epochs')
#plt.show()
#
plt.savefig('spcorRegadam-mse.png')

plt.clf()
plt.cla()
plt.close()

#import numpy as np
#import matplotlib.pyplot as plt

t = range(nepochs)
plt.plot(t, np.array(trainingp), 'r--', t, np.array(validationp), 'g--',linewidth=2)
plt.xlabel('# Epochs') 
plt.ylabel('p-value')
plt.legend(('Training', 'Validation'),
           loc='upper right')
plt.title('p-value vs. # epochs')
#plt.show()
plt.savefig('pvRegadam-mse.png')

