import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import DenseNet121_Weights, densenet121, DenseNet169_Weights, densenet169, VGG16_Weights, vgg16


def getModel(name, param, device='cpu'):
    if name == 'LSTM':
        return LSTM(*param).to(device)
    if name == 'TinyVGG':
        return TinyVGG(*param).to(device)
    if name == 'DenseNet121':
        return DenseNet121Model(*param).to(device)
    if name == 'DenseNet169':
        return DenseNet169Model(*param).to(device)
    if name == 'VGG16':
        return VGG16Model(*param).to(device)
    
    print('Model not found')
    return None


class theModel(pl.LightningModule):
    # def __init__(self, titleModelName = 'LSTM', titleParam = (4071, 128, 2, True, 18), posterModelName = 'TinyVGG', posterParam = (3, 32, 18), urModelName = None, urParam = None, num_labels=18, device='cpu'):
    def __init__(self, titleModel, posterModel, userRatingModel, num_labels=18):
        super(theModel, self).__init__()
        self.num_labesls = num_labels
        
        self.titleModel = titleModel
        self.posterModel = posterModel
        self.userRatingModel = userRatingModel
        #TODO: add user rating models

        # Assembling
        self.fc = nn.Linear(2*self.num_labesls, self.num_labesls)

    def forward(self, title, poster, user_rating):
        Tout = Pout = Uout = torch.tensor([]).to(self.device)
        if self.titleModel != None:
            Tout = self.titleModel(title)
        if self.posterModel != None:
            Pout = self.posterModel(poster)
        if self.userRatingModel != None:
            Uout = self.userRatingModel(user_rating)
        # Assembling
        out = self.fc(torch.cat((Tout, Pout, Uout), dim=1))
        return out

    def training_step(self, train_batch, batch_idx):
        title_tensor, img_tensor, ur_tensor, genre_tensor = self.getItemFromBatch(train_batch, batch_idx)

        output = self.forward(title_tensor, img_tensor, ur_tensor)
        loss = self.loss_fnc(output, genre_tensor)
        return loss

    def validation_step(self, val_batch, batch_idx):
        title_tensor, img_tensor, ur_tensor, genre_tensor = self.getItemFromBatch(val_batch, batch_idx)

        output = self.forward(title_tensor, img_tensor, ur_tensor)
        loss = self.loss_fnc(output, genre_tensor)
        self.log('val_loss', loss)
        # print('val_loss', loss)

    def predict_step(self, test_batch, batch_idx):
        title_tensor, img_tensor, ur_tensor, genre_tensor = self.getItemFromBatch(test_batch, batch_idx)

        output = self.forward(title_tensor, img_tensor, ur_tensor)
        return output, genre_tensor

    def getItemFromBatch(self, batch, idx):
        title, img, ur, genres = batch
        # title, img, genres = batch
        title_tensor = title.clone().detach().to(self.device)
        img_tensor = img.clone().detach().to(self.device)
        ur_tensor = ur.clone().detach().to(self.device)
        # ur_tensor = torch.tensor([]).to(self.device)
        genre_tensor = genres.clone().detach().to(self.device)
        return title_tensor, img_tensor, ur_tensor, genre_tensor
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def loss_fnc(self, logits, labels):
        return F.cross_entropy(logits, labels)

class LSTM(nn.Module):
    def __init__(self, input_size=4071, hidden_size=128, num_layers=2, bidirectional=True, num_labels=18) -> None:
        super(LSTM, self).__init__()
        print('LSTM', input_size, hidden_size, num_layers, bidirectional, num_labels)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_labesls = num_labels

        self.core = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
        )
        linear_size = self.hidden_size
        if self.bidirectional:
            linear_size *= 2

        self.linear = nn.Sequential(
            nn.Linear(linear_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_labesls),
            # nn.ReLU()
        )

    def forward(self, title):
        Tout, _ = self.core(title)
        # out = [batch_size, seq_len, hidden_size*bidirectional (vector size)]
        # => only take the last element (many to one RNN)
        Tout = Tout[:, -1, :]
        Tout = self.linear(Tout)
        return Tout       

class TinyVGG(nn.Module):
    def __init__(self, input_shape=3, hidden_units=32, image_size=256, num_labels=18) -> None:
        super(TinyVGG, self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.num_labesls = num_labels
        self.IMAGE_SIZE = (image_size, image_size)
        print('TinyVGG', input_shape, hidden_units, self.IMAGE_SIZE, num_labels)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_shape, out_channels=self.hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_units,
                      out_channels=self.hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_units,
                      out_channels=self.hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_units,
                      out_channels=self.hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.hidden_units *
                      int(self.IMAGE_SIZE[0]/4)*int(self.IMAGE_SIZE[0]/4), out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.num_labesls)
        )

    def forward(self, poster):
        Pout = self.conv_block_1(poster)
        Pout = self.conv_block_2(Pout)
        Pout = self.classifier(Pout)
        return Pout
    

class DenseNet121Model(nn.Module):
  def __init__(self, input_shape: int, output_shape: int):
    super().__init__()
    print('DenseNet121', input_shape, output_shape)
    self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    self.model.classifier = nn.Sequential(
        nn.Linear(in_features=1024, out_features=512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=output_shape)
    )

  def forward(self, x):
    return self.model(x)
  
class DenseNet169Model(nn.Module):
  def __init__(self, input_shape: int, output_shape: int):
    super().__init__()
    print('DenseNet169', input_shape, output_shape)
    self.model = densenet169(weights=DenseNet169_Weights.DEFAULT)
    self.model.classifier = nn.Sequential(
        # densenet 169
        nn.Linear(in_features=1664, out_features=1024),
        nn.ReLU(),
    )
    self.classifier = nn.Sequential(
        nn.Linear(in_features=1024, out_features=512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=output_shape)
    )

  def forward(self, x):
    return self.classifier(self.model(x))

class VGG16Model(nn.Module):
  def __init__(self, input_shape: int, output_shape: int):
    super().__init__()
    print('VGG16', input_shape, output_shape)
    self.model = vgg16(weights=VGG16_Weights.DEFAULT)
    self.model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=output_shape)
    )

  def forward(self, x):
    return self.model(x)
class multilabelLogisticRegression():
   pass

class simpleNN(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super(simpleNN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        print('simpleNN', input_shape, output_shape)

        self.core = nn.Sequential()
        while input_shape > 4*output_shape:
            self.core.add_module(f'linear{input_shape}', nn.Linear(input_shape, input_shape//2))
            self.core.add_module(f'relu{input_shape}', nn.ReLU())
            input_shape //= 2
        self.core.add_module(f'linear{input_shape}', nn.Linear(input_shape, output_shape))

    def forward(self, title):
        out = self.core(title)
        return out
   