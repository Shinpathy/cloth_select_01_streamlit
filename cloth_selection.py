import streamlit as st
#基本ライブラリ
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
import torchsummary
from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger

from torchvision.models import resnet18

from torch.utils.data import Dataset

#予測モデル構築
class Net(pl.LightningModule):
    
  def __init__(self):
    super().__init__()

    self.feature = resnet18(pretrained=True)

    self.fc = nn.Linear(1000, 2)

  def forward(self, x):
    h = self.feature(x)
    h = self.fc(h)
    return h

  def training_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('train_loss', loss, on_step=True, on_epoch=True)
    self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2), on_step=True, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('val_loss', loss, on_step=False, on_epoch=True)
    self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2), on_step=False, on_epoch=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('test_loss', loss, on_step=False, on_epoch=True)
    self.log('test_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2), on_step=False, on_epoch=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    return optimizer

#入力画像前処理コード
#データの前処理 学習時と同じ前処理を施す。
def preprocess_image(image):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = transform(image).unsqueeze(0)
    return image

#ネットワークの準備
net = Net().cpu().eval()
#重みの読み込み
net.load_state_dict(torch.load('/Users/shinichirotakeda/Desktop/Streamlit_for_app/cloth_selection.pt', map_location=torch.device('cpu')))

#streamlitアプリのメインコード
def main():
    st.title('Tシャツの好みを出力するアプリ')
    
    uploaded_image = st.file_uploader('Tシャツの画像をアップロードしてください', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        #予測ボタンが押されたら予測を実行
        if st.button('予測'):
            #画像を前処理
            preprocessed_image = preprocess_image(image)
            
            #モデルに画像を入力して予測
            if net is not None:
                with torch.no_grad():
                    
                    #予測
                    y_pred = net(preprocessed_image)

                    #確率値に変換
                    y_pred = F.softmax(y_pred)
                    
                    #最大値のラベルを取得
                    y_pred_label = torch.argmax(y_pred)
                    
                    #予測結果を表示
                    st.write(f'予測結果: {y_pred_label}')
if __name__ == '__main__':
    main()