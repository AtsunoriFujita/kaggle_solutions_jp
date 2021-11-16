# RSNA-MICCAI Brain Tumor Radiogenomic Classification

## 期間
- 2021/07 - 2021/10

## Overall
- 3DのMRI画像からMGMTプロモーターのメチル化有無を分類するタスク
- データのサイズはトータルで約140GBとボリュームがあるが学習データの患者数は600名に満たず、パブリックテストデータの感が数も90名程度。各患者ごとに4つの撮影方法によるデータが含まれている。PublicとPrivateの比率は22:78
- 評価指標: AUC
- 主なトピック（リーク有無、shakeの有無など）
- 1,555チーム参加
- そもそも分類が難しい、小サンプルということもあり、トップスコアでもAUC: 0.62174であった。激しくシェイクした結果となった。

## 1st place solution
- ディスカッション：https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/281347
- ソースコード
  - https://github.com/FirasBaba/rsna-resnet10
  - https://www.kaggle.com/rinnqd/monai-simple-prediction-from-flair
- 多くの参加者と同様に実験結果に一貫性がなくモデルが有用なパターンを認識できないことに苦悩した。最終的にはシンプルなモデルとなった。
- 最終的なモデル
  - 3D CNN
  - Resnet10
  - BCE loss
  - Adam optimizer
  - 15 epochs
  - LR: epoch 1->10; lr = 0.0001 | epoch 10 to 15 lr=0.00005
  - Image size: 256x256
  - Batch size: 8 (the bigger bs I use the worse CV I get, I was alternating between bs=4 and bs=8)
  - mixed-precisionを使わない。
  - 3D画像からは最高の中心画像を抽出した
  - RTX 3090で1エポック1分20秒。

## 2nd place solution
- ディスカッション:https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/280033
- ソースコード
  - https://www.kaggle.com/minhnhatphan/rnsa-21-cnn-lstm-train/notebook
  - https://www.kaggle.com/minhnhatphan/rnsa-21-cnn-lstm-inference
  - https://github.com/minhnhatphan/rnsa21-cnn-lstm
  - https://www.kaggle.com/minhnhatphan/rnsa21-cnn-lstm-refactored/notebook
- CNN-LSTMモデルを採用。
  - CNNのbackbornはEfficientNet B0
  - 15 epochs
  - Adam optimizer with learning rate = 1e-4
  - ShiftScaleRotate, RandomBrightnessContrast
  - Image size: 256x256

## 3rd place solution
- ディスカッション：https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/287713
- ソースコード
  - https://www.kaggle.com/cedricsoares/tf-efficientnet-transfer-learning-strat-split
  - https://github.com/cedricsoares/kaggle-rsna-miccai-brain-tumor-radiogenomic-classification
- 4つの撮影方法ごとにEfficientNet-B3で学習。
