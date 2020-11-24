# [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis/overview)

## 期間
- 2020/04 - 2020/07

## Overall
- 画像に埋め込まれた暗号化されたメッセージを検出する画像コンペ
- プレーンな画像75k枚とそれに対応する3つのアルゴリズム（JMiPOD, JUNIWARD, UERD）で暗号が埋め込まれた画像の計30万枚,テストデータは5k枚、PublicとPrivateの比率は20:80
- Solutionは画像を使ったモデルとDCT変換を使ったモデルの組み合わせが主。リークはなかった。小さなshakeがあり、1位はpublicLBは信用してはいけないと述べた。

## 1st place solution
- ディスカッション：https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168548
- ソースコード：なし
- 空間ドメイン（YCbCr）とDCTで学習したモデルを組み合わせた
- 空間ドメインにはimagenetで学習済みのseresnet18を使用（機能させるために最初のconvレイヤとmax-poolingのstrideを削除)。弱い信号を捉えるためにse-blockのChannel-attention は機能した。
- DCTで学習するために8×8=64DCTコンポーネントを入力チャネルに変換したため、元の3×512×512は(64×3=192)×64×64になる。192のDCTあたいはCNNの前にone-hot encodingした。CNNは6層の3x3convレイヤーでresidual connectionsとse-blockがあり、空間サイズは最後のGAPレイヤーまで64x64で一定。
- augmentationはrotate90とflipsの他にcutmixがかなり上手く機能した。また学習中に暗号が埋め込まれた画像に出会うたびに、変更した位置のDCT値に+1と-1をランダムに再割り当てした。
- 2nd-levelモデルは通常のfully-connected net

## 2nd place solution
- ディスカッション：https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/173489
- ソースコード：https://github.com/BloodAxe/Kaggle-2020-Alaska2/tree/master
- CVを信じた
- 画像のサイズは変更しない, augmentationsを使用する前に慎重に考える
- 標準の画像I/Oライブラリを使用しない（ピクセル値を[0..255]に丸めたりクリッピングしたりしない）
- より深いレイヤーは解像度が高いほど良かった
- 多様なアンサンブルを構築する
  - EfficientNet
  - MixNet
  - SRNet
  - 自作の特徴（DCTR/JRM）

## 3rd place solution
- ディスカッション：https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168870
- ソースコード：なし
- 2つのモデルのアンサンブル。1つは3チャンネルのカラー画像を入力に、もう一つはfeature engineeringとDCT係数を入力したもの。
- 3-channel color image model
  - 4クラス分類（プレーン、他3つのアルゴリズムを予測）
  - EfficientNet-b5を使用
    - augmentationはrotateとflipsの他にcutmix
    - 50epochのSGD+Cosine Annealingを4サイクル
    - HyperparametersはRegNet論文のEfficientNetとほぼ同じ
- DCT coefficient model
  - DCT値のone-hot encoding
  - positional encoding（役に立ったかわからない）
    - quantization matrix / 50
    - a matrix such that matrix[i, j] = cos(pi * (i % 8) / 16) * cos(pi * (j % 8) / 16)
  - EfficientNet-b2
    - 最初のstrideを2から1へ
    - Conv2dのdilationを8に変更
    - augmentationはrotateとflips
    - 50epochのSGD+Cosine Annealingを1サイクル
    - strideの変更が効いた
- DCTモデルは画像モデルに比べて精度が低いため、単純に平均するとうまく機能しない。そこでMLPを使用した。
  - 各モデルのsoftmax出力に、スケールの違いをキャンセルするために以下のBatchNormを使用
```python
x = torch.cat([bn(feat.reshape(-1, 1)).reshape(feat.shape) for feat, bn in zip(feats, self.bns)], dim=1)
```
  - さらにLightGBMを追加（MLPでの予測と各CNN特徴マップのt-SNEを特徴として使用）。hyperparameterはLightGBMTunerCVによって決めた。
