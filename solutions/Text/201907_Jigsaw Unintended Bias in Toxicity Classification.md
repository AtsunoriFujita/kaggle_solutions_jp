# [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview)

## 期間
- 2019/05 - 2019/07

## Overall
- テキストデータに対する2値分類。評価指標がOverall AUCとBias AUCの重み付け（[参考リンク](https://arxiv.org/abs/1903.04561)）とコンペ独自のメトリックを採用。
- データはオンライン上でのニュースに対するコメントなど、学習データは約180万レコードでPublicのテストデータは約10万件。2ステージ制で推論のみカーネルで行う必要のあるNotebookコンペ（当時のKernelコンペ）であった。
- テキストデータへのMLモデルがLSTMからBERTへと変わっていったコンペ。リークやshakeはなかった。上位のソリューションはBERTやLSTMのアンサンブルで損失関数が独自のものが多かった。

## 1st place solution
- ディスカッション：https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/103280
- プレゼン資料:https://docs.google.com/presentation/d/1Km1C8jKpofoRRnWRMka-SGrfRNd8jlp6TuasqVKY91E/edit#slide=id.p
- XLNet×2, BERT×2, GPT2(medium)のRankアベレージによるアンサンブル
- BERT, XLNet, GPT2に対してCustom head（MLP）を使用。
- Custom loss
```python
def compute(predictions, labels, subgroups, power=5.0, score_function=F.binary_cross_entropy_with_logits):
    subgroup_positive_mask = subgroups &amp; (labels.unsqueeze(-1) >= 0.5)
    subgroup_negative_mask = subgroups &amp; ~(labels.unsqueeze(-1) >= 0.5)
    background_positive_mask = ~subgroups &amp; (labels.unsqueeze(-1) >= 0.5)
    background_negative_mask = ~subgroups &amp; ~(labels.unsqueeze(-1) >= 0.5)

    bpsn_mask = (background_positive_mask | subgroup_negative_mask).float()
    bnsp_mask = (background_negative_mask | subgroup_positive_mask).float()
    subgroups = subgroups.float()

    bce = score_function(predictions,labels, reduction="none")

    sb = (bce.unsqueeze(-1) * subgroups).sum(0).div(subgroups.sum(0).clamp(1.)).pow(power).mean().pow(1/power)
    bpsn = (bce.unsqueeze(-1) * bpsn_mask).sum(0).div(bpsn_mask.sum(0).clamp(1.)).pow(power).mean().pow(1/power)
    bnsp = (bce.unsqueeze(-1) * bnsp_mask).sum(0).div(bnsp_mask.sum(0).clamp(1.)).pow(power).mean().pow(1/power)
    loss = (bce.mean() + sb + bpsn + bnsp) /4
    return loss
```
- SWAとcheckpoint ensemble


## 2nd place solution
- ディスカッション：https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100661
- ソースコードなし
- モデルは, BERT-base-uncased, BERT-base-cased, BERT-large-uncased, BERT-large-cased, OpenGPT, OpenGPT2-small, OpenGPT2-medium, XLNet-large-casedをMax length=256で使用。
- Custom loss
```python
identity_columns = [
   'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
   'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
subgroup_bool = (train_df[subgroups].fillna(0) >=0.5).sum(axis=1) > 0
positive_bool = train_df['target'] >= 0.5
# Overall
weights = np.ones(len(train_df)) * 0.25
# Backgroud Positive and Subgroup Negative
weights[((~subgroup_bool) &amp; (positive_bool)) | ((subgroup_bool) &amp; (~positive_bool))] += 0.25
# Subgroup
weights[(subgroup_bool)] += 0.25
```
```python
# toxic_logits size: (batch_size x 11)
# idnet_logits size: (batch_size x 9)
toxic_logits, ident_logits = model(input_ids, segment_ids, input_mask)
toxic_loss = (F.cross_entropy(toxic_logits, toxic_target, reduction='none') * weight).mean()
ident_loss = F.binary_cross_entropy_with_logits(ident_logits, ident_target)
loss = 0.75*toxic_loss + 0.25*ident_loss
```
- 予測値を11個に分割し、以下のように計算
```python
cls_vals = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
for cls, v in enumerate(cls_vals):
    if target >= v:
        target = cls
```
```python
toxic_logits = toxic_logits[0] # the probability distribution of 11 classes
y_pred = sum([p*v for p, v in zip(toxic_logits, cls_vals)])
```
- ブレンディングは[こちら](https://medium.com/data-design/reaching-the-depths-of-power-geometric-ensembling-when-targeting-the-auc-metric-2f356ea3250e)を参考に以下のように重み付け
```python
p = 3.5
bert-base-cased**p * 0.25 +
bert-base-uncased**p * 0.25 +
bert-large-cased**p * 1.0 +
bert-large-uncased**p * 1.0 +
opengpt2**p * 1.0 +
opengpt2-medium**p * 2.0 +
openai-gpt**p *0.5 +
xlnet-large**p *1.0
```

## 3rd place solution
- ディスカッション：https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471
- ソースコード（の一部）：https://www.kaggle.com/sakami/single-lstm-3rd-place
- LSTM, BERT, GPT2のアンサンブル
- LSTMには4つのモデル（LSTM+GRU×2, LSTM+Capsule+Attention, LSTM+Conv）を使用
  - 学習は5fold, 6epoch
  - EMAとcheckpoint ensembleを使用
  - Pseudo labeling
  - GloveとFasttextの和を埋め込み
  - Target weight multiplied by log(toxicity_annotator_count + 2)
```python
  weights = np.ones((len(train),))
  weights += train[identity_columns].fillna(0).values.sum(axis=1) * 3
  weights += train['target'].values * 8
  weights /= weights.max()
```
- BERT、GPT2には[Head + tail tokens](https://arxiv.org/abs/1905.05583), [Dynamic learning rate decay](https://arxiv.org/abs/1905.05583)を使用し, 前回のtoxicコンペのデータでのファインチューニングを実施
- Optunaを使いブレンディングウェイトを決定
- pre-training, BERTへのPseudo labelingはうまく機能しなかった。
