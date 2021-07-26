# [Bristol-Myers Squibb – Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation)

## 期間
- 2021/03 - 2021/06

## Overall
- 分子構造画像から分子構造式を出力する画像コンペ
- 学習データは2,424,186枚、テストデータが1,616,107枚で、PublicとPrivateの比率は25:75
- 評価指標：Levenshtein distance
- Solutionはモデルアンサンブルと各チーム独自の後処理が主であった。リークやshakeはなかった。

## 1st place solution
- 共有なし

## 2nd place solution
- 共有なし

## 3rd place solution
- ディスカッション：https://www.kaggle.com/c/bms-molecular-translation/discussion/243932
- ソースコード：一部あり
  - [[LB0.87]Part of 3rd place solution](https://www.kaggle.com/bamps53/lb0-87-part-of-3rd-place-solution)
- Phase1: InChI candidates reranking
  - 学習データとテストデータの傾向の違いを小さくするために、学習時にソルトアンドペッパーノイズを追加して学習した。
- Phase2: InChI candidates reranking
  - Phase3に向けて、多様なモデルを作成する。
    - Swin Transformer + BERT Decoder (by KF & lyakaap)
    - Transformer in Transformer + BERT Decoder (by KF)
    - EfficientNet-v2 followed by ViT + BERT Decoder (by lyakaap)
    - EfficientNet-B4 + Transformer Decoder (image_size:416x736) (by camaro)
    - EfficientNet-B4 + Transformer Encoder/Decoder(image_size:300x600) (by camaro)
- Phase3: InChI candidates reranking
  - Phase2で生成した結果のリランキング。
    - rdkit.Chem.MolFromInchi 関数を使用して、各InChI候補の検証を行う。(is_valid)
    - 各InChI候補について、複数のモデルの学習に用いた損失（cross entropy / focal loss）を計算し、モデル間で平均化する。(loss)
    - 候補を "is_valid" の降順 → "loss" の昇順でソートし、最も高いスコアの InChI を最終出力とする。
- うまくいかなかったこと
  - yolov5を使った分子結合の物体検出
    - Dacon.aiの1位のソリューションのようなInChIの再構築
    - 検出結果を追加の入力チャンネルとして使用
  - extra_approved_InChIs.csvを使ったMLMの事前学習
  - InChI候補のランク学習
    - ビームサーチで作られた偽物を負例として、本物/偽物の分類とlevenshteinの距離予測を行う。

## 4th place solution
- ディスカッション：https://www.kaggle.com/c/bms-molecular-translation/discussion/243787
- ソースコード：なし
- モデル構造
  - CNN → Encoder → Decoder
  - CNNのバックボーンはresnest101とefficientnetv2_m
  - Encoderは6層のTransformer Encoder
  - Decoderは9層のTransformer Decoder
  - CNNからの埋め込みをTransformerの次元に合わせて512チャンネルに投影する。Encoderでは、DETRと同様に、keyとqueryの前に正弦波の位置埋め込みを各層に挿入する。Decoderでは、T5と同様に相対位置埋め込みを使用。
- モデルの学習
  - GPUリソースから大規模なモデルをスクラッチでトレーニングすることはしなかった。まず416x416の画像でモデルを学習し、その後、画像の解像度を上げてモデルを微調整した。2%のデータ、約48kを検証用に確保している。
  - pseudo labelingも行っており、非常に有用であることがわかった。0.64CV/0.78LBのサブミッション（rdkit ensemble of efv2 on 416 and 640）を使って疑似ラベルを生成し、その中から約130万個のサンプルを選びました。これらのサンプルを学習用データと結合する。
  - 学習の順序
    - efficientnetv2_m: 416(10エポック)→640(5エポック)→704(5エポック+擬似ラベル)→ 384×768(5エポック+擬似ラベル)
    - resnest101: 416(10エポック)→640(5エポック+擬似ラベル)→384×768（5エポック+疑似ラベル
- アンサンブル
  - LBが0.66-0.67のモデルを4つ使って、step-wise logit ensembleを作り、ビームサイズ3のビームサーチを使用。最尤を選択する代わりに、rdkitから最初の有効な予測を選びます。このようにして、CVを約0.02改善することができた。4つのモデルのアンサンブルは、LBで0.56を達成。CVは0.45程度になるはず。
  - これに加えて、初期のサブミッション（スコアが0.65以上）とrdkitによる0.56の投稿をマージしたところ、0.55になった。
  - 最後に、サブミットした3500件の無効な予測を選択し、サイズ32のビームサーチを行った。これにより、LBスコアは0.54に減少した。

## 5th place solution
- ディスカッション：https://www.kaggle.com/c/bms-molecular-translation/discussion/243943
- ソースコード：なし
- 全体像
  - Transformer Encoderを使わず、CNN(384x384 & 512x512) + Transformer Decoder (12~16層)というアーキテクチャを採用した。
  - 5つのバックボーンで学習した。(EffNet B3/B5/B7, ResNet200D, eca-nfnet-l0)
  - 正則化のためのノイズインジェクション
  - rdkitで生成した外部データを使って学習した。(しかし、あまり役に立たなかった)
  - デコード時のアンサンブル
  - 無効な予測のハンドリング（0.6以下にするための最も重要なトリック）。
- ノイズインジェクション
  - Exposure Bias対策、学習時にGTの文字をランダムに別の文字に置き換える。置き換えられた文字は損失の計算では無視されるが、モデルは前の文字が間違っていたという前提で、次の文字を正しく予測しなければならなくなる。
- 公式外部データ
  - 運営は公式外部データとして画像のないInChIを1000万枚公開した。rdkitはInChIを使って画像を生成できることがわかったので、その画像の一部（約100万〜200万）を外部データとして生成し、学習に加えた。しかし、後になって、この部分のデータではあまり改善されないことがわかった。しかし、性能を落とすこともなかったので、そのままにしておいた。
- 無効な予測のハンドリング
  - すべてのモデルを学習した後、デコードフェーズでアンサンブルを行った。元の予測のLBは0.62ですが、rdkitを使って予測を正則化し、LB0.60を得た。この時点で、予測されたInChIの中に有効でないレコードが約14,000あることがわかった。これらの無効な予測を置き換えるために、3つの方法で有効な予測を見つける。
    - 単一のモデルで生成された予測(固定の5000レコード)
    - アンサンブルによる上位12個の予測値の検索（固定の2500レコード)
    - 化学式の数値部分については、上位2つの予測値を使ってデコードする（500レコード）。
      - 例えば、元の予測値がC12H3で、2文字目と4文字目のtop2予測値が11と4だった場合、C11H3、C12H4と置き換えて次の部分をデコードします。
    - 無効な予測を有効な予測に置き換えることで、LBを大幅に改善できることを発見したのは、コンペ締め切り前夜のことでした。時間がなかったため、元の予測の無効な予測14000行のうち、8000行を処理することしかできませんでしたが、LBスコアは0.60から0.57に改善されました。
