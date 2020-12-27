# [Avito Demand Prediction Challenge](https://www.kaggle.com/c/avito-demand-prediction)

## 期間
- 2018/04 - 2018/06

## Overall
- ロシアの個人間の（デジタル）蚤の市での商品取引確率を予測するコンペ。
- 商品に関するデータ（出品される場所、商品カテゴリ、価格、商品画像、紹介文など）、学習用データが1,503,424レコード、テストデータが508,438レコード、PublicとPrivateの比率は31:69
- 評価指標：RMSE
- リークはなく、Shakeもほとんどなかった
- 参加者数：1,871 teams
- ソリューションのより詳細な情報はcopypasteさんの[ブログ](https://copypaste-ds.hatenablog.com/entry/2019/09/08/183255)が参考になります。

## 1st place solution
- ディスカッション：https://www.kaggle.com/c/avito-demand-prediction/discussion/59880
- ソースコード：なし
- スタッキングが非常に重要であったとのこと。
- 最終的には複数のLightGBM, NN, XGBを1層目、さらに複数のLightGBM, NN, XGBを2層目, NNを3層目にしたスタッキングモデル。
- Little Boat part NN
  - 全ての特徴が重要であったがおそらく順番はText, categorical, numerical, images
  - TextへのCNN、Attentionは機能せず
  - 画像に対するfine tuningは機能せず
  - 単語埋め込みとLSTMの間のspatial dropoutは機能
  - SolutionのLinkにネットワークのアーキテクチャあり
- thousandvoices part
  - 5個のLightGBM, 3個のNN, 1個のCatBoost, 1個のRidge
  - 複数のカテゴリカラムの組み合わせ
  - supplementary dataの活用（priceと掲載期間を予測するNNをトレーニングした。これが最も重要な特徴になった）
  - Treeモデルはnum_leavesを大きくするのがポイント（多くのKernelはこれを見逃していた）
- Georgiy Danshchin part
  - 時系列スプリットで検証したところ、CVとLBのギャップが安定していた。チーム参加後にk foldで検証を行うとギャップが大きくなった。user_idのtarget encodingに起因していることに気づき、2層目のスタッキングでのみこれらを使用することにした。
  - train + testで計算したカテゴリのパーセントエンコーディング
  - train_active+test_activeで計算されたuser_idのパーセントエンコーディング
  - 異なるカテゴリのMean target encoding
  - ステミングしたtitleをカテゴリ化
- Arsenal part
  - 50%をTreeモデルの改善、50%をチームのモデルの統合に費やした。
  - 1層目ではXGBoostを6個、LightGBMを13個、2層目ではXGBoostを7個、LightGBMを12個作成。
  - TextではTFIDF, textの統計量
  - ResNet50, InceptionV3, Xceptionからの特徴抽出
  - Countが5,000以上あるCategoryへのTarget encoding
  - 予測したprice, image_top_1, item_seq_number, day_diff
  - 様々なCategoruの平均価格、様々なレベルのdiff
  - チームマージ後に固定したuser_idでの5foldへ
  - HPOはbaysian optimization
  - 多様性を追加するためにマルチクラス分類モデルも追加

## 2nd place solution
- ディスカッション：https://www.kaggle.com/c/avito-demand-prediction/discussion/59871
- ソースコード：なし
- 6層のスタッキングモデル
  - NN, LightGBM, FM_FTRL, Ridge, CatBoost
- CVは10folds
- 特徴量
  - VGG16, ImageNet, ResNet50, MobileNetから画像の特徴抽出
  - テキストへはFasttextを独自学習。これを使用してテキストとcategoryの相互作用を計算。NNへの埋め込みにも使用
  - 各Categoryの組み合わせ（2次、3次）で平均価格、平均掲載日数を計算
  - Categoryのautoencoderでベクトルを抽出。user2vecを作り、user_idをベクトル化


## 3rd place solution
- ディスカッション：https://www.kaggle.com/c/avito-demand-prediction/discussion/59885
- ソースコード：なし
- 標準的な5fold validationと時系列スプリットのモデルのブレンディング（35%が時系列スプリット、65%が5fold）。どちらにもLightGBMとNNが主に使用されている。また5foldではスタッキングも行っている。
  - 他に5foldではExtratTeesRegressorも使用
- これらのモデルの出力を
- LightGBMの特徴量
  - title, descriptionからword(1, 2 grams), char(5 grams)でそれぞれTFIDF特徴量
  - Word2vec, fastTextによる単語のベクトル表現（Word2vecのほうがfastTextよりよかったとのこと）
  - Image quality統計量
  - Vgg16, Vgg19での特徴量
  - ResNet, Inception, Xceptionの予測値
  - 数値変数（priceなど）をbinに区切って離散化した特徴
  - user_idごとの単語数、掲載日などの集約統計量
  - 文章から算出した統計量
  - 緯度/経度などの位置情報
  - objectiveにはxentropyを使用
- NN
  - fasttext(pretrain), word2vec(独自訓練)をconcatしてbidirectional GRU
  - LightGBMで使用したいくつかの数値特徴量
  - カテゴリ変数をemmbeding layerで100次元に圧縮
  - lossはsigmoid出力を使用したbinary cross-entropy
  - textデータはnltkでstemming。title, description, paramsは分割を表す記号を挟んで結合
  - best NN 以外にチャネルごと（画像、テキストなど）にNNを構築。各々のNNの精度が高くなくてもstackingで有効だったとのこと。
