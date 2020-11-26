# [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy)

## 期間
- 2020/03 - 2020/07

## Overall
- ウォルマートの3つの州の10店舗で販売される3049アイテム（計30490アイテム）の28日間の販売数量を予測するコンペ
- 学習データの期間は2011-01-29から2016-05-22、Publicのテストデータは2016-05-23から2016-06-19、コンペ終盤にPublicのテストデータの正解が渡され、さらに28日間の販売数量の予測（Private）を行う。他に日付、イベント有無の入ったカレンダーデータとアイテムの価格のデータが提供されている。
- 評価指標はWRMSSE(Weighted Root Mean Squared Scaled Error)詳細は[こちら](https://mofc.unic.ac.cy/m5-competition/)
- リークはなかったがShakeが大きなコンペで上位は600-2600位ほどSheka upしての入賞であった。
- 参加者数: 5,558teams

## 1st place solution
- ディスカッション：https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684
- ソースコード：なし
- 単純なアプローチとのこと。再帰的パターンと非再帰的パターンのアンサンブル（それぞれがCV、Publicでお互いに優れていたため）
- 前処理
  - 価格に基づく特徴
  - カレンダーに基づく特徴
  - ターゲットラグに基づく特徴（再帰的/非再帰的）
  - ターゲットラグ移動平均/標準（再帰的/非再帰的）に基づく特徴
- CV strategies
  - cv1 : d_1830 ~ d_1857
  - cv2 : d_1858 ~ d_1885
  - cv3 : d_1886 ~ d_1913
  - public : d_1914 ~ d_1941
  - private : d_1942 ~ d_1969
  - without early stopping
- Model
  - LightGBM object=tweedie
- Modeling strategies
  - 再帰的、非再帰的のパターンで
    - ストアIDごと
    - ストアID-商品カテゴリIDごと
    - ストアID-部門IDごと


## 2nd place solution
- ディスカッション：https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/164599
- ソースコード：なし
- ローリングやLag、最終購入日などの特徴を使わなかった。
- ディスカッションで話題になっていた魔法の乗数アプローチ、カスタム損失関数の使用は慎重に捉えた
- LV1-5までのN-Beatsの予測（が正確に見えたので）と予測値が近くなるように15種類の乗数を使ったLightGBMの予測を組み合わせた。
- うまくいかなかったこと
  - ディスカッションのDeepARやR実装のモデル、線形モデルは機能しなかった

## 3rd place solution
- ディスカッション：https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/164374
- ソースコード：なし
- Tweedie lossを使用したDeepAR使用し、過去14日間のWRMSSEを基に複数のモデルのアンサンブルをおこなった。
- ネットワーク
  - ベースラインのネットワークを変更し、学習フェーズでローリング予測を使用
  - Batch size: 64
  - 300 epoch
  - optimizer: Adam
  - learning rate schedule: cosine annealing
- 特徴量（concatして使用）
  - 販売数量
    - ラグ1の値
    - 移動平均7、28日
  - カレンダー：すべての値は[-0.5,0.5]に正規化されます
    - 曜日
    - 月
    - 年
    - 週番号
    - 日
  - イベント
    - イベントタイプ：埋め込みを使用
    - イベント名：埋め込みを使用
  - SNAP：[0、1]
  - 価格
    - 生値
    - 時間の経過とともに正規化
    - 同じdept_id内で正規化
  - カテゴリー
    - state_id、store_id、cat_id、dept_id、item_id：埋め込みを使用
  - 売上ゼロ
    - 今日まで継続的な売上ゼロの日
- CV
  - 1914~1942は他の期間と特性が異なり信頼性が低いと判断(売り上げゼロから開始する商品が多かった)
  - 過去14期間（（1914、1886、1858、…1550）のWRMSSEを評価し、WRMSSEの平均が低いモデルを選択。各モデルについて、10エポック（200〜300エポック）ごとに評価し、選択した。評価トップ3のエポックでトレーニングされた8つのモデルから、最終的な予測のために8×3の選択されたモデルからアンサンブル。
  - 正確には24モデル+19モデルを使用（これらの違いはdropoutの有無）。しかしこれはヒューリスティックな戦略であった。
- うまくいかなかったこと
  - モデル
    - CNN, Transformers
  - 損失関数
    - CE loss
    - 損失として使用したWRMSSE
