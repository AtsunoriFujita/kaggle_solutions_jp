# [chaii - Hindi and Tamil Question Answering](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/overview)

## 期間
- 2021/08 - 2021/11

## Overall
- ヒンディー語とタミル語の読解(Reading Comprehension)タスク
- 学習用データは1,114件、テストデータは2,000件程度（Notebookコンペ）PublicとPrivateの比率は40:60
- word-level Jaccard score
- 外部データとしてTyDi QAを使用していたかが明暗を分けた（このデータを翻訳していたものがテストデータで使用されていた）。
- 943チーム参加

## 1st place solution
- ディスカッション: https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287923
- ソースコード
  - https://github.com/kldarek/chaii
  - https://www.kaggle.com/thedrcat/chaii-ensemble-10postuning
- XLM-Roberta Large, MURIL Large, Rembertを使用
  - XLM-Roberta: gradient accumulation: 4, learning rate: 1e-5, wd: 0.01
  - MURIL: gradient accumulation: 4, learning rate: 3e-5, wd: 0.01
  - Rembert: gradient accumulation: 16, learning rate: 1e-5, wd: 0.01
- 学習用データにTyDi QA(English, Bengali and Telugu)を使用
- CVで使われるようなテクニックをNLPへ適用


## 2nd place solution
- ディスカッション: https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287917
- ソースコード
  - https://www.kaggle.com/kanbehmw/tkm-tydi-rem-info-muril-top35
- 学習データとMLQAとTyDi QA (only Bengali and Telugu)を使用してトレーニング
- 全データを使用して2エポック学習
- CE lossとJaccard-based Soft Labels Since Cross-Entropyを使用
- モデル
  - XLM-Robertaを7モデル
  - RemBertを3モデル
  - InfoXLMを3モデル
  - MuRILを2モデル
- 後処理
  - Devanagari numbersをアラビア数字へ

## 3rd place solution
- ディスカッション: https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287929
- ソースコード
  - https://www.kaggle.com/nguyenduongthanh/simple-train
  - https://www.kaggle.com/nguyenduongthanh/simple-10-folds-of-muril-large
  - https://www.kaggle.com/nguyenduongthanh/fine-tune-muril-squad-v2
  - https://www.kaggle.com/nguyenduongthanh/muril-large-squad-v2
- SQUAD v2でファインチューニングしたMuril-Large
- 10 folds, max-length=384, doc_stride=128, lr=1e-5, gradient accumulation=1
