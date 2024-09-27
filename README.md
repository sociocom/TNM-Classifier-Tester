# TNM-Classifier

BERT で TNM 分類を行うプログラムです．[NTCIR-17 MedNLP-SC Radiology Report Subtask](https://repository.nii.ac.jp/records/2001285)で取り組んだコードを元にしたプログラムです．
[rye](https://rye.astral.sh/guide/installation/)で python のパッケージを管理しています．

## データの概要

入力 (data/)
| id | T | N | M | text |
| --- | --- | --- | --- | ---------------------------------------------------------------------------------------- |
| 0 | 4 | 3 | 0 | 左上葉気管支は閉塞して造影 CT で増強効果の乏しい 70mm の腫瘤があります。肺癌と考えます。 |

出力 (outputs/)
| id | T | N | M | text | TNM_pred |
| --- | --- | --- | --- | ---------------------------------------------------------------------------------------- | --- |
| 0 | 4 | 3 | 0 | 左上葉気管支は閉塞して造影 CT で増強効果の乏しい 70mm の腫瘤があります。肺癌と考えます。 | 420 |

## 使用方法

[Mecab 用辞書万病辞書](https://sociocom.naist.jp/j-meddic-for-mecab/)をダウンロードし，data ディレクトリ直下に配置する．

Rye を導入した後，rye sync で python ライブラリをインストール<br>

下記のコマンド一覧を参考に適切な引数を追加し，実行する．<br>
実行例

```
rye run python src/main.py -i='data/tnm_report.csv' -d='cpu' -b=8
```

出力ファイルは output ディレクトリ下に生成される．

## コマンド一覧

help が見れるコマンド: `rye run python src/main.py --help` <br>
入力すると以下の説明が表示される．

```
NAME
    main.py

SYNOPSIS
    main.py <flags>

FLAGS
    -p, --pretrained_model=PRETRAINED_MODEL
        Type: str
        Default: 'model'
    -i, --input_path=INPUT_PATH
        Type: str
        Default: 'data/tnm_report.csv'
    -o, --option_dic_path=OPTION_DIC_PATH
        Type: str
        Default: 'data/MANBYO_20190...
    -s, --seed=SEED
        Type: int
        Default: 2023
    -d, --device=DEVICE
        Type: str
        Default: device(type='cuda', ind...
    -b, --batch_size=BATCH_SIZE
        Type: int
        Default: 32
    -m, --max_length=MAX_LENGTH
        Type: int
        Default: 512
```
