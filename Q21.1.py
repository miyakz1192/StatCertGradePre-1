import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import io
    import numpy as np


    return io, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Q21.2のメモ
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [1]について
    """)
    return


@app.cell
def _(io, pd):
    #data = {
    #    "層の大きさ":  [20, 10,   5,     5],
    #    "層内平均":    [15, 150, 510, 1010],
    #    "層内標準偏差":[20, 70, 290, 80]
    #}
    #上記のように指定できるけど、生でCSVを書く手もある。
    table_data = """
    層の大きさ  層内平均  層内標準偏差
    20    15      20
    10   150      70
    5   510     290
    5  1010      80
    """
    #df = pd.DataFrame(data)
    df = pd.read_csv(io.StringIO(table_data.strip()), sep='\s+')
    print(df)

    return (df,)


@app.cell
def _(df, np):
    """
    ポイント
    ベクトル演算: numerator を計算した時点で、全行（h=1〜4）の結果が入ったリストのようなもの（Series）が出来上がります。
    numerator.sum(): これが数式の分母（

    ）に相当します。
    ブロードキャスト: numerator / numerator.sum() と書くだけで、各行の値を合計値で割ってくれます。
    """
    n=8
    # 1. 各行の「分子（N * sigma * sqrt(...)）」を一括計算
    numerator = df['層の大きさ'] * df['層内標準偏差'] * np.sqrt(df['層の大きさ'] / (df['層の大きさ'] - 1))
    # 2. 「分子 / 分子の合計（分母）」を計算
    # これで各レコードの割合（ウェイト）が算出されます
    result = (numerator / numerator.sum())*n

    print(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [2]について
    """)
    return


@app.cell
def _(df, io, pd):
    table_sample_size = """
    n_1  n_2  n_3  n_4
    2     2    2    2
    4     2    1    1
    1     2    4    1
    """
    df_sample_size = pd.read_csv(io.StringIO(table_sample_size.strip()), sep='\s+')
    print(df_sample_size)
    # 1. サンプルサイズテーブルの1行目（インデックス1）を取得
    # .values を使うことで、カラム名を無視して純粋な数値のリストとして扱えます
    sample_size_temp = df_sample_size.loc[1].values
    df["サンプルサイズ"] = sample_size_temp
    print(df)
    return (df_sample_size,)


@app.cell
def _(df, df_sample_size):
    #サンプル総数値(母総数値の推定値)の計算
    for index ,row in df_sample_size.iterrows():
        df["サンプルサイズ"] = row.values
        #print(df)
        print(((df["層の大きさ"] ** 2) * ((df["層の大きさ"]-df["サンプルサイズ"])/(df["層の大きさ"]-1)) * ((1/df["サンプルサイズ"])*(df["層内標準偏差"]**2))).sum())
    return


if __name__ == "__main__":
    app.run()
