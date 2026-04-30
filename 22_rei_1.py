import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import marimo as mo
    import pandas as pd
    import io
    import numpy as np

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    return PCA, StandardScaler, io, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    このノートは、P194の例1と、P197の例2を載せている。
    """)
    return


@app.cell
def _(io, pd):
    #   国語　数学　理科　社会
    test_data = """
        x_1  x_2  x_3 x_4
        2     2     3   1
        9     8     10  9
        8     3     2   7
        7     1     3   8
        2     9     8   2
        5     4     5   5
    """
    df_test_data = pd.read_csv(io.StringIO(test_data.strip()), sep='\s+')
    print(df_test_data)
    return (df_test_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    国語の分散と国語と数学の共分散をライブラリを用いずに計算してみる
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(df_test_data):
    def mu_test():
        mu_x_1 = (df_test_data["x_1"].sum()) / len(df_test_data["x_1"])
        print(mu_x_1)
        # 列ごとの平均はこんなふうにベタに書くこともできる。
        for col in df_test_data.columns:
            print(col)
            print((df_test_data[col].sum()) / len(df_test_data[col]))

    mu_test()
    return


@app.cell
def _(df_test_data):
    def mu_vect_test():
        #しかし、こんなふうに一発で書くこともできる。
        mu_vect = df_test_data.mean()
        print(mu_vect)
        #ベタ書きとあっているね。

    mu_vect_test()
    return


@app.cell
def _(df_test_data):
    #分散を２通りで求めてみる
    #簡単に書く方法と、ベタな方法
    def variance_test(col = "x_1"):
        #簡単な方法
        v1 = df_test_data[col].var()
        print(v1)
        #ベタな方法
        mu = df_test_data[col].mean()
        print(mu)
        #偏差ベクトルを得る
        deviation_vec = df_test_data[col] - mu
        print(deviation_vec)
        #偏差ベクトルを二乗する
        deviation_pow_vec = deviation_vec ** 2
        #その和を取り、n-1で割る
        n_1 = len(deviation_vec)-1
        v2 = deviation_pow_vec.sum() / n_1
        print(v2)
        #簡単な方法とあっている。
        return v2

    print(f"国語の分散={variance_test("x_1")}")
    return


@app.cell
def _(df_test_data):
    #次は共分散を求める。同じように簡単にやるやつと、ベタなやつ
    def covariance_test(col1="x_1", col2="x_2", disp=False):
        #簡単な方法
        cov1 = df_test_data[col1].cov(df_test_data[col2])
        if disp: print(cov1)

        #ベタな方法
        #最初にcol1とcol2の偏差ベクトルを得る
        col1_deviation_vec = df_test_data[col1] - df_test_data[col1].mean()
        col2_deviation_vec = df_test_data[col2] - df_test_data[col2].mean()
        #次にそれらの積のベクトルを得る
        cov2_deviation_mul_vec = col1_deviation_vec * col2_deviation_vec
        #それの和をとり、n-1で割る
        n_1 = len(cov2_deviation_mul_vec) - 1 
        cov2 = cov2_deviation_mul_vec.sum() / n_1
        if disp: print(cov2)

        return cov2

    print(f"国語と数学の共分散={covariance_test("x_1", "x_2",disp=True)}")
    return (covariance_test,)


@app.cell
def _(covariance_test, df_test_data, pd):
    #次は標本分散共分散行列と、標本相関行列の練習
    def cov_matrix_test():
        print("一番簡単な方法")
        #一番簡単な方法
        cov_mat1 = df_test_data.cov()
        print(cov_mat1)

        print("ベタな方法1")
        #ベタな方法1
        ##中心化する
        df_test_data_meaned = df_test_data - df_test_data.mean()
        ##転置する
        df_test_data_meaned_T = df_test_data_meaned.T
        print(df_test_data_meaned)
        print(df_test_data_meaned_T)
        ###行列の積をとり、n-1で割る
        n_1 = len(df_test_data_meaned) - 1
        print(n_1)
        cov_mat2 = (df_test_data_meaned_T @ df_test_data_meaned) / n_1
        print(cov_mat2)

        print("超絶ベタな方法2")
        cols = df_test_data.columns
        rows = range(len(df_test_data))
        cov_mat3 = pd.DataFrame(index=rows, columns=cols)
        print(cov_mat3)
        for c in df_test_data.columns:
            for r in df_test_data.columns:
                cov_mat3.at[c, r] = covariance_test(c, r)

        print(cov_mat3)

    cov_matrix_test()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    標本相関行列の練習
    """)
    return


@app.cell
def _(df_test_data, np):
    def corrcoef_test():
        #超かんたんな方法
        corrcoef_mat1 = df_test_data.corr()
        print(corrcoef_mat1)

        #geminiに教えてもらった方法
        corrcoef_mat1 = np.corrcoef(df_test_data.T)
        print(corrcoef_mat1)

        #計算の理解のために、素朴な方法にトライ。
        df_test_data_meaned = (df_test_data - df_test_data.mean())/np.sqrt(df_test_data.var())
        ##転置する
        df_test_data_meaned_T = df_test_data_meaned.T
        #print(df_test_data_meaned)
        #print(df_test_data_meaned_T)
        ###行列の積をとり、n-1で割る
        n_1 = len(df_test_data_meaned) - 1
        #print(n_1)
        cov_mat2 = (df_test_data_meaned_T @ df_test_data_meaned) / n_1
        print(cov_mat2)

    corrcoef_test()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    固有値問題を解く
    """)
    return


@app.cell
def _(df_test_data, np):
    def solve_eigenvalue():
        data = df_test_data
        print(data)
        # 1. データの準備（標準化済みと仮定）※ここがちょっとおかしいか。そもそも標準化していない。
        # 2. 相関係数行列（分散共分散行列）を作成
        cor_matrix = np.corrcoef(data.T)

        # 3. 固有値問題の解決
        eigen_values, eigen_vectors = np.linalg.eigh(cor_matrix)

        # 固有値の大きい順にソート（eighは昇順で返ることが多いため）
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        print(idx)
        print(eigen_values)
        print(eigen_vectors)

    solve_eigenvalue() #しかし、これだと、なぜか教科書の固有値20.2,19.4,0.85,0.18と合わない。
    return


@app.cell
def _(df_test_data, np):
    def solve_eigenvalue2():
        data = df_test_data
        print("入力データ")
        print(data)
        print("入力データ.T")
        print(data.T)
        # 1. データの準備（標準化済みと仮定）
        # 2. 分散共分散行列を作成
        cor_matrix = np.cov(data.T) #転置する必要がある。
        #cor_matrix = np.cov(data)

        # 3. 固有値問題の解決
        eigen_values, eigen_vectors = np.linalg.eigh(cor_matrix)

        # 固有値の大きい順にソート（eighは昇順で返ることが多いため）
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        print(idx)
        print(eigen_values)
        print(eigen_vectors)
        print("参考までに転置する→列で見ると表22.2と同じデータ（絶対値でみる）")
        print(', '.join(df_test_data.columns.values))
        print(eigen_vectors.T)

    solve_eigenvalue2() #これだと教科書通りの固有値が得られた。np.covを使っているのが教科書の計算の進め方(np.corrcoefで標準化はしていない)。
    #ちなみに、得られた結果が教科書とは違う。絶対値はあっている感じだが、符号が違う。
    #これに対するgeminiの解説は以下。
    #実は、固有ベクトルは「向き」を表すものなので、プラスとマイナスが全て逆転して出力されることがありますが、数学的にはどちらも正解です。（例：NumPyで [0.5, -0.5] と出たものが、Scikit-learnで [-0.5, 0.5] と出ても、同じ直線を指しているので間違いではありません。）
    #ってことで、絶対値としては、教科書の表22.2固有ベクトルと同じである。
    #符号が違うのだが、これは行で見て-1倍して等しい状態になっていればOK。ほしいのは軸であり方向は不要。180度ぐるっと回して同じ方向であればOKってこと。
    #つまり、solve_eigenvalue2では
    #[-0.37027138 -0.5572669  -0.61056538 -0.42374829]
    #これはPCA()で求めたものの-1倍になっている。
    #第1主成分  0.370271  0.557267  0.610565  0.423748
    #なので問題なし。
    #ちなみに、第1~4主成分それぞれが、固有ベクトルになることに注意！
    return


@app.cell
def _(df_test_data, np):
    #gemini実装（参考）
    def solve_eigenvalue3():
        data = df_test_data

        # ddof=0 を指定して標本分散（分母n）にする
        cov_matrix = np.cov(data.T, ddof=0)

        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        # 並び替え（列を入れ替える）
        eigen_vectors = eigen_vectors[:, idx]

        print("固有値 (PCAの結果と一致するはず):")
        print(eigen_values)

        print("\n固有ベクトル (行が各主成分):")
        # PCA().components_ と同じ形式にするために転置
        print(eigen_vectors.T)

    solve_eigenvalue3()
    return


@app.cell
def _(PCA, df_test_data, pd):
    def using_PCA():

        # 1.データの準備
        data = df_test_data

        # 2-1. データの標準化（平均0, 分散1に変換）
        # 相関係数行列の固有値問題にするために必須の工程です
        #scaler = StandardScaler()
        #standardized_data = scaler.fit_transform(data) # ここで変数を定義！
        # 2-2. データを標準化しない場合
        standardized_data = data

        # 3. PCAの実行
        pca = PCA()
        pca.fit(standardized_data)

        # 結果の表示
        print("固有値（各主成分の分散）:")
        print(pca.explained_variance_)

        print("\n寄与率:")
        print(pca.explained_variance_ratio_)

        # --- ここから追加：主成分ウェイト（固有ベクトル）の表示 ---
        print("\n主成分ウェイト（固有ベクトル）の表:")
        loadings_df = pd.DataFrame(
            pca.components_.T, 
            index=data.columns, 
            columns=[f"第{i+1}主成分" for i in range(len(data.columns))]
        )
        print(loadings_df.T) #教科書に合わせて見やすくするように、転置を行う。
        # ---------------------------------------------------

        return pca

    pca = using_PCA()
    return (pca,)


@app.cell
def _(df_test_data, np, pca, pd):
    def calculate_pc_scores(pca):
        # 1. データの準備（平均を引いて「中心化」するのが鉄則）
        # ※PCAはデータの中心を原点に移動してから回転させる処理だからです
        centered_data = df_test_data - df_test_data.mean()

        # 2. 固有ベクトル（主成分ウェイト）を取得
        # 前述の通り、行列計算のために「列ベクトル」の状態にする
        W = pca.components_.T 

        # 3. 行列の掛け算（ドット積）
        # (6人x4変数) × (4変数x4主成分) = (6人x4主成分の得点)
        pc_scores = np.dot(centered_data, W)

        # 見やすくデータフレーム化
        pc_scores_df = pd.DataFrame(
            pc_scores,
            columns=[f"第{i+1}主成分得点" for i in range(W.shape[1])]
        )

        print("各個人の主成分得点:")
        print(pc_scores_df)

    calculate_pc_scores(pca)
    return


@app.cell
def _(df_test_data, np, pca, pd):
    def run_principal_component_loading(pca):
        # 1. 固有値の平方根（標準偏差）を求める
        eigen_stds = np.sqrt(pca.explained_variance_)

        # 2. 固有ベクトルに固有値の平方根を掛ける
        # pca.components_ (固有ベクトル) の各行に、対応する標準偏差を掛ける
        loadings = pca.components_.T * eigen_stds

        # 3. データフレーム化
        loadings_df = pd.DataFrame(
            loadings,
            index=df_test_data.columns,
            columns=[f"第{i+1}主成分負荷量" for i in range(len(eigen_stds))]
        )

        print("主成分負荷量の表:")
        #print(loadings_df)
        print(loadings_df.T)

    run_principal_component_loading(pca)
    #この結果は教科書とは異なる。
    return


@app.cell
def _(PCA, StandardScaler, df_test_data, np):
    def run_principal_component_loading2():
        # 標準化したデータでのPCAをやり直す
        scaler = StandardScaler()
        std_data = scaler.fit_transform(df_test_data)

        pca_std = PCA()
        pca_std.fit(std_data)

        # 標準化ベースの主成分負荷量を計算
        std_loadings = pca_std.components_.T * np.sqrt(pca_std.explained_variance_)
        print(std_loadings)
        print(std_loadings.T)
        print(std_loadings[0, 0]) # 第1主成分のx_1

    run_principal_component_loading2()
    #これでも合わない。
    return


@app.cell
def _(df_test_data, np, pca, pd):
    def run_principal_component_loading3():

        # 1行目(x=2,2,3,1)を除いた5人分のデータにする
        df_sub = df_test_data.iloc[1:] 


        # 1. 元の各変数の標準偏差を求める
        # ※分散共分散行列(ddof=0)ベースなので、標準偏差もddof=0で計算
        stds = df_test_data.std(ddof=0)
        # 不偏標準偏差を求める
        stds = df_sub.std(ddof=1)

        # 2. 固有値の平方根を取得
        eigen_stds = np.sqrt(pca.explained_variance_)

        # 3. 主成分負荷量の計算
        # (固有ベクトル * √固有値) / (元の変数の標準偏差)
        loadings = (pca.components_.T * eigen_stds) / stds.values.reshape(-1, 1)

        # データフレームで表示（第1列が第1主成分負荷量）
        loadings_df = pd.DataFrame(
            loadings,
            index=df_test_data.columns,
            columns=[f"第{i+1}主成分負荷量" for i in range(len(eigen_stds))]
        )
        #print(loadings_df)
        print(loadings_df.T)

    run_principal_component_loading3()
    #まだ教科書の表の数値とは乖離があるんだよな。。。
    return


@app.cell
def _(df_test_data, np, pd):
    def solve_textbook_loadings():
        # 1. データの準備（6行すべて使用）
        data = df_test_data 

        # 2. 分散共分散行列を「不偏分散(ddof=1)」で作成
        # np.covのデフォルトは ddof=1 ですが、明示します
        cov_matrix = np.cov(data.T, ddof=1)

        # 3. 固有値・固有ベクトルの算出
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

        # 降順ソート
        idx = eigen_values.argsort()[::-1]
        evals = eigen_values[idx]
        evecs = eigen_vectors[:, idx]

        # 4. 元の変数の「不偏標準偏差(ddof=1)」を算出
        # ここを ddof=1 にするのが最大のポイントです
        stds = data.std(ddof=1).values

        # 5. 主成分負荷量の計算
        # 公式: (固有ベクトル * sqrt(固有値)) / 元の標準偏差
        # evecs[変数, 主成分] * sqrt(evals[主成分]) / stds[変数]
        loadings = (evecs * np.sqrt(evals)) / stds[:, np.newaxis]

        # 結果の表示
        loadings_df = pd.DataFrame(
            loadings,
            index=data.columns,
            columns=[f"第{i+1}主成分負荷量" for i in range(len(evals))]
        )

        print("固有値 (教科書の20.2, 19.4... と一致):")
        print(evals)
        print("\n主成分負荷量の表 (x_1の第1主成分が0.551になるはず):")
        print(loadings_df)
        print(loadings_df.T)

    solve_textbook_loadings()
    return


if __name__ == "__main__":
    app.run()
