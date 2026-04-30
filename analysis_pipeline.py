from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "第一步_清洗后建模样本.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

COL_MAP = {
    "省份": "province",
    "年份": "year",
    "所属区域": "region",
    "推荐建模样本标记": "model_sample",
    "基准因变量": "y_base",
    "稳健性因变量_住院率": "y_hosp",
    "稳健性因变量_两周就诊率": "y_2week",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "O3": "o3",
    "CO": "co",
    "ln_常住人口": "ln_pop",
    "ln_人均GDP": "ln_gdp",
    "城镇化率": "urban_rate",
    "第二产业占比": "industry2_share",
    "医疗卫生机构数_个": "medical_orgs",
    "医院数_个": "hospitals",
}

POLLUTANTS = ["pm25", "pm10", "so2", "no2", "o3"]
CONTROLS = ["ln_pop", "ln_gdp", "urban_rate", "industry2_share", "medical_orgs"]
Y_LIST = ["y_base", "y_hosp", "y_2week"]

CN = {
    "y_base": "基准健康负担代理指标",
    "y_hosp": "住院率",
    "y_2week": "两周就诊率",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "so2": "SO2",
    "no2": "NO2",
    "o3": "O3",
    "PC1": "综合污染因子 PC1",
    "PC2": "污染结构因子 PC2",
}


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def result_table(model, variables, model_name, y_name):
    rows = []
    for var in variables:
        rows.append(
            {
                "模型": model_name,
                "因变量": CN.get(y_name, y_name),
                "变量": CN.get(var, var),
                "变量名": var,
                "系数": model.params.get(var, np.nan),
                "聚类稳健标准误": model.bse.get(var, np.nan),
                "p值": model.pvalues.get(var, np.nan),
                "显著性": stars(model.pvalues.get(var, np.nan)),
                "R2": model.rsquared,
                "样本量": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def fit(formula, df):
    return smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["province"]})


def main():
    raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = raw.rename(columns=COL_MAP).copy()
    if "model_sample" in df.columns:
        df = df[df["model_sample"] == 1].copy()

    profile = []
    profile.append("# 数据自动诊断报告\n")
    profile.append(f"- 观测数：{raw.shape[0]}\n")
    profile.append(f"- 字段数：{raw.shape[1]}\n")
    profile.append(f"- 省份数：{raw['省份'].nunique()}\n")
    profile.append(f"- 年份范围：{raw['年份'].min()}—{raw['年份'].max()}\n")
    profile.append(f"- 区域分类：{', '.join(raw['所属区域'].dropna().unique())}\n")
    profile.append("\n## 建议用途\n")
    profile.append("该数据适合用于省级面板数据的描述性统计、污染物相关性分析、固定效应回归、稳健性检验、主成分分析和区域异质性分析。\n")
    (OUT / "data_profile.md").write_text("".join(profile), encoding="utf-8")

    raw.describe(include="all").transpose().to_csv(OUT / "descriptive_statistics.csv", encoding="utf-8-sig")
    raw.isna().sum().reset_index().rename(columns={"index": "字段", 0: "缺失数量"}).to_csv(OUT / "missing_check.csv", index=False, encoding="utf-8-sig")

    year_cols = ["基准因变量", "稳健性因变量_住院率", "稳健性因变量_两周就诊率", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
    raw.groupby("年份")[year_cols].mean(numeric_only=True).to_csv(OUT / "summary_by_year.csv", encoding="utf-8-sig")
    raw.groupby("所属区域")[year_cols].mean(numeric_only=True).to_csv(OUT / "summary_by_region.csv", encoding="utf-8-sig")

    corr_cols = ["基准因变量", "稳健性因变量_住院率", "稳健性因变量_两周就诊率", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "ln_常住人口", "ln_人均GDP", "城镇化率", "第二产业占比", "医疗卫生机构数_个"]
    raw[corr_cols].corr(numeric_only=True).to_csv(OUT / "correlation_matrix.csv", encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    trend = raw.groupby("年份")[["PM2.5", "PM10", "SO2", "NO2", "O3"]].mean(numeric_only=True)
    for col in trend.columns:
        plt.plot(trend.index, trend[col], marker="o", label=col)
    plt.xlabel("Year")
    plt.ylabel("Mean value")
    plt.title("Pollutant Trends by Year")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pollutant_trends.png", dpi=200)
    plt.close()

    results = []
    for y in Y_LIST:
        rhs = POLLUTANTS + CONTROLS
        ols_formula = f"{y} ~ " + " + ".join(rhs) + " + C(year)"
        fe_formula = f"{y} ~ " + " + ".join(rhs) + " + C(province) + C(year)"
        results.append(result_table(fit(ols_formula, df), POLLUTANTS, "OLS + 年份效应", y))
        results.append(result_table(fit(fe_formula, df), POLLUTANTS, "双向固定效应", y))
    pd.concat(results, ignore_index=True).round(6).to_csv(OUT / "regression_results_all.csv", index=False, encoding="utf-8-sig")

    x = df[POLLUTANTS].dropna()
    xs = StandardScaler().fit_transform(x)
    pca = PCA()
    scores = pca.fit_transform(xs)
    explained = pd.DataFrame({
        "主成分": [f"PC{i+1}" for i in range(len(POLLUTANTS))],
        "特征值": pca.explained_variance_,
        "方差解释率": pca.explained_variance_ratio_,
        "累计解释率": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(OUT / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    loadings = pd.DataFrame(pca.components_.T, index=[CN[p] for p in POLLUTANTS], columns=[f"PC{i+1}" for i in range(len(POLLUTANTS))])
    loadings.to_csv(OUT / "pca_loadings.csv", encoding="utf-8-sig")

    df_pca = df.loc[x.index].copy()
    df_pca["PC1"] = scores[:, 0]
    df_pca["PC2"] = scores[:, 1]
    pca_formula = "y_base ~ PC1 + PC2 + " + " + ".join(CONTROLS) + " + C(province) + C(year)"
    pca_model = fit(pca_formula, df_pca)
    result_table(pca_model, ["PC1", "PC2"], "PCA + 双向固定效应", "y_base").round(6).to_csv(OUT / "pca_regression.csv", index=False, encoding="utf-8-sig")

    df["midwest"] = (df["region"] != "东部").astype(int)
    interactions = [f"{p}:midwest" for p in POLLUTANTS]
    hetero_formula = "y_base ~ " + " + ".join(POLLUTANTS + interactions + CONTROLS) + " + C(province) + C(year)"
    hetero_model = fit(hetero_formula, df)
    result_table(hetero_model, POLLUTANTS + interactions, "区域异质性交互项", "y_base").round(6).to_csv(OUT / "region_interaction_results.csv", index=False, encoding="utf-8-sig")

    report = """# 自动分析结果索引

本脚本已生成以下文件：

- `data_profile.md`：数据结构与适用场景说明
- `descriptive_statistics.csv`：描述性统计
- `missing_check.csv`：缺失值检查
- `summary_by_year.csv`：年度均值
- `summary_by_region.csv`：区域均值
- `correlation_matrix.csv`：相关矩阵
- `regression_results_all.csv`：OLS 与固定效应模型结果
- `pca_explained_variance.csv`：PCA 方差解释率
- `pca_loadings.csv`：PCA 载荷矩阵
- `pca_regression.csv`：主成分回归结果
- `region_interaction_results.csv`：区域异质性交互项结果
- `pollutant_trends.png`：污染物年度趋势图

建议先查看 `regression_results_all.csv`，再结合 `pca_regression.csv` 和 `region_interaction_results.csv` 撰写论文式分析。
"""
    (OUT / "analysis_index.md").write_text(report, encoding="utf-8")
    print("分析完成。结果已输出到 outputs 文件夹。")


if __name__ == "__main__":
    main()
