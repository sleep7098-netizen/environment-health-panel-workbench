from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="环境健康统计建模工作台",
    page_icon="📊",
    layout="wide",
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "model_sample.csv"

COL_MAP = {
    "省份编码": "province_id",
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
    "常住人口_万人": "pop_10k",
    "人均GDP_元": "gdp_pc",
    "地区生产总值_亿元": "gdp",
    "城镇化率": "urban_rate",
    "第二产业占比": "industry2_share",
    "医疗卫生机构数_个": "medical_orgs",
    "医院数_个": "hospitals",
    "城镇人口_万人": "urban_pop_10k",
    "第二产业增加值_亿元": "industry2_value",
    "污染覆盖月份数": "pollution_months",
    "污染数据是否完整年": "pollution_full_year",
}

Y_OPTIONS = {
    "基准健康负担代理指标": "y_base",
    "稳健性指标：住院率": "y_hosp",
    "稳健性指标：两周就诊率": "y_2week",
}

POLLUTANT_OPTIONS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "O3": "o3",
    "CO": "co",
}

CONTROL_OPTIONS = {
    "ln(常住人口)": "ln_pop",
    "ln(人均GDP)": "ln_gdp",
    "城镇化率": "urban_rate",
    "第二产业占比": "industry2_share",
    "医疗卫生机构数": "medical_orgs",
    "医院数": "hospitals",
}

CN_NAME = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "so2": "SO2",
    "no2": "NO2",
    "o3": "O3",
    "co": "CO",
    "ln_pop": "ln(常住人口)",
    "ln_gdp": "ln(人均GDP)",
    "urban_rate": "城镇化率",
    "industry2_share": "第二产业占比",
    "medical_orgs": "医疗卫生机构数",
    "hospitals": "医院数",
    "y_base": "基准健康负担代理指标",
    "y_hosp": "住院率",
    "y_2week": "两周就诊率",
}


def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, encoding="utf-8-sig")
    return pd.read_csv(DEFAULT_DATA, encoding="utf-8-sig")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.rename(columns=COL_MAP).copy()
    if "model_sample" in d.columns:
        d = d[d["model_sample"] == 1].copy()
    return d


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


def extract_results(model, variables):
    rows = []
    for var in variables:
        rows.append(
            {
                "变量": CN_NAME.get(var, var),
                "模型变量名": var,
                "系数": model.params.get(var, np.nan),
                "聚类稳健标准误": model.bse.get(var, np.nan),
                "t值": model.tvalues.get(var, np.nan),
                "p值": model.pvalues.get(var, np.nan),
                "显著性": stars(model.pvalues.get(var, np.nan)),
            }
        )
    out = pd.DataFrame(rows)
    numeric_cols = ["系数", "聚类稳健标准误", "t值", "p值"]
    out[numeric_cols] = out[numeric_cols].round(4)
    return out


def run_model(d, y, pollutants, controls, model_type):
    rhs = pollutants + controls
    if not rhs:
        raise ValueError("至少需要选择一个解释变量。")

    if model_type == "混合 OLS + 年份效应":
        formula = f"{y} ~ " + " + ".join(rhs) + " + C(year)"
        shown_vars = pollutants
    elif model_type == "双向固定效应":
        formula = f"{y} ~ " + " + ".join(rhs) + " + C(province) + C(year)"
        shown_vars = pollutants
    else:
        d = d.copy()
        d["midwest"] = (d["region"] != "东部").astype(int)
        interactions = [f"{p}:midwest" for p in pollutants]
        formula = f"{y} ~ " + " + ".join(pollutants + interactions + controls) + " + C(province) + C(year)"
        shown_vars = pollutants + interactions

    model = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["province"]})
    return model, extract_results(model, shown_vars), formula


def csv_download(df):
    return df.to_csv(index=False, encoding="utf-8-sig")


def build_interpretation_text(result_df, y_label, model_type):
    lines = [
        f"当前模型以“{y_label}”为因变量，采用“{model_type}”设定。",
        "从显著性看，应重点关注 p 值低于 0.10、0.05 或 0.01 的变量。",
        "由于本数据为省级年度面板数据，且健康变量为代理指标，因此结果更适合解释为统计关联，而不宜直接表述为严格因果效应。",
    ]
    sig = result_df[result_df["显著性"] != ""]
    if len(sig) > 0:
        parts = []
        for _, row in sig.iterrows():
            direction = "正向" if row["系数"] > 0 else "负向"
            parts.append(f"{row['变量']}呈{direction}关联，系数为 {row['系数']}，显著性为 {row['显著性']}")
        lines.append("模型中具有统计显著性的变量包括：" + "；".join(parts) + "。")
    else:
        lines.append("当前选择的核心变量未达到常规显著性水平，需要结合稳健性检验和变量相关性进一步判断。")
    lines.append("在撰写论文时，建议同时讨论多污染物共线性、样本尺度较粗、遗漏变量和健康代理指标构造等限制。")
    return "\n\n".join(lines)


st.title("📊 环境健康统计建模工作台")
st.caption("基于省级面板数据的污染暴露、健康负担、固定效应、稳健性检验、主成分分析与区域异质性分析")

with st.sidebar:
    st.header("数据与模型设置")
    uploaded = st.file_uploader("上传 CSV 数据，或使用默认建模样本", type=["csv"])
    df_raw = load_data(uploaded)
    d = prepare(df_raw)

    y_label = st.selectbox("选择因变量", list(Y_OPTIONS.keys()))
    y = Y_OPTIONS[y_label]

    pollutant_labels = st.multiselect(
        "选择污染物变量",
        list(POLLUTANT_OPTIONS.keys()),
        default=["PM2.5", "PM10", "SO2", "NO2", "O3"],
    )
    pollutants = [POLLUTANT_OPTIONS[x] for x in pollutant_labels]

    control_labels = st.multiselect(
        "选择控制变量",
        list(CONTROL_OPTIONS.keys()),
        default=["ln(常住人口)", "ln(人均GDP)", "城镇化率", "第二产业占比", "医疗卫生机构数"],
    )
    controls = [CONTROL_OPTIONS[x] for x in control_labels]

    model_type = st.selectbox("选择模型", ["混合 OLS + 年份效应", "双向固定效应", "区域异质性交互项"])

    st.divider()
    st.caption("建议：论文主模型优先使用双向固定效应；异质性分析选择交互项模型。")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["数据概览", "趋势与相关性", "模型实验室", "PCA 综合污染因子", "异质性分析", "报告辅助"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("观测数", df_raw.shape[0])
    c2.metric("字段数", df_raw.shape[1])
    c3.metric("省份数", df_raw["省份"].nunique())
    c4.metric("年份范围", f"{df_raw['年份'].min()}—{df_raw['年份'].max()}")

    st.subheader("区域分布")
    region_count = df_raw["所属区域"].value_counts().reset_index()
    region_count.columns = ["所属区域", "观测数"]
    st.dataframe(region_count, width="stretch")

    st.subheader("数据预览")
    st.dataframe(df_raw.head(25), width="stretch")

    st.subheader("缺失值检查")
    missing = df_raw.isna().sum().reset_index()
    missing.columns = ["字段", "缺失数量"]
    st.dataframe(missing, width="stretch")

    st.download_button(
        "下载缺失值检查表",
        csv_download(missing),
        file_name="missing_check.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("年度均值趋势")
    trend_cols = ["基准因变量"] + pollutant_labels
    trend_cols = [x for x in trend_cols if x in df_raw.columns]
    trend_df = df_raw.groupby("年份")[trend_cols].mean(numeric_only=True).reset_index()
    fig = px.line(trend_df, x="年份", y=trend_cols, markers=True, title="年度均值趋势")
    st.plotly_chart(fig, width="stretch")

    st.subheader("区域均值对比")
    region_cols = ["基准因变量"] + pollutant_labels
    region_cols = [x for x in region_cols if x in df_raw.columns]
    region_df = df_raw.groupby("所属区域")[region_cols].mean(numeric_only=True).reset_index()
    fig2 = px.bar(region_df, x="所属区域", y=region_cols, barmode="group", title="不同区域均值对比")
    st.plotly_chart(fig2, width="stretch")

    st.subheader("相关矩阵")
    corr_labels = ["基准因变量", "稳健性因变量_住院率", "稳健性因变量_两周就诊率"] + pollutant_labels + control_labels
    corr_labels = [x for x in corr_labels if x in df_raw.columns]
    corr = df_raw[corr_labels].corr(numeric_only=True)
    fig3 = px.imshow(corr, text_auto=".2f", aspect="auto", title="变量相关矩阵")
    st.plotly_chart(fig3, width="stretch")
    st.download_button("下载相关矩阵", corr.to_csv(encoding="utf-8-sig"), file_name="correlation_matrix.csv")

with tab3:
    st.subheader("模型实验室")
    if len(pollutants) == 0:
        st.warning("请至少选择一个污染物变量。")
    else:
        try:
            model, result, formula = run_model(d, y, pollutants, controls, model_type)
            st.code(formula, language="text")
            st.dataframe(result, width="stretch")
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{model.rsquared:.4f}")
            c2.metric("调整后 R²", f"{model.rsquared_adj:.4f}")
            c3.metric("样本量", int(model.nobs))
            st.download_button("下载当前模型结果", csv_download(result), file_name="model_result.csv", mime="text/csv")

            st.subheader("自动解释草稿")
            st.info(build_interpretation_text(result, y_label, model_type))
        except Exception as exc:
            st.error(f"模型运行失败：{exc}")

with tab4:
    st.subheader("PCA 综合污染因子")
    selected_pollutants = [POLLUTANT_OPTIONS[x] for x in pollutant_labels]
    if len(selected_pollutants) < 2:
        st.warning("PCA 至少需要两个污染物变量。")
    else:
        x = d[selected_pollutants].dropna()
        xs = StandardScaler().fit_transform(x)
        pca = PCA()
        scores = pca.fit_transform(xs)
        explained = pd.DataFrame(
            {
                "主成分": [f"PC{i+1}" for i in range(len(selected_pollutants))],
                "特征值": pca.explained_variance_.round(4),
                "方差解释率": pca.explained_variance_ratio_.round(4),
                "累计解释率": np.cumsum(pca.explained_variance_ratio_).round(4),
            }
        )
        loadings = pd.DataFrame(
            pca.components_.T,
            index=[CN_NAME.get(c, c) for c in selected_pollutants],
            columns=[f"PC{i+1}" for i in range(len(selected_pollutants))],
        ).round(4)

        st.write("方差解释结果")
        st.dataframe(explained, width="stretch")
        st.write("主成分载荷矩阵")
        st.dataframe(loadings, width="stretch")

        d_pca = d.loc[x.index].copy()
        d_pca["PC1"] = scores[:, 0]
        if scores.shape[1] > 1:
            d_pca["PC2"] = scores[:, 1]
            pca_formula = f"{y} ~ PC1 + PC2 + " + " + ".join(controls) + " + C(province) + C(year)"
            vars_show = ["PC1", "PC2"]
        else:
            pca_formula = f"{y} ~ PC1 + " + " + ".join(controls) + " + C(province) + C(year)"
            vars_show = ["PC1"]
        pca_model = smf.ols(pca_formula, data=d_pca).fit(cov_type="cluster", cov_kwds={"groups": d_pca["province"]})
        pca_result = extract_results(pca_model, vars_show)
        st.write("主成分固定效应回归")
        st.code(pca_formula, language="text")
        st.dataframe(pca_result, width="stretch")

with tab5:
    st.subheader("区域异质性分析")
    if len(pollutants) == 0:
        st.warning("请至少选择一个污染物变量。")
    else:
        try:
            model_h, result_h, formula_h = run_model(d, y, pollutants, controls, "区域异质性交互项")
            st.code(formula_h, language="text")
            st.dataframe(result_h, width="stretch")
            fig_h = px.bar(result_h, x="变量", y="系数", error_y="聚类稳健标准误", title="核心变量与区域交互项估计结果")
            st.plotly_chart(fig_h, width="stretch")
            st.caption("交互项表示非东部地区相对于东部地区的差异效应。")
        except Exception as exc:
            st.error(f"异质性模型运行失败：{exc}")

with tab6:
    st.subheader("报告辅助")
    st.markdown(
        "这个模块用于把统计结果转化为论文写作提纲。它不调用任何外部服务，只生成可复制的研究报告模板。"
    )
    template = f"""请根据下列信息撰写一段本科论文或研究报告中的“实证结果分析”文字：

研究主题：空气污染与呼吸系统健康负担的统计关联
数据结构：中国省级年度面板数据
因变量：{y_label}
模型类型：{model_type}
核心污染物：{', '.join(pollutant_labels)}
控制变量：{', '.join(control_labels)}

写作要求：
1. 先说明模型目的；
2. 再解释核心变量的方向、显著性和不确定性；
3. 强调结果属于统计关联，不直接等同于因果效应；
4. 结合多污染物共线性、省级年度数据尺度和健康代理指标限制进行谨慎讨论；
5. 最后给出可以放入论文的小结。
"""
    st.text_area("研究报告写作模板", template, height=320)

