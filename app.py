# app.py
import re
from io import BytesIO
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Utilities
# -----------------------------
WEEK_ORDER_ZH = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEK_ORDER_ZH_MAP = {
    "Monday": "周一",
    "Tuesday": "周二",
    "Wednesday": "周三",
    "Thursday": "周四",
    "Friday": "周五",
    "Saturday": "周六",
    "Sunday": "周日",
}

DEFAULT_TUANGOU_DATE_CANDIDATES = ["下单时间", "订单时间", "创建时间", "支付时间"]
DEFAULT_TUANGOU_AMOUNT_CANDIDATES = ["订单实收", "实收金额", "实收", "用户实付", "付款金额"]

DEFAULT_XIAOHUI_DATE_CANDIDATES = ["核销时间", "券核销时间", "核销完成时间", "下单时间"]
DEFAULT_XIAOHUI_AMOUNT_CANDIDATES = ["订单实收", "核销金额", "券用户实付金额", "用户实付金额", "实收金额"]


def pick_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def clean_amount_series(s: pd.Series) -> pd.Series:
    """
    Robustly parse currency-like strings to numeric.
    Handles: ￥1,234.56  |  1,234  |  1 234  |  (123) -> -123  |  —/空 -> NaN
    """
    if s is None:
        return s

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    # Convert to string, normalize
    x = s.astype(str).str.strip()

    # Normalize common "missing" tokens
    x = x.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "—": np.nan,
            "-": np.nan,
            "--": np.nan,
        }
    )

    # Parentheses negative: (123.45) -> -123.45
    x = x.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    # Remove currency symbols and spaces and commas
    x = x.str.replace(r"[￥¥$,]", "", regex=True)
    x = x.str.replace(r"\s+", "", regex=True)

    # Keep only valid number pattern (optional leading -, digits, optional .digits)
    # If there are other characters, to_numeric will coerce anyway
    return pd.to_numeric(x, errors="coerce")


def coerce_datetime(series: pd.Series) -> pd.Series:
    # Try normal parsing; if many NaT, try with infer and errors
    dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    return dt


def build_daily_summary(
    tuangou_df: pd.DataFrame,
    xiaohui_df: pd.DataFrame,
    tuangou_date_col: str,
    tuangou_amount_col: str,
    xiaohui_date_col: str,
    xiaohui_amount_col: str,
    merge_how: str = "outer",
) -> pd.DataFrame:
    # ---团购---
    tg = tuangou_df.copy()
    tg_dt = coerce_datetime(tg[tuangou_date_col])
    tg["日期_dt"] = tg_dt
    tg["日期"] = tg["日期_dt"].dt.date
    tg_amt = clean_amount_series(tg[tuangou_amount_col])
    tg[tuangou_amount_col] = tg_amt

    tuangou_daily = (
        tg.groupby("日期", dropna=False)[tuangou_amount_col]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={tuangou_amount_col: "团购订单实收汇总"})
    )

    # ---核销---
    xh = xiaohui_df.copy()
    xh_dt = coerce_datetime(xh[xiaohui_date_col])
    xh["日期_dt"] = xh_dt
    xh["日期"] = xh["日期_dt"].dt.date
    xh_amt = clean_amount_series(xh[xiaohui_amount_col])
    xh[xiaohui_amount_col] = xh_amt

    xiaohui_daily = (
        xh.groupby("日期", dropna=False)[xiaohui_amount_col]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={xiaohui_amount_col: "核销订单实收汇总"})
    )

    # 合并
    merged = pd.merge(tuangou_daily, xiaohui_daily, on="日期", how=merge_how)
    merged[["团购订单实收汇总", "核销订单实收汇总"]] = merged[
        ["团购订单实收汇总", "核销订单实收汇总"]
    ].fillna(0)

    merged["总订单实收金额"] = merged["团购订单实收汇总"] + merged["核销订单实收汇总"]

    # safe ratio
    merged["核销占比"] = np.where(
        merged["总订单实收金额"] > 0,
        (merged["核销订单实收汇总"] / merged["总订单实收金额"] * 100).round(2),
        0.0,
    )

    merged["日期"] = pd.to_datetime(merged["日期"], errors="coerce")
    merged["星期几_en"] = merged["日期"].dt.day_name()
    merged["星期几"] = merged["星期几_en"].map(WEEK_ORDER_ZH_MAP).fillna(merged["星期几_en"])
    merged["月份"] = merged["日期"].dt.to_period("M").astype(str)

    # 排序
    merged = merged.sort_values("日期").reset_index(drop=True)

    return merged


def build_weekly_monthly(merged_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = merged_daily.copy()

    # Weekly
    # Use English day_name for ordering, then display Chinese
    df["weekday_en"] = df["日期"].dt.day_name()
    weekly = (
        df.groupby("weekday_en", dropna=False)
        .agg(
            团购订单实收汇总=("团购订单实收汇总", "sum"),
            核销订单实收汇总=("核销订单实收汇总", "sum"),
            总订单实收金额=("总订单实收金额", "sum"),
            天数=("日期", "nunique"),
        )
        .reset_index()
    )
    weekly["核销占比"] = np.where(
        weekly["总订单实收金额"] > 0,
        (weekly["核销订单实收汇总"] / weekly["总订单实收金额"] * 100).round(2),
        0.0,
    )
    weekly["星期几"] = weekly["weekday_en"].map(WEEK_ORDER_ZH_MAP).fillna(weekly["weekday_en"])

    # Order by Monday..Sunday
    cat = pd.Categorical(weekly["weekday_en"], categories=WEEK_ORDER_ZH, ordered=True)
    weekly = weekly.assign(_order=cat).sort_values("_order").drop(columns=["_order"])
    weekly = weekly[["星期几", "团购订单实收汇总", "核销订单实收汇总", "总订单实收金额", "核销占比", "天数"]]

    # Monthly
    monthly = (
        df.groupby("月份", dropna=False)
        .agg(
            团购订单实收汇总=("团购订单实收汇总", "sum"),
            核销订单实收汇总=("核销订单实收汇总", "sum"),
            总订单实收金额=("总订单实收金额", "sum"),
            天数=("日期", "nunique"),
        )
        .reset_index()
    )
    monthly["核销占比"] = np.where(
        monthly["总订单实收金额"] > 0,
        (monthly["核销订单实收汇总"] / monthly["总订单实收金额"] * 100).round(2),
        0.0,
    )
    monthly = monthly.sort_values("月份").reset_index(drop=True)
    monthly = monthly[["月份", "团购订单实收汇总", "核销订单实收汇总", "总订单实收金额", "核销占比", "天数"]]

    return weekly, monthly


def format_money_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def export_excel_bytes(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    tuangou_sample: pd.DataFrame,
    xiaohui_sample: pd.DataFrame,
) -> BytesIO:
    """
    Write to BytesIO so it can be downloaded in Streamlit (deployment-friendly).
    """
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        daily_export = daily.copy()
        daily_export["日期"] = daily_export["日期"].dt.strftime("%Y-%m-%d")
        daily_export.to_excel(writer, sheet_name="每日汇总", index=False)
        weekly.to_excel(writer, sheet_name="每周汇总", index=False)
        monthly.to_excel(writer, sheet_name="每月汇总", index=False)

        # sample sheets (limit to 1000 rows like your original)
        tuangou_sample.head(1000).to_excel(writer, sheet_name="团购数据样本", index=False)
        xiaohui_sample.head(1000).to_excel(writer, sheet_name="核销数据样本", index=False)

    bio.seek(0)
    return bio


def compute_quality_report(df: pd.DataFrame, date_col: str, amount_col: str) -> dict:
    dt = coerce_datetime(df[date_col])
    amt = clean_amount_series(df[amount_col])

    date_total = len(df)
    date_ok = dt.notna().sum()
    amt_ok = amt.notna().sum()

    return {
        "rows": int(date_total),
        "date_ok": int(date_ok),
        "date_bad": int(date_total - date_ok),
        "date_bad_pct": float((date_total - date_ok) / max(date_total, 1) * 100),
        "amt_ok": int(amt_ok),
        "amt_bad": int(date_total - amt_ok),
        "amt_bad_pct": float((date_total - amt_ok) / max(date_total, 1) * 100),
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="订单实收金额统计工具", layout="wide")
st.title("📊 订单实收金额统计工具（Streamlit 版）")

with st.expander("使用说明", expanded=True):
    st.markdown(
        """
- 上传 **团购成交明细** 与 **核销明细** 两个 Excel（.xlsx）
- 程序会自动识别“日期列/金额列”，并在页面展示；如识别不对可手动选择
- 处理完成后可直接下载汇总结果（不落盘，适合云端部署）
"""
    )

colA, colB = st.columns(2)
with colA:
    tuangou_file = st.file_uploader("上传：团购成交明细（.xlsx）", type=["xlsx"], key="tuangou")
with colB:
    xiaohui_file = st.file_uploader("上传：核销明细（.xlsx）", type=["xlsx"], key="xiaohui")

merge_how = st.selectbox(
    "合并口径（按日期）",
    options=["outer", "inner", "left", "right"],
    index=0,
    help="outer=全部日期；inner=两边都有的日期；left=以团购为准；right=以核销为准",
)

if tuangou_file and xiaohui_file:
    # Read previews
    try:
        tuangou_df = pd.read_excel(tuangou_file)
        xiaohui_df = pd.read_excel(xiaohui_file)
    except Exception as e:
        st.error(f"读取 Excel 失败：{e}")
        st.stop()

    st.subheader("1) 自动识别列（可手动调整）")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**团购成交明细**")
        tg_date_auto = pick_first_existing(list(tuangou_df.columns), DEFAULT_TUANGOU_DATE_CANDIDATES)
        tg_amt_auto = pick_first_existing(list(tuangou_df.columns), DEFAULT_TUANGOU_AMOUNT_CANDIDATES)

        tg_date_col = st.selectbox(
            "团购：日期列",
            options=list(tuangou_df.columns),
            index=list(tuangou_df.columns).index(tg_date_auto) if tg_date_auto in tuangou_df.columns else 0,
            help="通常是：下单时间",
        )
        tg_amt_col = st.selectbox(
            "团购：金额列",
            options=list(tuangou_df.columns),
            index=list(tuangou_df.columns).index(tg_amt_auto) if tg_amt_auto in tuangou_df.columns else 0,
            help="通常是：订单实收",
        )

        st.caption(f"自动识别：日期={tg_date_auto or '未识别'}；金额={tg_amt_auto or '未识别'}")

    with c2:
        st.markdown("**核销明细**")
        xh_date_auto = pick_first_existing(list(xiaohui_df.columns), DEFAULT_XIAOHUI_DATE_CANDIDATES)
        xh_amt_auto = pick_first_existing(list(xiaohui_df.columns), DEFAULT_XIAOHUI_AMOUNT_CANDIDATES)

        xh_date_col = st.selectbox(
            "核销：日期列",
            options=list(xiaohui_df.columns),
            index=list(xiaohui_df.columns).index(xh_date_auto) if xh_date_auto in xiaohui_df.columns else 0,
            help="通常是：核销时间 / 券核销时间",
        )
        xh_amt_col = st.selectbox(
            "核销：金额列",
            options=list(xiaohui_df.columns),
            index=list(xiaohui_df.columns).index(xh_amt_auto) if xh_amt_auto in xiaohui_df.columns else 0,
            help="通常是：订单实收 / 核销金额",
        )

        st.caption(f"自动识别：日期={xh_date_auto or '未识别'}；金额={xh_amt_auto or '未识别'}")

    # Data quality checks
    st.subheader("2) 数据质量检查（解析失败会提示）")
    q1, q2 = st.columns(2)
    tg_q = compute_quality_report(tuangou_df, tg_date_col, tg_amt_col)
    xh_q = compute_quality_report(xiaohui_df, xh_date_col, xh_amt_col)

    with q1:
        st.markdown("**团购：日期/金额可解析情况**")
        st.write(
            {
                "总行数": tg_q["rows"],
                "日期解析失败(行)": tg_q["date_bad"],
                "日期解析失败(%)": round(tg_q["date_bad_pct"], 2),
                "金额解析失败(行)": tg_q["amt_bad"],
                "金额解析失败(%)": round(tg_q["amt_bad_pct"], 2),
            }
        )
        if tg_q["date_bad_pct"] > 5 or tg_q["amt_bad_pct"] > 5:
            st.warning("团购文件：解析失败比例偏高，可能列选错或格式不规范。建议检查列选择与内容。")

    with q2:
        st.markdown("**核销：日期/金额可解析情况**")
        st.write(
            {
                "总行数": xh_q["rows"],
                "日期解析失败(行)": xh_q["date_bad"],
                "日期解析失败(%)": round(xh_q["date_bad_pct"], 2),
                "金额解析失败(行)": xh_q["amt_bad"],
                "金额解析失败(%)": round(xh_q["amt_bad_pct"], 2),
            }
        )
        if xh_q["date_bad_pct"] > 5 or xh_q["amt_bad_pct"] > 5:
            st.warning("核销文件：解析失败比例偏高，可能列选错或格式不规范。建议检查列选择与内容。")

    # Optional date filter (nice to have)
    st.subheader("3) 生成汇总 + 下载")
    st.caption("点击后会按你选择的列与合并口径生成每日/每周/每月汇总，并提供 Excel 下载。")

    if st.button("🚀 开始处理并生成报表", type="primary"):
        with st.spinner("处理中..."):
            # build summaries
            daily = build_daily_summary(
                tuangou_df=tuangou_df,
                xiaohui_df=xiaohui_df,
                tuangou_date_col=tg_date_col,
                tuangou_amount_col=tg_amt_col,
                xiaohui_date_col=xh_date_col,
                xiaohui_amount_col=xh_amt_col,
                merge_how=merge_how,
            )
            weekly, monthly = build_weekly_monthly(daily)

            # totals
            total_tuangou = float(daily["团购订单实收汇总"].sum())
            total_xiaohui = float(daily["核销订单实收汇总"].sum())
            total_amount = float(daily["总订单实收金额"].sum())
            ratio = (total_xiaohui / total_amount * 100) if total_amount > 0 else 0.0

            # export
            excel_bytes = export_excel_bytes(
                daily=daily,
                weekly=weekly,
                monthly=monthly,
                tuangou_sample=tuangou_df,
                xiaohui_sample=xiaohui_df,
            )

        st.success("处理完成！")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("团购订单实收总额", f"{total_tuangou:,.2f}")
        m2.metric("核销订单实收总额", f"{total_xiaohui:,.2f}")
        m3.metric("总订单实收金额", f"{total_amount:,.2f}")
        m4.metric("核销占比(%)", f"{ratio:,.2f}")

        st.markdown("### 📅 每日汇总")
        st.dataframe(daily, use_container_width=True)

        st.markdown("### 📆 每周汇总")
        st.dataframe(weekly, use_container_width=True)

        st.markdown("### 🗓 每月汇总")
        st.dataframe(monthly, use_container_width=True)

        filename = f"订单实收汇总_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.download_button(
            label="⬇️ 下载 Excel 汇总结果",
            data=excel_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.info("请先上传两个 .xlsx 文件（团购成交明细 + 核销明细）。")
