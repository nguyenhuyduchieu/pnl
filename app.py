import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# ===================== CONFIG =====================
BASE = Path(__file__).resolve().parent
MY_CSV_PATH   = BASE / "data" / "my_invest.csv"
VN30_CSV_PATH = BASE / "data" / "VN30.csv"

# Ưu tiên dùng CLOSE để tính return % cho VN30 (khớp con số ~14% bạn mong muốn)
VN30_USE_CLOSE_FOR_KPI = True

st.set_page_config(
    page_title="HieuDwc Investment",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== STYLES =====================
st.markdown("""
<style>
/* Ẩn menu góc phải (bao gồm View source / Fork) */
div[data-testid="stToolbar"] { display: none; }

/* Ẩn các badge/decoration của Streamlit Cloud (nếu có) */
div[data-testid="stDecoration"] { display: none; }

/* Ẩn menu cũ + header/footer chung */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* Ẩn nút Deploy trên Cloud */
button[kind="header"] { display: none; }

/* Một số class badge cũ trên vài bản build */
a.viewerBadge_container__r5tak,
a.viewerBadge_link__1S137,
a[kind="viewerBadge"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="big-title">HieuDwc Investment</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtle">Hiệu suất danh mục</h2>', unsafe_allow_html=True)

# ===================== HELPERS =====================
def _read_csv_any(path: str) -> pd.DataFrame:
    """Đọc CSV, tự đoán delimiter, giữ nguyên cột."""
    return pd.read_csv(path, sep=None, engine="python")

@st.cache_data
def load_my_total_percent(path: str) -> pd.Series:
    """
    Trả về Series my_total_percent (index Datetime).
    Yêu cầu: có cột 'Datetime' và 'total_percent' hoặc index là Datetime.
    """
    df = _read_csv_any(path)
    df.columns = df.columns.str.strip()

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
    else:
        # Trường hợp file có index là thời gian
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.rename_axis("Datetime")

    s = pd.to_numeric(df["total_percent"], errors="coerce").sort_index()
    return s

@st.cache_data
def load_vn30_data(path: str) -> pd.DataFrame:
    """
    Trả về DataFrame có thể gồm:
      - vn30_total_percent: % tích lũy (nếu có hoặc tính từ Close)
      - vn30_close: giá đóng cửa (nếu có)
      - vn30_percent: % ngày (nếu có)
    Index là Date (datetime).
    """
    df = _read_csv_any(path)
    df.columns = df.columns.str.strip()

    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame(index=df.index)

    if "Close" in df.columns:
        out["vn30_close"] = df["Close"].astype(float)

    if "total_percent" in df.columns:
        out["vn30_total_percent"] = df["total_percent"].astype(float)
    else:
        if "Close" not in df.columns:
            raise ValueError("VN30.csv cần có 'total_percent' hoặc 'Close'.")
        ret = df["Close"].pct_change().fillna(0) * 100.0  # % ngày
        out["vn30_total_percent"] = ret.cumsum()

    if "percent" in df.columns:
        out["vn30_percent"] = df["percent"].astype(float)

    return out

def rebase_to_zero(s: pd.Series) -> pd.Series:
    """Đưa series tích lũy (%) về 0 tại điểm đầu tiên có dữ liệu."""
    s = s.dropna()
    if s.empty:
        return s
    return s - s.iloc[0]

def asof_value(s: pd.Series, ts: pd.Timestamp):
    """Giá trị gần nhất <= ts (an toàn cho ngày nghỉ)."""
    s = s.dropna().sort_index()
    s = s.loc[s.index <= ts]
    if s.empty:
        return np.nan
    return s.iloc[-1]

def period_return_pct_from_total(total_pct: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """
    % lợi nhuận compounding từ series total_%:
      r = ((1 + end/100) / (1 + start/100) - 1)*100
    """
    e0 = 1.0 + asof_value(total_pct, start_date)/100.0
    e1 = 1.0 + asof_value(total_pct, end_date)/100.0
    if np.isnan(e0) or np.isnan(e1):
        return np.nan
    return (e1/e0 - 1.0)*100.0

def period_return_pct_from_close(close: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """
    % lợi nhuận từ giá Close:
      r = (Close_end / Close_start - 1)*100
    """
    c0 = asof_value(close, start_date)
    c1 = asof_value(close, end_date)
    if np.isnan(c0) or np.isnan(c1) or c0 == 0:
        return np.nan
    return (c1/c0 - 1.0) * 100.0

def last_delta_from_series(s: pd.Series, start_ts, end_ts):
    """Delta ngày gần nhất dựa trên total% (hiệu số, không compounding)."""
    s_win = s.loc[(s.index >= start_ts) & (s.index <= end_ts)].dropna()
    if len(s_win) < 2:
        return 0.0
    return s_win.iloc[-1] - s_win.iloc[-2]

def pnl_from_value_difference(total_pct: pd.Series,
                              start_date: pd.Timestamp,
                              end_date: pd.Timestamp,
                              capital: float):
    """
    Tính PnL theo 'giá trị tất toán - giá trị đầu kỳ' với mốc reset tại ngày bắt đầu.
    - Lấy delta% = total_pct(end) - total_pct(start)
    - value_start = capital
    - value_end   = capital * (1 + delta/100)
    -> PnL = capital * delta/100
    -> % so với vốn đầu kỳ = delta
    """
    p0 = asof_value(total_pct, pd.Timestamp(start_date))
    p1 = asof_value(total_pct, pd.Timestamp(end_date))

    if np.isnan(p0) or np.isnan(p1):
        return np.nan, np.nan, np.nan, np.nan

    delta = p1 - p0
    v0 = capital
    v1 = capital * (1.0 + delta/100.0)
    pnl = v1 - v0
    pct_vs_start = delta
    return pnl, v0, v1, pct_vs_start


# ===================== LOAD =====================
try:
    my_total = load_my_total_percent(MY_CSV_PATH)  # total % gốc của bạn
    vn30_df  = load_vn30_data(VN30_CSV_PATH)       # chứa vn30_total_percent và có thể có vn30_close
except Exception as e:
    st.error(f"Không đọc được file: {e}")
    st.stop()

if my_total.dropna().empty:
    st.error("Dữ liệu my_invest.csv không có total_percent hợp lệ.")
    st.stop()

vn30_total_full = vn30_df["vn30_total_percent"]
vn30_close_full = vn30_df["vn30_close"] if "vn30_close" in vn30_df.columns else None

# Cắt VN30 từ NGÀY BẮT ĐẦU của dữ liệu bạn
start_my = my_total.dropna().index.min()
vn30_cut_total = vn30_total_full.loc[vn30_total_full.index >= start_my]

# Rebase cả hai về 0 (để so sánh)
my_total_cut   = my_total.loc[my_total.index >= start_my]
my_total_base0 = rebase_to_zero(my_total_cut)
vn30_base0     = rebase_to_zero(vn30_cut_total)

# ----------------- HỢP (UNION) ngày, KHÔNG fill ở đây -----------------
base_union = (
    pd.concat(
        [
            my_total_base0.rename("my_total_percent"),
            vn30_base0.rename("vn30_total_percent"),
        ],
        axis=1,
        join="outer",
    ).sort_index()
)

# Dải ngày hợp lệ cho slider: giao (intersection) để đảm bảo cả 2 chuỗi đều có dữ liệu gốc
inter = base_union.dropna(how="any").index
if inter.empty:
    st.error("Không có ngày trùng nhau giữa 2 chuỗi (sau khi cắt theo ngày bắt đầu).")
    st.stop()

min_date = inter.min()
max_date = inter.max()

date_range = st.slider(
    "Khoảng thời gian hiển thị",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM-DD",
)

idx_start = pd.to_datetime(date_range[0])
idx_end   = pd.to_datetime(date_range[1])

# ----------------- DỮ LIỆU ĐỂ VẼ: LẤY UNION + FFILL/ BFILL TRONG CỬA SỔ -----------------
view = base_union.loc[(base_union.index >= idx_start) & (base_union.index <= idx_end)].copy()
# Fill để đường line không bị đứt đoạn do NaN
view = view.sort_index().ffill().bfill()
view.index.name = "Date"

# ===================== KPIs (COMPOUNDING) =====================
# Tôi: từ total % gốc (không fill)
my_kpi = period_return_pct_from_total(my_total, idx_start, idx_end)

# VN30: ưu tiên CLOSE nếu có; fallback sang total_percent gốc
if VN30_USE_CLOSE_FOR_KPI and (vn30_close_full is not None):
    vn_kpi = period_return_pct_from_close(vn30_close_full, idx_start, idx_end)
else:
    vn_kpi = period_return_pct_from_total(vn30_total_full, idx_start, idx_end)

# Delta ngày gần nhất (từ total% gốc)
d_my = last_delta_from_series(my_total, idx_start, idx_end)
d_vn = last_delta_from_series(vn30_total_full, idx_start, idx_end)

# Hiển thị 2 KPI
c1, c2 = st.columns(2)
c1.metric("Tôi (Return %)",  f"{my_kpi:,.2f} %",  f"{d_my:+,.2f} %")
c2.metric("VN30 (Return %)", f"{vn_kpi:,.2f} %", f"{d_vn:+,.2f} %")

# ===================== CHART =====================
if not view.empty:
    plot_df = view.reset_index().melt(id_vars="Date", var_name="Series", value_name="Value")
    plot_df["Series"] = plot_df["Series"].map({
        "my_total_percent": "Kết quả đầu tư",
        "vn30_total_percent": "VN30"
    })

    line = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Total %"),
            color=alt.Color("Series:N", legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Series:N", title=""),
                alt.Tooltip("Value:Q", title="Total %", format=".2f"),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(line, use_container_width=True)
else:
    st.info("Không có dữ liệu trong khoảng thời gian đã chọn.")

# ===================== TÍNH TOÁN LỢI NHUẬN (CHỈ DANH MỤC CỦA BẠN) =====================
st.subheader("💰 Tính lợi nhuận")

max_raw_date = my_total.dropna().index.max()

col_left, col_mid, col_right = st.columns([1,1,1])
with col_left:
    start_date = st.date_input(
        "Ngày bắt đầu",
        value=my_total.dropna().index.min().date(),
        min_value=my_total.dropna().index.min().date(),
        max_value=max_raw_date.date()
    )
with col_mid:
    end_date = st.date_input(
        "Ngày tất toán",
        value=max_raw_date.date(),
        min_value=my_total.dropna().index.min().date(),
        max_value=max_raw_date.date()
    )
with col_right:
    capital = st.number_input("Số tiền đầu tư (VND)", value=100_000_000, min_value=0, step=1_000_000)

if pd.Timestamp(end_date) < pd.Timestamp(start_date):
    st.warning("Ngày tất toán phải >= ngày bắt đầu.")
else:
    pnl_my, v0_my, v1_my, pct_vs_start = pnl_from_value_difference(
        my_total, pd.Timestamp(start_date), pd.Timestamp(end_date), capital
    )

    st.markdown("**Danh mục của tôi**")
    if np.isfinite(pnl_my):
        colA, colB = st.columns(2)
        colA.metric("Lãi/Lỗ (VND)", f"{pnl_my:,.0f}")
        colB.metric("Lợi nhuận (%) so với vốn đầu kỳ", f"{pct_vs_start:,.2f} %")

        # Thông tin giá trị đầu/ cuối kỳ
        st.caption(f"Giá trị đầu kỳ: {v0_my:,.0f} VND  •  Giá trị tất toán: {v1_my:,.0f} VND")
    else:
        st.info("Không đủ dữ liệu trong khoảng đã chọn.")
