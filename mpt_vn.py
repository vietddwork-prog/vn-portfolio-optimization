"""
Tối ưu hóa danh mục cổ phiếu Việt Nam theo Modern Portfolio Theory
So sánh: Max Sharpe, Min Volatility, Risk Parity (ERC), HRP, Equal Weight
Kèm VN-Index làm mốc và báo cáo tương tác output/report.html

Cài đặt:
    pip install -U --extra-index-url https://vnstocks.com/api/packages vnstock
    pip install pyportfolioopt matplotlib scipy pandas plotly

Chạy:
    python mpt_vn.py            # dữ liệu thật từ vnstock
    python mpt_vn.py --demo     # dữ liệu giả lập (không cần mạng) để thử code
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pypfopt import expected_returns, risk_models, EfficientFrontier, HRPOpt, plotting
from pypfopt.discrete_allocation import DiscreteAllocation

# Vá tương thích: SciPy bản mới bỏ hằng _LINKAGE_METHODS mà HRPOpt của PyPortfolioOpt cần
import scipy.cluster.hierarchy as _sch
if not hasattr(_sch, "_LINKAGE_METHODS"):
    _sch._LINKAGE_METHODS = {"single", "complete", "average", "weighted", "centroid", "median", "ward"}

# ============ BƯỚC 0: CẤU HÌNH ============
TICKERS = ["FPT", "VCB", "HPG", "MWG", "VNM", "GAS", "MBB", "REE"]
START, END = "2019-01-01", "2026-10-31"
RISK_FREE = 0.05          # lãi suất phi rủi ro/năm — hãy thay bằng lãi tiết kiệm 12 tháng hiện tại
MAX_WEIGHT = 0.30         # tối đa 30% vào một mã để tránh dồn hết vào 1-2 mã
CAPITAL = 200_000_000     # vốn (VND) để quy ra số cổ phiếu
LOT = 100                 # lô chẵn HOSE
TRADING_DAYS = 252
OUT = "output"
os.makedirs(OUT, exist_ok=True)


# ============ BƯỚC 1: LẤY DỮ LIỆU ============
def load_prices_vnstock(tickers, start, end):
    from vnstock import Quote
    frames = {}
    for t in tickers:
        try:
            df = Quote(symbol=t, source="vci").history(start=start, end=end, interval="1D")
            s = df.set_index(pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize())["close"]
            frames[t] = s
            print(f"  {t}: {len(s)} phiên")
        except Exception as e:
            print(f"  {t}: LỖI {e}")
    return pd.DataFrame(frames)


def load_prices_demo(tickers, start, end, seed=42):
    """Giả lập giá có tương quan (2 nhân tố: thị trường + ngân hàng) để thử code."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    market = rng.normal(0.0004, 0.011, n)
    bank = rng.normal(0, 0.007, n)
    data = {}
    for i, t in enumerate(tickers):
        beta = rng.uniform(0.6, 1.4)
        bank_load = 1.0 if t in ("VCB", "MBB") else 0.0
        r = rng.uniform(-0.0001, 0.0005) + beta * market + bank_load * bank + rng.normal(0, 0.012, n)
        data[t] = 30 * np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=idx)


# ============ BƯỚC 1b: CHỈ SỐ THAM CHIẾU (VN-Index) ============
BENCHMARK = "VNINDEX"


def load_benchmark_vnstock(start, end, symbol=BENCHMARK):
    """Thử API thế hệ 5 trước, nếu không được thì dùng lớp Quote kiểu cũ."""
    df = None
    try:
        from vnstock import Market
        df = Market().index(symbol).ohlcv(start=start, end=end)
    except Exception:
        try:
            from vnstock import Quote
            df = Quote(symbol=symbol, source="vci").history(start=start, end=end, interval="1D")
        except Exception as e:
            print(f"  Không lấy được {symbol}: {e}")
            return None
    if "time" not in df.columns:
        df = df.reset_index()
    tcol = "time" if "time" in df.columns else df.columns[0]
    s = df.set_index(pd.to_datetime(df[tcol]))["close"]
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    print(f"  {symbol}: {len(s)} phiên")
    return s.sort_index()


def load_benchmark_demo(prices):
    """Giả lập VN-Index = trung bình giá các mã giả lập."""
    return (prices / prices.iloc[0]).mean(axis=1) * 1000


# ============ BƯỚC 2: LÀM SẠCH ============
def clean(prices, min_coverage=0.9):
    prices = prices.sort_index()
    prices = prices.loc[:, prices.notna().mean() >= min_coverage]   # bỏ mã thiếu dữ liệu
    prices = prices.ffill().dropna()                                 # điền ngày nghỉ giao dịch
    rets = prices.pct_change().dropna()
    # Cảnh báo bước nhảy bất thường (>|15%|) — thường do chia tách/cổ tức cổ phiếu chưa điều chỉnh
    jumps = (rets.abs() > 0.15).sum()
    if jumps.any():
        print("  ⚠ Biến động >15%/phiên (kiểm tra giá điều chỉnh):", jumps[jumps > 0].to_dict())
    return prices


# ============ BƯỚC 3-6: CÁC CHIẾN LƯỢC ============
def w_max_sharpe(prices):
    mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    S = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
    try:
        ef.max_sharpe(risk_free_rate=RISK_FREE)
    except Exception:          # mọi mã đều có lợi nhuận kỳ vọng < RISK_FREE
        ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
        ef.min_volatility()
    return pd.Series(ef.clean_weights())


def w_min_vol(prices):
    S = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    ef = EfficientFrontier(None, S, weight_bounds=(0, MAX_WEIGHT))
    ef.min_volatility()
    return pd.Series(ef.clean_weights())


def w_risk_parity(prices):
    """Equal Risk Contribution: mỗi mã đóng góp rủi ro bằng nhau."""
    S = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf().values
    n = S.shape[0]

    def obj(w):
        port_var = w @ S @ w
        rc = w * (S @ w) / port_var
        return ((rc - 1 / n) ** 2).sum()

    res = minimize(obj, np.ones(n) / n, bounds=[(0, 1)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return pd.Series(res.x, index=prices.columns).round(5)


def w_hrp(prices):
    rets = prices.pct_change().dropna()
    return pd.Series(HRPOpt(rets).optimize())


def w_equal(prices):
    return pd.Series(1 / prices.shape[1], index=prices.columns)


STRATEGIES = {
    "Max Sharpe": w_max_sharpe,
    "Min Vol": w_min_vol,
    "Risk Parity": w_risk_parity,
    "HRP": w_hrp,
    "Equal Weight": w_equal,
}


# ============ BƯỚC 7: VẼ EFFICIENT FRONTIER ============
def plot_frontier(prices, weights_dict):
    mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    S = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    fig, ax = plt.subplots(figsize=(9, 6))
    ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # 5.000 danh mục ngẫu nhiên làm nền
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(len(mu)), 5000)
    r, v = w @ mu.values, np.sqrt(np.einsum("ij,jk,ik->i", w, S.values, w))
    sc = ax.scatter(v, r, c=(r - RISK_FREE) / v, cmap="viridis", s=4, alpha=0.4)
    fig.colorbar(sc, label="Sharpe")

    # Từng cổ phiếu
    vol_i = np.sqrt(np.diag(S.values))
    ax.scatter(vol_i, mu.values, marker="o", color="gray")
    for t, x, y in zip(mu.index, vol_i, mu.values):
        ax.annotate(t, (x, y), fontsize=8, xytext=(4, 2), textcoords="offset points")

    # Các chiến lược
    for name, wt in weights_dict.items():
        wt = wt.reindex(mu.index).fillna(0).values
        ax.scatter(np.sqrt(wt @ S.values @ wt), wt @ mu.values, marker="*", s=250, label=name, edgecolor="k")
    ax.set_xlabel("Biến động năm (σ)")
    ax.set_ylabel("Lợi nhuận kỳ vọng năm")
    ax.set_title("Efficient Frontier — cổ phiếu Việt Nam (in-sample)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/efficient_frontier.png", dpi=150)
    plt.close(fig)


# ============ BƯỚC 8: BACKTEST WALK-FORWARD (out-of-sample) ============
def backtest(prices, lookback_years=2, rebalance="QE", cost=0.0025):
    """Mỗi quý: tối ưu trên 2 năm trước đó, nắm giữ quý kế tiếp. cost = phí + thuế bán ~0,25% mỗi chiều xoay vòng."""
    rets = prices.pct_change().dropna()
    dates = rets.resample(rebalance).last().index
    start_i = next(i for i, d in enumerate(dates) if d >= rets.index[0] + pd.DateOffset(years=lookback_years))
    results, hist = {}, {}
    for name, fn in STRATEGIES.items():
        daily, prev_w, wh = [], pd.Series(0.0, index=prices.columns), {}
        for i in range(start_i, len(dates) - 1):
            train = prices.loc[dates[i] - pd.DateOffset(years=lookback_years): dates[i]]
            w = fn(train).reindex(prices.columns).fillna(0)
            wh[dates[i]] = w
            turnover = (w - prev_w).abs().sum()
            hold = rets.loc[dates[i]: dates[i + 1]].iloc[1:]
            r = hold @ w
            r.iloc[0] -= turnover * cost
            daily.append(r)
            # trôi tỷ trọng theo giá trong kỳ nắm giữ
            grown = w * (1 + hold).prod()
            prev_w = grown / grown.sum()
        results[name] = pd.concat(daily)
        hist[name] = pd.DataFrame(wh).T
    return pd.DataFrame(results), hist


def metrics(daily):
    eq = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    mdd = (eq / eq.cummax() - 1).min()
    return pd.DataFrame({
        "CAGR": cagr, "Volatility": vol,
        "Sharpe": (cagr - RISK_FREE) / vol,
        "Max Drawdown": mdd,
        "Calmar": cagr / -mdd,
    })


# ============ BƯỚC 9: QUY RA SỐ CỔ PHIẾU (lô 100) ============
def to_shares(weights, prices, capital=CAPITAL, lot=LOT):
    latest = prices.iloc[-1] * 1000   # vnstock trả giá theo nghìn đồng
    lot_price = latest * lot
    da = DiscreteAllocation(weights[weights > 0].to_dict(), lot_price, total_portfolio_value=capital)
    lots, leftover = da.greedy_portfolio()
    df = pd.DataFrame({"Số lô": lots}).assign(**{"Số CP": lambda d: d["Số lô"] * lot})
    df["Giá (VND)"] = latest.reindex(df.index)
    df["Giá trị"] = df["Số CP"] * df["Giá (VND)"]
    return df, leftover


# ============ BƯỚC 10: BÁO CÁO TƯƠNG TÁC (Plotly) ============
# Bảng màu: 8 màu phân loại đã kiểm tra an toàn cho người mù màu, dùng theo thứ tự cố định
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENCH_COLOR = "#52514e"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#f0f6fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _layout(fig, title, ytitle=None, yfmt=None, height=420):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color=INK)),
        height=height, margin=dict(l=60, r=24, t=56, b=48),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK2),
        legend=dict(orientation="h", y=-0.2, x=0, traceorder="normal", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(color=INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     title_text=ytitle, tickformat=yfmt)
    return fig


def build_report(prices, W, daily, m, weight_hist, demo=False):
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    strat = [c for c in daily.columns if c != "VN-Index"]
    color = {s: SERIES[i] for i, s in enumerate(strat)}
    color["VN-Index"] = BENCH_COLOR
    tick_color = {t: SERIES[i % 8] for i, t in enumerate(prices.columns)}

    def line_style(name):
        if name == "VN-Index":
            return dict(color=BENCH_COLOR, width=2, dash="dash")
        return dict(color=color[name], width=2)

    figs = []

    # 1. Đường giá trị danh mục
    eq = (1 + daily).cumprod()
    f = go.Figure()
    for c in eq.columns:
        f.add_trace(go.Scatter(x=eq.index, y=eq[c], name=c, line=line_style(c),
                               hovertemplate="%{y:.2f}×"))
    _layout(f, "Giá trị 1 đồng vốn đầu tư (out-of-sample, đã trừ phí)", "Lần vốn ban đầu", ".2f")
    f.update_layout(hovermode="x unified")
    figs.append(("equity", f, "wide"))

    # 2. Drawdown
    dd = eq / eq.cummax() - 1
    f = go.Figure()
    for c in dd.columns:
        f.add_trace(go.Scatter(x=dd.index, y=dd[c], name=c, line=line_style(c),
                               hovertemplate="%{y:.1%}"))
    _layout(f, "Mức sụt giảm từ đỉnh (drawdown)", None, ".0%", height=360)
    f.update_layout(hovermode="x unified")
    figs.append(("drawdown", f, "wide"))

    # 3. Rủi ro – lợi nhuận của từng chiến lược (backtest)
    f = go.Figure()
    for c in m.index:
        f.add_trace(go.Scatter(
            x=[m.loc[c, "Volatility"]], y=[m.loc[c, "CAGR"]], mode="markers+text", name=c,
            text=[c], textposition="top center", textfont=dict(color=INK2),
            marker=dict(size=14, color=color[c], line=dict(color=SURFACE, width=2),
                        symbol="diamond" if c == "VN-Index" else "circle"),
            hovertemplate=f"<b>{c}</b><br>Lợi nhuận/năm %{{y:.1%}}<br>Biến động %{{x:.1%}}"
                          f"<br>Sharpe {m.loc[c, 'Sharpe']:.2f}<extra></extra>"))
    _layout(f, "Rủi ro và lợi nhuận thực tế", "Lợi nhuận/năm (CAGR)", ".0%")
    f.update_xaxes(title_text="Biến động/năm", tickformat=".0%")
    f.update_layout(showlegend=False)
    figs.append(("riskret", f, "half"))

    # 4. Max Drawdown dạng cột ngang
    order = m["Max Drawdown"].sort_values().index
    f = go.Figure(go.Bar(
        y=list(order), x=m.loc[order, "Max Drawdown"], orientation="h",
        marker=dict(color=[color[c] for c in order], cornerradius=4),
        text=[f"{v:.1%}" for v in m.loc[order, "Max Drawdown"]], textposition="outside",
        textfont=dict(color=INK2), hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    _layout(f, "Sụt giảm sâu nhất (càng ngắn càng tốt)", None, None)
    f.update_xaxes(tickformat=".0%", range=[m["Max Drawdown"].min() * 1.25, 0])
    f.update_yaxes(gridcolor="rgba(0,0,0,0)")
    figs.append(("mdd", f, "half"))

    # 5. Tỷ trọng hiện tại (cột chồng 100%)
    f = go.Figure()
    for t in W.index:
        f.add_trace(go.Bar(y=list(W.columns), x=W.loc[t], name=t, orientation="h",
                           marker=dict(color=tick_color[t], line=dict(color=SURFACE, width=2)),
                           hovertemplate=f"<b>{t}</b> %{{x:.1%}}<extra>%{{y}}</extra>"))
    _layout(f, "Tỷ trọng đề xuất trên toàn bộ dữ liệu", None, None, height=380)
    f.update_layout(barmode="stack")
    f.update_xaxes(tickformat=".0%", range=[0, 1])
    f.update_yaxes(gridcolor="rgba(0,0,0,0)", autorange="reversed")
    figs.append(("weights", f, "wide"))

    # 6. Tỷ trọng theo thời gian: Max Sharpe vs HRP (small multiples)
    for name in ["Max Sharpe", "HRP"]:
        wh = weight_hist[name].reindex(columns=prices.columns).fillna(0)
        f = go.Figure()
        for t in wh.columns:
            f.add_trace(go.Bar(x=wh.index, y=wh[t], name=t,
                               marker=dict(color=tick_color[t], line=dict(color=SURFACE, width=1)),
                               hovertemplate=f"<b>{t}</b> %{{y:.1%}}<extra></extra>"))
        _layout(f, f"Tỷ trọng mỗi quý — {name}", None, ".0%", height=360)
        f.update_layout(barmode="stack", bargap=0.15, hovermode="x unified")
        f.update_yaxes(range=[0, 1])
        figs.append((f"hist_{name}", f, "half"))

    # 7. Ma trận tương quan
    corr = prices.pct_change().dropna().corr()
    f = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=0, zmax=1,
        colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
        text=np.round(corr.values, 2), texttemplate="%{text}",
        xgap=2, ygap=2, colorbar=dict(title="Tương quan", tickfont=dict(color=MUTED)),
        hovertemplate="%{y} – %{x}: %{z:.2f}<extra></extra>"))
    _layout(f, "Tương quan lợi nhuận ngày giữa các mã", None, None, height=440)
    f.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    figs.append(("corr", f, "half"))

    # 8. Efficient Frontier tương tác
    mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    S = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    rng = np.random.default_rng(0)
    rw = rng.dirichlet(np.ones(len(mu)), 3000)
    rr, rv = rw @ mu.values, np.sqrt(np.einsum("ij,jk,ik->i", rw, S.values, rw))
    targets = np.linspace(rr.min(), mu.clip(upper=mu.max()).max() * 0.98, 40)
    fr_v, fr_r = [], []
    for tr in targets:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
            ef.efficient_return(tr)
            r_, v_, _ = ef.portfolio_performance(risk_free_rate=RISK_FREE)
            fr_v.append(v_); fr_r.append(r_)
        except Exception:
            pass
    f = go.Figure()
    f.add_trace(go.Scatter(x=rv, y=rr, mode="markers", name="Danh mục ngẫu nhiên",
                           marker=dict(size=4, color=AXIS, opacity=0.5), hoverinfo="skip"))
    f.add_trace(go.Scatter(x=fr_v, y=fr_r, mode="lines", name="Efficient frontier",
                           line=dict(color=INK, width=2), hovertemplate="σ %{x:.1%} · R %{y:.1%}<extra></extra>"))
    vol_i = np.sqrt(np.diag(S.values))
    f.add_trace(go.Scatter(x=vol_i, y=mu.values, mode="markers+text", text=list(mu.index),
                           textposition="top right", textfont=dict(color=MUTED, size=11), name="Cổ phiếu",
                           marker=dict(size=8, color=MUTED), hovertemplate="%{text}<extra></extra>"))
    for c in W.columns:
        w = W[c].reindex(mu.index).fillna(0).values
        f.add_trace(go.Scatter(x=[np.sqrt(w @ S.values @ w)], y=[w @ mu.values], mode="markers",
                               name=c, marker=dict(size=16, symbol="star", color=color[c],
                                                   line=dict(color=SURFACE, width=2)),
                               hovertemplate=f"<b>{c}</b><br>σ %{{x:.1%}} · R %{{y:.1%}}<extra></extra>"))
    _layout(f, "Efficient Frontier (in-sample — mô hình đã “nhìn thấy đáp án”)", "Lợi nhuận kỳ vọng/năm", ".0%", height=440)
    f.update_xaxes(title_text="Biến động/năm", tickformat=".0%")
    figs.append(("frontier", f, "half"))

    # ---- Bảng chỉ số ----
    better_high = {"CAGR": True, "Volatility": False, "Sharpe": True, "Max Drawdown": True, "Calmar": True}
    labels = {"CAGR": "Lợi nhuận/năm", "Volatility": "Biến động", "Sharpe": "Sharpe",
              "Max Drawdown": "Sụt giảm sâu nhất", "Calmar": "Calmar"}
    pct = {"CAGR", "Volatility", "Max Drawdown"}
    best = {k: (m[k].idxmax() if better_high[k] else m[k].idxmin()) for k in m.columns}
    rows = ""
    for s in m.index:
        cells = "".join(
            f'<td class="{"best" if best[k] == s else ""}">{m.loc[s, k]:.1%}</td>' if k in pct
            else f'<td class="{"best" if best[k] == s else ""}">{m.loc[s, k]:.2f}</td>'
            for k in m.columns)
        sw = f'<span class="sw" style="background:{color[s]}"></span>'
        rows += f"<tr><th>{sw}{s}</th>{cells}</tr>"
    head = "".join(f"<th>{labels[k]}</th>" for k in m.columns)
    table = f'<table><thead><tr><th>Chiến lược</th>{head}</tr></thead><tbody>{rows}</tbody></table>'

    period = f"{daily.index[0]:%d/%m/%Y} – {daily.index[-1]:%d/%m/%Y}"
    note = "DỮ LIỆU GIẢ LẬP — chỉ để thử code" if demo else "Dữ liệu thật từ vnstock"
    cfg = dict(displaylogo=False, responsive=True)
    blocks = "".join(
        f'<section class="card {size}">{fig.to_html(full_html=False, include_plotlyjs=False, config=cfg)}</section>'
        for _, fig, size in figs)

    html = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Báo cáo tối ưu danh mục</title>
<script>{get_plotlyjs()}</script>
<style>
 body{{margin:0;background:#f9f9f7;color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
 main{{max-width:1240px;margin:0 auto;padding:24px 16px 48px}}
 h1{{font-size:24px;margin:0 0 4px}} .sub{{color:{INK2};margin:0 0 20px;font-size:14px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
 .card{{background:{SURFACE};border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:8px;min-width:0}}
 .wide{{grid-column:1/-1}}
 @media (max-width:860px){{.grid{{grid-template-columns:1fr}}}}
 .tablewrap{{overflow-x:auto}}
 table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
 th,td{{padding:9px 12px;text-align:right;border-bottom:1px solid {GRID};white-space:nowrap}}
 thead th{{color:{MUTED};font-weight:500}} tbody th{{text-align:left;font-weight:500}}
 td.best{{font-weight:700;color:#006300}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:0}}
 .foot{{color:{MUTED};font-size:12px;margin-top:20px}}
</style></head><body><main>
<h1>Báo cáo tối ưu danh mục cổ phiếu Việt Nam</h1>
<p class="sub">{note} · Backtest walk-forward {period} · Tái cân bằng mỗi quý · Lãi phi rủi ro {RISK_FREE:.0%} · Mã: {", ".join(prices.columns)}</p>
<section class="card wide tablewrap" style="padding:12px 16px;margin-bottom:16px">{table}
<p class="foot" style="margin:8px 0 0">In đậm xanh = tốt nhất ở mỗi cột. Rê chuột lên biểu đồ để xem số liệu; bấm vào chú thích để ẩn/hiện từng đường.</p></section>
<div class="grid">{blocks}</div>
<p class="foot">Công cụ học tập, không phải khuyến nghị đầu tư. Kết quả quá khứ không đảm bảo cho tương lai.</p>
</main></body></html>"""
    with open(f"{OUT}/report.html", "w", encoding="utf-8") as fh:
        fh.write(html)


# ============ CHẠY ============
if __name__ == "__main__":
    demo = "--demo" in sys.argv
    print("Bước 1: lấy dữ liệu", "(giả lập)" if demo else "(vnstock)")
    prices = load_prices_demo(TICKERS, START, END) if demo else load_prices_vnstock(TICKERS, START, END)
    prices = clean(prices)
    prices.to_csv(f"{OUT}/prices.csv")
    print(f"  Dữ liệu sạch: {prices.shape[0]} phiên × {prices.shape[1]} mã")

    print("\nBước 3-6: tỷ trọng trên toàn bộ dữ liệu")
    W = pd.DataFrame({n: f(prices) for n, f in STRATEGIES.items()}).fillna(0)
    print((W * 100).round(1).to_string())
    W.to_csv(f"{OUT}/weights.csv")

    print("\nBước 7: vẽ Efficient Frontier →", f"{OUT}/efficient_frontier.png")
    plot_frontier(prices, {c: W[c] for c in W})

    print("\nBước 8: backtest walk-forward (tái cân bằng mỗi quý, có phí)")
    daily, weight_hist = backtest(prices)
    bench = load_benchmark_demo(prices) if demo else load_benchmark_vnstock(START, END)
    if bench is not None:
        b = bench.reindex(prices.index).ffill().pct_change()
        daily["VN-Index"] = b.reindex(daily.index).fillna(0)
    m = metrics(daily)
    print(m.round(3).to_string())
    m.to_csv(f"{OUT}/backtest_metrics.csv")
    ax = (1 + daily).cumprod().plot(figsize=(10, 5), title="Giá trị danh mục out-of-sample (1 = vốn ban đầu)")
    ax.set_ylabel("Giá trị")
    plt.tight_layout()
    plt.savefig(f"{OUT}/backtest_equity.png", dpi=150)
    plt.close()

    print("\nBước 9: quy ra số cổ phiếu cho chiến lược HRP")
    alloc, left = to_shares(W["HRP"], prices)
    print(alloc.to_string())
    print(f"  Tiền mặt còn lại: {left:,.0f} VND")

    print("\nBước 10: tạo báo cáo tương tác →", f"{OUT}/report.html")
    try:
        build_report(prices, W, daily, m, weight_hist, demo)
        print("  Mở bằng lệnh:  start output\\report.html")
    except ImportError:
        print("  Cần cài plotly:  python -m pip install plotly")
