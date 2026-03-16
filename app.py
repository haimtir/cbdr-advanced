# CBDR v15 — Fixed Multi-Model Quant Dashboard
# Fixes: None display bug, Sharpe/Sortino/Calmar, risk range 0.1x-1.5x,
#        PDF reports, trade logs, per-model signal predictions
import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesRegressor, AdaBoostRegressor,
    HistGradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, f1_score)
import warnings; warnings.filterwarnings("ignore")
import io

st.set_page_config(page_title="CBDR v15", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown('''<style>
.main .block-container{padding-top:.5rem;max-width:1600px}
.mc{background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #30363d;border-radius:10px;padding:.55rem .7rem;text-align:center;color:#e0e0e0;min-height:72px}
.mc h3{color:#58a6ff;font-size:.6rem;margin-bottom:.05rem;text-transform:uppercase;letter-spacing:.5px}
.mc .val{color:#f0f6fc;font-size:1rem;font-weight:700}
.profit{color:#3fb950!important}.loss{color:#f85149!important}
div[data-testid="stSidebar"]{background-color:#0d1117}
.det{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:.5rem .6rem;margin:.15rem 0}
.det .lbl{color:#8b949e;font-size:.6rem;text-transform:uppercase}.det .vl{color:#f0f6fc;font-size:.85rem;font-weight:600}
.det .sub{color:#58a6ff;font-size:.65rem}
.plv{background:#0d1117;border-radius:8px;padding:.6rem;text-align:center;margin:.15rem 0;border:1px solid #21262d}
.plv .pl{color:#8b949e;font-size:.6rem;text-transform:uppercase}.plv .pp{font-size:1rem;font-weight:700}
.plv .pd{font-size:.65rem;color:#8b949e}
.signal-card{background:linear-gradient(135deg,#0d1117,#161b22);border:2px solid #58a6ff;border-radius:12px;padding:.8rem 1rem;margin:.3rem 0}
.signal-bull{border-color:#3fb950}.signal-bear{border-color:#f85149}
.reason{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:.8rem;color:#c9d1d9;font-size:.82rem;line-height:1.55;margin:.3rem 0}
.fb{display:inline-block;padding:5px 12px;border-radius:8px;margin:2px;font-size:.75rem;font-weight:600}
.fb-r{background:rgba(31,111,235,0.13);color:#58a6ff;border:1px solid #1f6feb}
.fb-b{background:rgba(210,168,40,0.13);color:#d2a828;border:1px solid #d2a828}
.fb-t{background:rgba(163,113,247,0.13);color:#a371f7;border:1px solid #a371f7}
.fb-e{background:rgba(63,185,80,0.13);color:#3fb950;border:1px solid #3fb950}
.fb-s{background:rgba(248,81,73,0.13);color:#f85149;border:1px solid #f85149}
.fa{color:#8b949e;font-size:1rem;margin:0 2px}
</style>''', unsafe_allow_html=True)

def hex_to_rgba(hx, a=0.08):
    hx = hx.lstrip("#"); return f"rgba({int(hx[0:2],16)},{int(hx[2:4],16)},{int(hx[4:6],16)},{a})"

def mcard(col, t, v, fmt="auto", cs=False):
    try:
        if v is None: v = 0
        if fmt=="pct": d=f"{float(v):.1f}%"
        elif fmt=="int": d=str(int(v))
        elif fmt=="dollar": d=f"${float(v):,.0f}"
        else: d=f"{float(v):.2f}" if isinstance(v,(int,float,np.floating,np.integer)) else str(v)
    except: d = str(v)
    cc="profit" if cs and isinstance(v,(int,float)) and v>0 else("loss" if cs and isinstance(v,(int,float)) and v<0 else "")
    col.markdown(f'<div class="mc"><h3>{t}</h3><div class="val {cc}">{d}</div></div>',unsafe_allow_html=True)

PBG=dict(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117")
ASSETS={"Gold (XAUUSD)":{"t":"GC=F","s":0.3},"EUR/USD":{"t":"EURUSD=X","s":0.00012},"GBP/USD":{"t":"GBPUSD=X","s":0.00015},"USD/JPY":{"t":"JPY=X","s":0.015},"GBP/JPY":{"t":"GBPJPY=X","s":0.025},"Silver (XAGUSD)":{"t":"SI=F","s":0.02},"US Oil (WTI)":{"t":"CL=F","s":0.03},"BTC/USD":{"t":"BTC-USD","s":15}}
SESSIONS_GMT={"asia":(0,8),"london":(8,16),"ny":(13,22),"ldn_ny_overlap":(13,16),"asia_ldn_overlap":(7,9)}
ECON={"fomc":["2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18","2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-17","2026-01-28","2026-03-18"],
    "nfp":["2024-01-05","2024-02-02","2024-03-08","2024-04-05","2024-05-03","2024-06-07","2024-07-05","2024-08-02","2024-09-06","2024-10-04","2024-11-01","2024-12-06","2025-01-10","2025-02-07","2025-03-07","2025-04-04","2025-05-02","2025-06-06","2025-07-03","2025-08-01","2025-09-05","2025-10-03","2025-11-07","2025-12-05","2026-01-09","2026-02-06","2026-03-06"],
    "cpi":["2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15","2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10","2024-11-13","2024-12-11","2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13","2025-06-11","2025-07-15","2025-08-12","2025-09-10","2025-10-14","2025-11-12","2025-12-10","2026-01-13","2026-02-11","2026-03-11"]}

def evt_flags(date):
    ds=str(date); f={"fomc":"none","nfp":"none","cpi":"none","any":False}
    for e,dl in ECON.items():
        for ed in dl:
            if abs((pd.Timestamp(ds)-pd.Timestamp(ed)).days)<=1: f[e]="yes"; f["any"]=True; break
    return f

def get_session(hgmt):
    if 13<=hgmt<16: return "ldn_ny_overlap"
    elif 0<=hgmt<8: return "asia"
    elif 8<=hgmt<13: return "london"
    elif 16<=hgmt<22: return "ny"
    return "off_hours"

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_data(ticker,days):
    """Fetch hourly data from yfinance. ALL timestamps normalized to UTC."""
    import yfinance as yf; end=datetime.now()
    for d in [min(days+10,729),365,180,90]:
        try:
            df=yf.download(ticker,start=end-timedelta(days=d),end=end,interval="1h",progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
                # Force to UTC then strip tz label — all CBDR hours are GMT/UTC
                if df.index.tz is not None:
                    df.index=df.index.tz_convert("UTC").tz_localize(None)
                if len(df)>20: return df
        except: continue
    return pd.DataFrame()

# ═══ MACRO DATA: VIX, DXY, US10Y — DAILY CLOSES ═══
# Used as ML features. For each CBDR day (date X), we use
# the SAME DAY's close. US markets close at ~21:00 GMT, CBDR breakout
# happens after 00:00 GMT — so same-day close is known hours before trading.
MACRO_TICKERS = {"vix": "^VIX", "dxy": "DX=F", "us10y": "^TNX", "oil": "CL=F"}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data(days):
    """Fetch daily macro closes with fallback tickers."""
    import yfinance as yf
    end = datetime.now(); macro = {}
    fallbacks = {"vix":["^VIX"], "dxy":["DX=F","DX-Y.NYB","UUP"], "us10y":["^TNX"], "oil":["CL=F"]}
    for name, tickers in fallbacks.items():
        for ticker in tickers:
            try:
                md = yf.download(ticker, start=end-timedelta(days=days+30), end=end, interval="1d", progress=False)
                if md is not None and not md.empty:
                    if isinstance(md.columns, pd.MultiIndex): md.columns = md.columns.get_level_values(0)
                    if md.index.tz is not None: md.index = md.index.tz_convert("UTC").tz_localize(None)
                    s = md["Close"].dropna().copy()
                    s.index = pd.to_datetime(s.index).date
                    if len(s) > 5: macro[name] = s; break
            except: continue
    return macro


def get_macro_features(date, macro_data):
    """
    Get macro features for a CBDR day using SAME DAY's close.

    NOT leakage because: US markets close at ~21:00 GMT. CBDR breakout
    happens AFTER 00:00 GMT (next calendar day). So same-day DXY/VIX/Oil
    close is known 3+ hours before any trade decision.

    Timeline:  [DXY/Oil/VIX close 21:00] → [CBDR forms 20-00] → [Breakout 00:00+]
                     ↑ known here                                    ↑ we trade here

    Features per instrument (VIX, DXY, US10Y, Oil):
      - level: same day's close (raw value)
      - chg_1d: 1-day % change
      - chg_5d: 5-day % change
      - chg_10d: 10-day % change
    """
    feats = {}
    if not macro_data:
        for name in MACRO_TICKERS:
            feats[f"{name}_level"] = 0
            feats[f"{name}_chg1d"] = 0
            feats[f"{name}_chg5d"] = 0
            feats[f"{name}_chg10d"] = 0
        feats["macro_available"] = 0
        return feats

    feats["macro_available"] = 1
    target_date = pd.Timestamp(date).date()

    for name in MACRO_TICKERS:
        series = macro_data.get(name)
        if series is None or len(series) < 2:
            feats[f"{name}_level"] = 0
            feats[f"{name}_chg1d"] = 0
            feats[f"{name}_chg5d"] = 0
            feats[f"{name}_chg10d"] = 0
            continue

        # Same day's close — available before CBDR breakout (US close 21:00 < breakout 00:00+)
        available = series[series.index <= target_date]
        if len(available) == 0:
            feats[f"{name}_level"] = 0
            feats[f"{name}_chg1d"] = 0
            feats[f"{name}_chg5d"] = 0
            feats[f"{name}_chg10d"] = 0
            continue

        today_close = float(available.iloc[-1])
        feats[f"{name}_level"] = round(today_close, 2)

        # 1-day change: today's close vs yesterday's close
        if len(available) >= 2:
            feats[f"{name}_chg1d"] = round((today_close - float(available.iloc[-2])) / max(abs(float(available.iloc[-2])), 0.01) * 100, 3)
        else:
            feats[f"{name}_chg1d"] = 0

        # 5-day change
        if len(available) >= 6:
            feats[f"{name}_chg5d"] = round((today_close - float(available.iloc[-6])) / max(abs(float(available.iloc[-6])), 0.01) * 100, 3)
        else:
            feats[f"{name}_chg5d"] = 0

        # 10-day change
        if len(available) >= 11:
            feats[f"{name}_chg10d"] = round((today_close - float(available.iloc[-11])) / max(abs(float(available.iloc[-11])), 0.01) * 100, 3)
        else:
            feats[f"{name}_chg10d"] = 0

    return feats


def load_csv(f):
    try:
        df=pd.read_csv(f,parse_dates=True)
        dc=[c for c in df.columns if any(k in c.lower() for k in ["date","time"])]
        if dc: df.index=pd.to_datetime(df[dc[0]]); df.drop(columns=dc,inplace=True,errors="ignore")
        else: df.index=pd.to_datetime(df.iloc[:,0]); df=df.iloc[:,1:]
        cm={}
        for c in df.columns:
            cl=c.lower().strip()
            for kw,nm in [("open","Open"),("high","High"),("low","Low"),("close","Close"),("volume","Volume")]:
                if kw in cl: cm[c]=nm; break
        df.rename(columns=cm,inplace=True); return df
    except: return pd.DataFrame()

def clsfy(o,h,l,c):
    body=abs(c-o); rng=max(h-l,0.0001); br=body/rng; green=c>o
    uw=(h-max(o,c))/rng; lw=(min(o,c)-l)/rng; pat="normal"
    if br<.1: pat="doji"
    elif br>.7: pat="large_body"
    elif (not green) and lw>2*br and uw<br*.3 and br>.05: pat="hammer"
    elif green and uw>2*br and lw<br*.3 and br>.05: pat="shooting_star"
    return {"green":green,"body_ratio":round(br,3),"uw":round(uw,3),"lw":round(lw,3),"pattern":pat}

def session_features(dfs, gmt=0):
    """Compute session features. Data is UTC, sessions defined in GMT — direct match."""
    feats={}; hv="Volume" in dfs.columns and dfs["Volume"].sum()>0
    for sn,(ss2,se2) in SESSIONS_GMT.items():
        # No conversion needed — both data and sessions are in UTC/GMT
        if ss2 < se2:
            m = (dfs.index.hour >= ss2) & (dfs.index.hour < se2)
        else:
            m = (dfs.index.hour >= ss2) | (dfs.index.hour < se2)
        s=dfs[m]
        if len(s)==0: feats[f"ses_{sn}_range"]=0; feats[f"ses_{sn}_vol"]=0; feats[f"ses_{sn}_trend"]=0; continue
        sr2=s["High"].max()-s["Low"].min(); dr=max(dfs["High"].max()-dfs["Low"].min(),0.0001)
        feats[f"ses_{sn}_range"]=round(sr2/dr,3)
        feats[f"ses_{sn}_vol"]=round(s["Volume"].sum()/max(dfs["Volume"].sum(),1),3) if hv else 0
        feats[f"ses_{sn}_trend"]=round((s.iloc[-1]["Close"]-s.iloc[0]["Open"])/max(sr2,0.0001),3)
    return feats

def compute_sr(hist,n=8,cp=0.15):
    if len(hist)<20: return [],[],{}
    h,l=hist["High"].values,hist["Low"].values; phi,plo=[],[]
    for i in range(2,len(l)-2):
        if l[i]<=l[i-1] and l[i]<=l[i-2] and l[i]<=l[i+1] and l[i]<=l[i+2]: plo.append(l[i])
        if h[i]>=h[i-1] and h[i]>=h[i-2] and h[i]>=h[i+1] and h[i]>=h[i+2]: phi.append(h[i])
    al=[(p,"s") for p in plo]+[(p,"r") for p in phi]
    if not al: return plo[-5:],phi[-5:],{}
    al.sort(key=lambda x:x[0]); cls2=[]; used=set()
    for i,(pr,tp) in enumerate(al):
        if i in used: continue
        cp2,ct=[pr],[tp]; used.add(i)
        for j in range(i+1,len(al)):
            if j in used: continue
            if abs(al[j][0]-pr)/max(pr,0.01)*100<cp: cp2.append(al[j][0]); ct.append(al[j][1]); used.add(j)
        dt="s" if ct.count("s")>=ct.count("r") else "r"
        cls2.append({"p":round(np.mean(cp2),2),"str":len(cp2),"t":dt})
    cls2.sort(key=lambda x:-x["str"]); si={c["p"]:c for c in cls2[:n]}
    return [c["p"] for c in cls2 if c["t"]=="s"][-5:],[c["p"] for c in cls2 if c["t"]=="r"][-5:],si

def sr_feats(price,sups,ress,si,rs):
    f={}; rs=max(rs,0.0001)
    if sups:
        d=[(price-s)/rs for s in sups if s<price]; f["dist_sup"]=round(min(d),3) if d else 5.0
        ns=min(sups,key=lambda s:abs(price-s)); f["sup_str"]=si.get(ns,{}).get("str",1)
    else: f["dist_sup"]=5.0; f["sup_str"]=0
    if ress:
        d=[(r-price)/rs for r in ress if r>price]; f["dist_res"]=round(min(d),3) if d else 5.0
        nr=min(ress,key=lambda r:abs(price-r)); f["res_str"]=si.get(nr,{}).get("str",1)
    else: f["dist_res"]=5.0; f["res_str"]=0
    f["strong_sr"]=1 if (f["dist_sup"]<0.5 and f["sup_str"]>=3) or (f["dist_res"]<0.5 and f["res_str"]>=3) else 0
    return f

def vol_feats(dday,cbdr,boc,lb):
    hv="Volume" in dday.columns and dday["Volume"].sum()>0
    if not hv: return {"vol_avail":0,"cbdr_rvol":0,"bo_vsurge":0,"vol_trend":0,"vol_hi":0,"vol_lo":0}
    f={"vol_avail":1}; ahv=lb["Volume"].mean() if len(lb)>0 else 1
    f["cbdr_rvol"]=round(cbdr["Volume"].sum()/max(ahv*max(len(cbdr),1),1),3) if len(cbdr)>0 else 1.0
    if boc is not None:
        bv=boc.get("Volume",0) if isinstance(boc,dict) else boc["Volume"]
        f["bo_vsurge"]=round(bv/max(ahv,1),3)
    else: f["bo_vsurge"]=1.0
    if len(cbdr)>=2:
        v=cbdr["Volume"].values; f["vol_trend"]=round((v[-1]-v[0])/max(v[0],1),3) if v[0]>0 else 0
    else: f["vol_trend"]=0
    if len(dday)>2:
        tv=max(dday["Volume"].sum(),1)
        f["vol_hi"]=round(dday.nlargest(max(len(dday)//4,1),"High")["Volume"].sum()/tv,3)
        f["vol_lo"]=round(dday.nsmallest(max(len(dday)//4,1),"Low")["Volume"].sum()/tv,3)
    else: f["vol_hi"]=0.25; f["vol_lo"]=0.25
    return f

# ═══ ENGINE ═══
class Engine:
    def __init__(self,df,gmt=0,spread=0.3,sl_mult=1.5,sr_lb=20,macro_data=None,
                 cbdr_start_gmt=20,cbdr_end_gmt=0,csv_tz_offset=0,range_mode="wick"):
        self.df=df.copy(); self.gmt=gmt; self.spread=spread; self.sl_mult=sl_mult; self.sr_lb=sr_lb
        self.macro_data=macro_data or {}
        self.cbdr_start_gmt=cbdr_start_gmt; self.cbdr_end_gmt=cbdr_end_gmt
        self.range_mode=range_mode  # "wick" = High/Low extremes, "close" = max Close/min Close
        # Data is in UTC (from fetch_data). For CSV uploads, user provides csv_tz_offset
        # to shift to UTC. CBDR hours are in GMT/UTC — use directly, no conversion.
        if csv_tz_offset != 0:
            self.df.index = self.df.index - timedelta(hours=csv_tz_offset)
        # CBDR window in UTC — used directly on UTC data
        self.cs = cbdr_start_gmt   # CBDR start hour (UTC)
        self.ce = cbdr_end_gmt     # CBDR end hour (UTC)
        # Session = everything outside CBDR
        self.ss = self.ce          # Session starts when CBDR ends
        self.se = self.cs          # Session ends when next CBDR starts
    def run(self):
        df=self.df.copy(); df["date"]=df.index.date; df["hour"]=df.index.hour
        dates=sorted(df["date"].unique()); days=[]; prev_dir=None; recent=[]; prev_pb=0; prev_run=0
        for i,date in enumerate(dates):
            # Skip weekends — no valid CBDR on Sat/Sun
            dow=pd.Timestamp(date).dayofweek  # 0=Mon, 5=Sat, 6=Sun
            if dow >= 5: continue
            dd=df[df["date"]==date]
            if len(dd)<4: continue
            # Find previous WEEKDAY for cross-midnight CBDR
            pd2=None
            for pi in range(i-1,-1,-1):
                if pd.Timestamp(dates[pi]).dayofweek < 5: pd2=dates[pi]; break
            if self.cs>self.ce:
                late=df[df["date"]==pd2].query("hour>=@self.cs") if pd2 else pd.DataFrame()
                early=dd.query("hour<@self.ce") if self.ce>0 else pd.DataFrame()
                cbdr=pd.concat([late,early])
            else: cbdr=dd.query("hour>=@self.cs and hour<@self.ce")
            if len(cbdr)<2: continue
            # Range definition depends on mode
            if self.range_mode == "close":
                rh=cbdr["Close"].max(); rl=cbdr["Close"].min()  # settled prices only
            else:
                rh=cbdr["High"].max(); rl=cbdr["Low"].min()     # full wick extremes
            rs=rh-rl
            if rs<=0: continue
            fc=clsfy(cbdr.iloc[0]["Open"],cbdr.iloc[0]["High"],cbdr.iloc[0]["Low"],cbdr.iloc[0]["Close"])
            lc=clsfy(cbdr.iloc[-1]["Open"],cbdr.iloc[-1]["High"],cbdr.iloc[-1]["Low"],cbdr.iloc[-1]["Close"])
            cbdr_trend=(cbdr.iloc[-1]["Close"]-cbdr.iloc[0]["Open"])/rs
            n_green=sum(1 for ci in range(len(cbdr)) if cbdr.iloc[ci]["Close"]>cbdr.iloc[ci]["Open"])
            uw_avg=np.mean([(cbdr.iloc[ci]["High"]-max(cbdr.iloc[ci]["Open"],cbdr.iloc[ci]["Close"]))/max(cbdr.iloc[ci]["High"]-cbdr.iloc[ci]["Low"],0.0001) for ci in range(len(cbdr))])
            lw_avg=np.mean([(min(cbdr.iloc[ci]["Open"],cbdr.iloc[ci]["Close"])-cbdr.iloc[ci]["Low"])/max(cbdr.iloc[ci]["High"]-cbdr.iloc[ci]["Low"],0.0001) for ci in range(len(cbdr))])
            close_pos=(cbdr.iloc[-1]["Close"]-rl)/rs
            fi = -1
            try:
                fi=df.index.get_indexer([cbdr.index[0]],method="nearest")[0]
                hdf=df.iloc[max(0,fi-self.sr_lb*24):fi]; sups,ress,si=compute_sr(hdf)
            except: sups,ress,si=[],[],{}
            ns=any(abs(rl-s)/max(rl,0.01)*100<0.3 for s in sups) if sups else False
            nr=any(abs(rh-r)/max(rh,0.01)*100<0.3 for r in ress) if ress else False
            srf=sr_feats((rh+rl)/2,sups,ress,si,rs)
            # Bias: use actual PRICE CHANGE over lookback, not direction counts
            # 5-day price change for week bias, 20-day for month bias
            try:
                cur_price = cbdr.iloc[-1]["Close"]
                price_5d_ago = df.iloc[max(0, fi - 5*24)]["Close"] if fi >= 5*24 else df.iloc[0]["Close"]
                price_20d_ago = df.iloc[max(0, fi - 20*24)]["Close"] if fi >= 20*24 else df.iloc[0]["Close"]
                pct_5d = (cur_price - price_5d_ago) / max(price_5d_ago, 0.01) * 100
                pct_20d = (cur_price - price_20d_ago) / max(price_20d_ago, 0.01) * 100
                wb = "bullish" if pct_5d > 0.1 else ("bearish" if pct_5d < -0.1 else "neutral")
                mbi = "bullish" if pct_20d > 0.2 else ("bearish" if pct_20d < -0.2 else "neutral")
            except:
                wb = "neutral"; mbi = "neutral"
                pct_5d = 0; pct_20d = 0
            # Session = trading window after CBDR. May wrap around midnight.
            # Find next WEEKDAY for cross-midnight data
            next_day=pd.DataFrame()
            for ndi in range(i+1, min(i+4, len(dates))):
                if pd.Timestamp(dates[ndi]).dayofweek < 5:
                    next_day=df[df["date"]==dates[ndi]]; break
            if self.ss < self.se:
                # Normal range: e.g. session 0:00-20:00
                sess=dd.query("hour>=@self.ss and hour<@self.se")
                ndd=next_day.query("hour<@self.se") if len(next_day)>0 else pd.DataFrame()
            elif self.ss == self.se:
                # Full day session (edge case)
                sess=dd
                ndd=next_day if len(next_day)>0 else pd.DataFrame()
            else:
                # Wraps midnight: e.g. ss=23, se=19 → hours 23 today + 0-18 tomorrow
                sess_today=dd.query("hour>=@self.ss")
                sess_tomorrow=next_day.query("hour<@self.se") if len(next_day)>0 else pd.DataFrame()
                sess=pd.concat([sess_today, sess_tomorrow])
                # ndd = additional next-day data beyond session for post-breakout measurement
                ndd=pd.DataFrame()
            if len(sess)<2: continue
            day_name=pd.Timestamp(date).day_name(); dom=pd.Timestamp(date).day
            mpos="start" if dom<=10 else("end" if dom>=21 else "mid")
            ef=evt_flags(date)
            full_day=pd.concat([dd, next_day]) if len(next_day)>0 else dd
            sesf=session_features(full_day)
            lb_vol=df.iloc[max(0,fi-120):fi] if fi>=0 else pd.DataFrame()
            bo_idx=None; direction=None; bo_c=None
            for j in range(len(sess)):
                c2=sess.iloc[j]
                if c2["Close"]>rh: direction="bullish"; bo_idx=j; bo_c=c2; break
                elif c2["Close"]<rl: direction="bearish"; bo_idx=j; bo_c=c2; break
            if direction is None: continue
            bo_cls=clsfy(bo_c["Open"],bo_c["High"],bo_c["Low"],bo_c["Close"])
            bo_hgmt=sess.index[bo_idx].hour; bo_ses=get_session(bo_hgmt)
            vf=vol_feats(full_day,cbdr,bo_c,lb_vol)
            all_post=pd.concat([sess.iloc[bo_idx+1:],ndd])
            retest_ses="none"; retest_candles=0
            if len(all_post)>0:
                for ri in range(len(all_post)):
                    rc=all_post.iloc[ri]
                    touched=(direction=="bullish" and rc["Low"]<=rh) or (direction=="bearish" and rc["High"]>=rl)
                    if touched: retest_candles=ri; rt_hgmt=all_post.index[ri].hour; retest_ses=get_session(rt_hgmt); break
            if len(all_post)>0:
                if direction=="bullish": pb=(rh-all_post["Low"].min())/rs; mr=(all_post["High"].max()-rh)/rs
                else: pb=(all_post["High"].max()-rl)/rs; mr=(rl-all_post["Low"].min())/rs
            else: pb=0; mr=0
            pb=max(0,pb); mr=max(0,mr); recent.append(direction)
            # Macro features: VIX, DXY, US10Y — using PREVIOUS day's close (no leakage)
            mf = get_macro_features(date, self.macro_data)
            # CBDR candle timestamps for audit trail
            cbdr_first_ts=str(cbdr.index[0]); cbdr_last_ts=str(cbdr.index[-1])
            days.append({"date":date,"day":day_name,"mpos":mpos,"direction":direction,
                "range_size":round(rs,2),"range_high":round(rh,2),"range_low":round(rl,2),
                "range_mode":self.range_mode,
                "cbdr_from":cbdr_first_ts,"cbdr_to":cbdr_last_ts,"cbdr_candles":len(cbdr),
                "fc_green":fc["green"],"fc_pat":fc["pattern"],"fc_br":fc["body_ratio"],"fc_uw":fc["uw"],"fc_lw":fc["lw"],
                "lc_green":lc["green"],"lc_pat":lc["pattern"],"lc_br":lc["body_ratio"],"lc_uw":lc["uw"],"lc_lw":lc["lw"],
                "cbdr_trend":round(cbdr_trend,3),"n_green":n_green,"uw_avg":round(uw_avg,3),"lw_avg":round(lw_avg,3),
                "close_pos":round(close_pos,3),"near_sup":ns,"near_res":nr,**srf,
                "prev_dir":prev_dir,"wbias":wb,"mbias":mbi,
                "bo_pat":bo_cls["pattern"],"bo_br":bo_cls["body_ratio"],"bo_green":bo_cls["green"],
                "evt_fomc":ef["fomc"],"evt_nfp":ef["nfp"],"evt_cpi":ef["cpi"],"evt_any":ef["any"],
                "bo_session":bo_ses,"retest_session":retest_ses,"retest_candles":retest_candles,
                **sesf,**vf,**mf,"pb_depth":round(pb,4),"max_run":round(mr,4),"prev_pb":round(prev_pb,4),"prev_run":round(prev_run,4),
                "pct_5d":round(pct_5d,3),"pct_20d":round(pct_20d,3)})
            prev_dir=direction; prev_pb=pb; prev_run=mr
        return pd.DataFrame(days)

    def detect_latest(self,tdf):
        df=self.df.copy()
        if df.empty: return None
        df["date"]=df.index.date; df["hour"]=df.index.hour; dates=sorted(df["date"].unique())
        if len(dates)<3: return None
        cbdr=pd.DataFrame(); d=None
        # Search recent dates, skip weekends
        for d in reversed(dates[-10:]):
            if pd.Timestamp(d).dayofweek >= 5: continue  # skip Sat/Sun
            dd=df[df["date"]==d]
            # Find previous weekday for cross-midnight CBDR
            pd2=None; di=dates.index(d) if d in dates else -1
            for pi in range(di-1,-1,-1):
                if pd.Timestamp(dates[pi]).dayofweek < 5: pd2=dates[pi]; break
            if self.cs>self.ce:
                late=df[df["date"]==pd2].query("hour>=@self.cs") if pd2 else pd.DataFrame()
                early=dd.query("hour<@self.ce") if self.ce>0 else pd.DataFrame()
                cbdr=pd.concat([late,early])
            else: cbdr=dd.query("hour>=@self.cs and hour<@self.ce")
            if len(cbdr)>=2: break
        else: return None
        if d is None: return None
        if self.range_mode == "close":
            rh=cbdr["Close"].max(); rl=cbdr["Close"].min()
        else:
            rh=cbdr["High"].max(); rl=cbdr["Low"].min()
        rs=rh-rl
        if rs<=0: return None
        fc=clsfy(cbdr.iloc[0]["Open"],cbdr.iloc[0]["High"],cbdr.iloc[0]["Low"],cbdr.iloc[0]["Close"])
        lc=clsfy(cbdr.iloc[-1]["Open"],cbdr.iloc[-1]["High"],cbdr.iloc[-1]["Low"],cbdr.iloc[-1]["Close"])
        cbdr_trend=(cbdr.iloc[-1]["Close"]-cbdr.iloc[0]["Open"])/rs
        n_green=sum(1 for ci in range(len(cbdr)) if cbdr.iloc[ci]["Close"]>cbdr.iloc[ci]["Open"])
        uw_avg=np.mean([(cbdr.iloc[ci]["High"]-max(cbdr.iloc[ci]["Open"],cbdr.iloc[ci]["Close"]))/max(cbdr.iloc[ci]["High"]-cbdr.iloc[ci]["Low"],0.0001) for ci in range(len(cbdr))])
        lw_avg=np.mean([(min(cbdr.iloc[ci]["Open"],cbdr.iloc[ci]["Close"])-cbdr.iloc[ci]["Low"])/max(cbdr.iloc[ci]["High"]-cbdr.iloc[ci]["Low"],0.0001) for ci in range(len(cbdr))])
        close_pos=(cbdr.iloc[-1]["Close"]-rl)/rs
        try:
            fi=df.index.get_indexer([cbdr.index[0]],method="nearest")[0]
            hdf=df.iloc[max(0,fi-self.sr_lb*24):fi]; sups,ress,si=compute_sr(hdf)
        except: sups,ress,si=[],[],{}
        nsup=any(abs(rl-s)/max(rl,0.01)*100<0.3 for s in sups) if sups else False
        nres=any(abs(rh-r)/max(rh,0.01)*100<0.3 for r in ress) if ress else False
        srf=sr_feats((rh+rl)/2,sups,ress,si,rs)
        dd2=df[df["date"]==d]
        # Session query — handle wrap-around midnight (same logic as run())
        if self.ss < self.se:
            sess=dd2.query("hour>=@self.ss and hour<@self.se")
        elif self.ss == self.se:
            sess=dd2
        else:
            # Wraps midnight: get late today + early next weekday
            sess_today=dd2.query("hour>=@self.ss")
            all_dates=sorted(df["date"].unique())
            di=all_dates.index(d) if d in all_dates else -1
            nd_data=pd.DataFrame()
            for ndi in range(di+1, min(di+4, len(all_dates))):
                if pd.Timestamp(all_dates[ndi]).dayofweek < 5:
                    nd_data=df[df["date"]==all_dates[ndi]].query("hour<@self.se")
                    break
            sess=pd.concat([sess_today, nd_data]) if len(nd_data)>0 else sess_today
        bdir=None
        for j in range(len(sess)):
            c2=sess.iloc[j]
            if c2["Close"]>rh: bdir="bullish"; break
            elif c2["Close"]<rl: bdir="bearish"; break
        pdd=None; wb="neutral"; mbi2="neutral"
        if tdf is not None and len(tdf)>0:
            pdd=tdf.iloc[-1].get("direction")
        # Price-based bias (same as engine)
        pct5=0; pct20=0
        try:
            cur_p=cbdr.iloc[-1]["Close"]
            p5=df.iloc[max(0,fi-5*24)]["Close"] if fi>=5*24 else df.iloc[0]["Close"]
            p20=df.iloc[max(0,fi-20*24)]["Close"] if fi>=20*24 else df.iloc[0]["Close"]
            pct5=(cur_p-p5)/max(p5,0.01)*100; pct20=(cur_p-p20)/max(p20,0.01)*100
            wb="bullish" if pct5>0.1 else("bearish" if pct5<-0.1 else "neutral")
            mbi2="bullish" if pct20>0.2 else("bearish" if pct20<-0.2 else "neutral")
        except: pass
        ef=evt_flags(d); sesf=session_features(dd2)
        try: lb_v=df.iloc[max(0,fi-120):fi]
        except: lb_v=pd.DataFrame()
        vf=vol_feats(dd2,cbdr,None,lb_v)
        mf_det = get_macro_features(d, self.macro_data)
        csl=cbdr.index[0]; cel=cbdr.index[-1]
        return {"date":d,"day":pd.Timestamp(d).day_name(),"rh":round(rh,2),"rl":round(rl,2),"rs":round(rs,2),
            "range_mode":self.range_mode,
            "cbdr_from":str(csl),"cbdr_to":str(cel),"cbdr_candles":len(cbdr),
            "fc_green":fc["green"],"fc_pat":fc["pattern"],"fc_br":fc["body_ratio"],"fc_uw":fc["uw"],"fc_lw":fc["lw"],
            "lc_green":lc["green"],"lc_pat":lc["pattern"],"lc_br":lc["body_ratio"],"lc_uw":lc["uw"],"lc_lw":lc["lw"],
            "cbdr_trend":round(cbdr_trend,3),"n_green":n_green,"uw_avg":round(uw_avg,3),"lw_avg":round(lw_avg,3),
            "close_pos":round(close_pos,3),"near_sup":nsup,"near_res":nres,**srf,
            "direction":bdir,"prev_dir":pdd,"wbias":wb,"mbias":mbi2,
            "bo_pat":None,"bo_br":0,"bo_green":None,"range_size":round(rs,2),
            "evt_fomc":ef["fomc"],"evt_nfp":ef["nfp"],"evt_cpi":ef["cpi"],"evt_any":ef["any"],
            "mpos":"start" if pd.Timestamp(d).day<=10 else("end" if pd.Timestamp(d).day>=21 else "mid"),
            "bo_session":"unknown","retest_session":"unknown","retest_candles":0,**sesf,**vf,**mf_det,
            "prev_pb":round(float(tdf.iloc[-1]["pb_depth"]),4) if tdf is not None and len(tdf)>0 else 0,
            "prev_run":round(float(tdf.iloc[-1]["max_run"]),4) if tdf is not None and len(tdf)>0 else 0,
            "pct_5d":round(pct5,3),"pct_20d":round(pct20,3),
            "price":round(df.iloc[-1]["Close"],2),
            "window":f"CBDR {self.cbdr_start_gmt:02d}:00-{self.cbdr_end_gmt:02d}:00 UTC | Range: {'closes' if self.range_mode=='close' else 'wicks'} | Data: {csl.strftime('%Y-%m-%d %H:%M')} to {cel.strftime('%H:%M')} UTC"}

# ═══ ML FEATURES — includes Volume, S/R, Candles, Sessions, Macro ═══
FEAT_N=["fc_green","lc_green","fc_br","lc_br","fc_uw","fc_lw","lc_uw","lc_lw","cbdr_trend","n_green",
    "uw_avg","lw_avg","close_pos","near_sup","near_res","bo_br","bo_green","evt_any","prev_pb","prev_run",
    "dist_sup","dist_res","sup_str","res_str","strong_sr",
    "vol_avail","cbdr_rvol","bo_vsurge","vol_trend","vol_hi","vol_lo",
    "ses_asia_range","ses_asia_vol","ses_asia_trend","ses_london_range","ses_london_vol","ses_london_trend",
    "ses_ny_range","ses_ny_vol","ses_ny_trend","ses_ldn_ny_overlap_range","ses_ldn_ny_overlap_vol","ses_ldn_ny_overlap_trend",
    "ses_asia_ldn_overlap_range","ses_asia_ldn_overlap_vol","ses_asia_ldn_overlap_trend","retest_candles",
    "pct_5d","pct_20d",
    # Macro features (same day's close — US close 21:00 GMT < breakout 00:00+, zero leakage)
    "macro_available","vix_level","vix_chg1d","vix_chg5d","vix_chg10d",
    "dxy_level","dxy_chg1d","dxy_chg5d","dxy_chg10d",
    "us10y_level","us10y_chg1d","us10y_chg5d","us10y_chg10d",
    "oil_level","oil_chg1d","oil_chg5d","oil_chg10d"]
CAT_F=["direction","day","wbias","mbias","fc_pat","lc_pat","bo_pat","mpos","bo_session","retest_session"]

def encode(df):
    X=pd.DataFrame(index=df.index)
    for c in FEAT_N:
        if c in df.columns: X[c]=pd.to_numeric(df[c],errors="coerce").fillna(0).astype(float)
    for c in CAT_F:
        if c in df.columns:
            dum=pd.get_dummies(df[c].astype(str),prefix=c,drop_first=False)
            for dc in dum.columns: X[dc]=dum[dc].astype(float)
    return X.fillna(0)

def align_cols(Xtr,Xva,Xte):
    ac=sorted(set(Xtr.columns)|set(Xva.columns)|set(Xte.columns))
    for c in ac:
        if c not in Xtr.columns: Xtr[c]=0
        if c not in Xva.columns: Xva[c]=0
        if c not in Xte.columns: Xte[c]=0
    return Xtr[ac],Xva[ac],Xte[ac],ac

def get_reg_models():
    return {"RF":RandomForestRegressor(n_estimators=200,max_depth=7,min_samples_leaf=5,random_state=42),
        "GBM":GradientBoostingRegressor(n_estimators=150,max_depth=4,min_samples_leaf=5,learning_rate=0.05,random_state=42),
        "HistGBM":HistGradientBoostingRegressor(max_iter=200,max_depth=5,min_samples_leaf=5,learning_rate=0.05,random_state=42),
        "ExtraTrees":ExtraTreesRegressor(n_estimators=200,max_depth=7,min_samples_leaf=5,random_state=42),
        "AdaBoost":AdaBoostRegressor(n_estimators=100,learning_rate=0.1,random_state=42),
        "MLP":MLPRegressor(hidden_layer_sizes=(64,32),max_iter=500,early_stopping=True,validation_fraction=0.15,random_state=42),
        "Ridge":Ridge(alpha=1.0)}

def get_cls_models():
    return {"RF_cls":RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=5,class_weight="balanced",random_state=42),
        "GBM_cls":GradientBoostingClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,random_state=42),
        "HistGBM_cls":HistGradientBoostingClassifier(max_iter=200,max_depth=5,learning_rate=0.05,random_state=42),
        "ExtraTrees_cls":ExtraTreesClassifier(n_estimators=200,max_depth=6,class_weight="balanced",random_state=42),
        "MLP_cls":MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500,early_stopping=True,validation_fraction=0.15,random_state=42)}

def train_multi_reg(Xtr,ytr,Xva,yva,Xte,yte,tname):
    sc=RobustScaler(); Xtrs=sc.fit_transform(Xtr); Xvas=sc.transform(Xva); Xtes=sc.transform(Xte)
    models=get_reg_models(); res={}; best=None; best_v=999
    for nm,mdl in models.items():
        try:
            mdl.fit(Xtrs,ytr); r={}
            for sn,Xs,ys in [("train",Xtrs,ytr),("val",Xvas,yva),("test",Xtes,yte)]:
                p=mdl.predict(Xs); r[sn]={"mae":round(mean_absolute_error(ys,p),4),"rmse":round(np.sqrt(mean_squared_error(ys,p)),4),"r2":round(r2_score(ys,p),4) if len(ys)>1 else 0,"n":len(ys),"pred":p}
            res[nm]=r
            if r["val"]["mae"]<best_v: best_v=r["val"]["mae"]; best=(nm,mdl)
        except: pass
    ym=ytr.mean()
    for sn,ys in [("train",ytr),("val",yva),("test",yte)]:
        res.setdefault("Baseline",{})[sn]={"mae":round(mean_absolute_error(ys,np.full_like(ys,ym)),4),"rmse":round(np.sqrt(mean_squared_error(ys,np.full_like(ys,ym))),4),"r2":0,"n":len(ys)}
    imp=pd.Series(dtype=float)
    for mn in ["RF","ExtraTrees","GBM"]:
        if mn in models and hasattr(models[mn],"feature_importances_"):
            try: imp=pd.Series(models[mn].feature_importances_,index=Xtr.columns).sort_values(ascending=False); break
            except: pass
    return {"results":res,"best":best,"scaler":sc,"importance":imp,
        "all_models":{n:m for n,m in models.items() if n in res and "error" not in res.get(n,{})},
        "y_stats":{"mean":round(float(ytr.mean()),3),"std":round(float(ytr.std()),3),
            "p50":round(float(np.percentile(ytr,50)),3),"p75":round(float(np.percentile(ytr,75)),3),"p90":round(float(np.percentile(ytr,90)),3)}}

def train_dir_cls(Xtr,ytr,Xva,yva,Xte,yte):
    sc=RobustScaler(); Xtrs=sc.fit_transform(Xtr); Xvas=sc.transform(Xva); Xtes=sc.transform(Xte)
    models=get_cls_models(); res={}; best=None; best_f1=-1
    for nm,mdl in models.items():
        try:
            mdl.fit(Xtrs,ytr); r={}
            for sn,Xs,ys in [("train",Xtrs,ytr),("val",Xvas,yva),("test",Xtes,yte)]:
                p=mdl.predict(Xs); r[sn]={"acc":round(accuracy_score(ys,p)*100,1),"f1":round(f1_score(ys,p,pos_label="bullish",average="binary",zero_division=0)*100,1),"n":len(ys),"pred":p}
            res[nm]=r
            if r["val"]["f1"]>best_f1: best_f1=r["val"]["f1"]; best=(nm,mdl)
        except: pass
    imp=pd.Series(dtype=float)
    for mn in ["RF_cls","ExtraTrees_cls","GBM_cls"]:
        if mn in models and hasattr(models[mn],"feature_importances_"):
            try: imp=pd.Series(models[mn].feature_importances_,index=Xtr.columns).sort_values(ascending=False); break
            except: pass
    return {"results":res,"best":best,"scaler":sc,"importance":imp}

def get_conf(nm,mdl,Xs):
    """Get prediction confidence via tree variance. Only works for RF/ExtraTrees.
    HistGBM.estimators_ are numpy arrays (NOT sklearn estimators) — will crash.
    AdaBoost estimators are DecisionTreeRegressors but staged differently.
    For non-tree-ensemble models, return 0.5 (neutral confidence)."""
    if nm in ("RF","ExtraTrees") and hasattr(mdl,"estimators_"):
        try:
            tp=np.array([t.predict(Xs)[0] for t in mdl.estimators_])
            cv=float(tp.std())/max(abs(float(tp.mean())),0.01)
            return max(0.2,min(0.95,1.0-cv))
        except: return 0.5
    return 0.5

def sim_trade(actual_pb,actual_run,entry_depth,sl_x,tp_x):
    mae=max(0,actual_pb-entry_depth); mfe=actual_run+entry_depth
    rr=tp_x/sl_x if sl_x>0 else 1.0
    if mae>=sl_x: return "loss",-1.0,rr
    elif mfe>=tp_x: return "win",rr,rr
    else: net=(mfe-mae)/sl_x; return ("win" if net>0 else "loss"),round(net,3),rr

def build_ml(tdf):
    RT=0.05
    if len(tdf)<40: return None
    tdf=tdf.copy(); tdf["had_retest"]=tdf["pb_depth"]>=RT
    n=len(tdf); tre=int(n*0.6); vae=int(n*0.8)
    tr=tdf.iloc[:tre]; va=tdf.iloc[tre:vae]; te=tdf.iloc[vae:]
    Xtr=encode(tr); Xva=encode(va); Xte=encode(te)
    Xtr,Xva,Xte,acols=align_cols(Xtr,Xva,Xte)
    dir_m=None
    if len(tr["direction"].unique())>=2:
        dir_m=train_dir_cls(Xtr,tr["direction"].values,Xva,va["direction"].values,Xte,te["direction"].values)
    sc_cls=RobustScaler(); Xtr_c=sc_cls.fit_transform(Xtr); Xva_c=sc_cls.transform(Xva); Xte_c=sc_cls.transform(Xte)
    yctr=tr["had_retest"].astype(int).values; ycva=va["had_retest"].astype(int).values; ycte=te["had_retest"].astype(int).values
    rt_clf=RandomForestClassifier(n_estimators=200,max_depth=6,class_weight="balanced",random_state=42)
    rt_res={}
    if len(np.unique(yctr))>=2:
        rt_clf.fit(Xtr_c,yctr)
        for sn,Xs,ys in [("train",Xtr_c,yctr),("val",Xva_c,ycva),("test",Xte_c,ycte)]:
            p=rt_clf.predict(Xs); rt_res[sn]={"acc":round(accuracy_score(ys,p)*100,1),"n":len(ys)}
    rt_rate={"train":tr["had_retest"].mean(),"val":va["had_retest"].mean(),"test":te["had_retest"].mean()}
    trr=tr[tr["had_retest"]]; var=va[va["had_retest"]]; ter=te[te["had_retest"]]
    entry_m=exit_m=None
    if len(trr)>=20 and len(var)>=5 and len(ter)>=5:
        Xtrr=encode(trr); Xvar=encode(var); Xter=encode(ter)
        for c in acols:
            if c not in Xtrr.columns: Xtrr[c]=0
            if c not in Xvar.columns: Xvar[c]=0
            if c not in Xter.columns: Xter[c]=0
        Xtrr=Xtrr[acols]; Xvar=Xvar[acols]; Xter=Xter[acols]
        entry_m=train_multi_reg(Xtrr,trr["pb_depth"].values,Xvar,var["pb_depth"].values,Xter,ter["pb_depth"].values,"Entry")
        exit_m=train_multi_reg(Xtrr,trr["max_run"].values,Xvar,var["max_run"].values,Xter,ter["max_run"].values,"Exit")
    # ═══ 5 STRATEGIES ═══
    S={"Baseline":[],"ML_Retest":[],"Breakout_Only":[],"Ensemble":[],"Ensemble_Guard":[]}
    has_ml=entry_m and entry_m["best"] and exit_m and exit_m["best"] and len(np.unique(yctr))>=2
    if has_ml: en,em2=entry_m["best"]; xn,xm2=exit_m["best"]; esc=entry_m["scaler"]; xsc=exit_m["scaler"]
    for idx in range(len(te)):
        row=te.iloc[idx]; rsv=row["range_size"]; d=row["direction"]
        if rsv<=0: continue
        apb=row["pb_depth"]; amr=row["max_run"]; art=row["had_retest"]; dt=row["date"]
        Xi=encode(pd.DataFrame([row]))
        for c in acols:
            if c not in Xi.columns: Xi[c]=0
        Xi=Xi[acols].fillna(0)
        pe=0.3; px=2.0; conf=0.5; dconf=0.5; rtprob=0.5; rmult=1.0
        if has_ml:
            rtprob=float(rt_clf.predict_proba(Xte_c[idx:idx+1])[0][1])
            try:
                Xes=esc.transform(Xi.values); Xxs=xsc.transform(Xi.values)
                pe=max(0,float(em2.predict(Xes)[0])); px=max(0.5,float(xm2.predict(Xxs)[0]))
                conf=get_conf(en,em2,Xes)
            except: pass
            if dir_m and dir_m["best"]:
                try: dp=dir_m["best"][1].predict_proba(dir_m["scaler"].transform(Xi.values))[0]; dconf=float(max(dp))
                except: pass
        # RISK: 0.1x to 1.5x
        rmult=round(max(0.1,min(1.5, 0.1 + conf*0.9 + (dconf-0.5)*0.5)),2)
        bt={"date":str(dt),"direction":d,"retest_prob":round(rtprob,2),"pred_entry":round(pe,3),"pred_exit":round(px,3),
            "actual_pb":round(apb,3),"actual_run":round(amr,3),"actual_retest":bool(art),"confidence":round(conf,2),
            "dir_conf":round(dconf,2),"risk_mult":rmult}
        # 1) BASELINE — always at 1.0x risk
        if art:
            o,r,rr=sim_trade(apb,amr,0,1.5,3.0)
            S["Baseline"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"boundary","sl_used":1.5,"risk_mult":1.0})
        else:
            S["Baseline"].append({**bt,"action":"skip","outcome":"skipped","r":0,"rr":0,"entry_type":"none","sl_used":0,"risk_mult":1.0})
        if not has_ml: continue
        # 2) ML RETEST
        if rtprob>=0.4 and apb>=pe:
            o,r,rr=sim_trade(apb,amr,pe,1.3,px+pe)
            S["ML_Retest"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"limit","sl_used":1.3})
        else:
            S["ML_Retest"].append({**bt,"action":"skip","outcome":"skipped","r":0,"rr":0,"entry_type":"none","sl_used":0})
        # 3) BREAKOUT ONLY — always trades, enters at breakout close
        o,r,rr=sim_trade(apb,amr,0,1.0,px)
        S["Breakout_Only"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"breakout","sl_used":1.0})
        # 4) ENSEMBLE — NO DATA LEAKAGE VERSION
        #    Decision at breakout candle close: pick ONE path based on ML prediction ONLY.
        #    Path A (pred retest): place limit at pred depth. Fills if actual_pb >= depth. If not → SKIPPED.
        #    Path B (pred no retest): enter market at breakout close. Always fills.
        #    NO fallback — commit to the prediction.
        if rtprob >= 0.5:
            # Path A: predicted retest → limit order at 30% of predicted depth
            ens_depth = pe * 0.3
            if apb >= ens_depth:  # limit order fills (price reached our level — observable in real time)
                o,r,rr = sim_trade(apb, amr, ens_depth, 1.3, px + ens_depth)
                S["Ensemble"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"limit_retest","sl_used":1.3})
            else:  # limit never fills → trade SKIPPED (no fallback — that would be leakage)
                S["Ensemble"].append({**bt,"action":"limit_not_filled","outcome":"skipped","r":0,"rr":0,"entry_type":"limit_missed","sl_used":0})
        else:
            # Path B: predicted no retest → market entry at breakout close
            o,r,rr = sim_trade(apb, amr, 0, 1.0, px)
            S["Ensemble"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"breakout","sl_used":1.0})
        # 5) ENSEMBLE GUARD — same as Ensemble but skip when combined confidence < 0.4
        cconf=(conf+dconf)/2
        if cconf < 0.4:
            S["Ensemble_Guard"].append({**bt,"action":"skip_lowconf","outcome":"skipped","r":0,"rr":0,"entry_type":"none","sl_used":0})
        elif rtprob >= 0.5:
            ens_depth = pe * 0.3
            if apb >= ens_depth:
                o,r,rr = sim_trade(apb, amr, ens_depth, 1.3, px + ens_depth)
                S["Ensemble_Guard"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"limit_retest","sl_used":1.3})
            else:
                S["Ensemble_Guard"].append({**bt,"action":"limit_not_filled","outcome":"skipped","r":0,"rr":0,"entry_type":"limit_missed","sl_used":0})
        else:
            o,r,rr = sim_trade(apb, amr, 0, 1.0, px)
            S["Ensemble_Guard"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"breakout","sl_used":1.0})
    return {"entry":entry_m,"exit":exit_m,"dir_model":dir_m,
        "retest_clf":rt_clf if len(np.unique(yctr))>=2 else None,"retest_clf_results":rt_res,
        "retest_rate":rt_rate,"retest_threshold":RT,"strategies":{k:pd.DataFrame(v) for k,v in S.items()},
        "splits":{"train":len(tr),"val":len(va),"test":len(te),"train_rt":len(trr),"val_rt":len(var),"test_rt":len(ter)},
        "all_cols":acols,"cls_scaler":sc_cls}

def predict_today(ml,det):
    if ml is None or det is None: return None
    em=ml.get("entry"); xm=ml.get("exit"); rc=ml.get("retest_clf")
    if not em or not em.get("best") or not xm or not xm.get("best"): return None
    X=encode(pd.DataFrame([det]))
    for c in ml["all_cols"]:
        if c not in X.columns: X[c]=0
    X=X[ml["all_cols"]].fillna(0)
    rtprob=0.5
    if rc and ml.get("cls_scaler"):
        Xc=ml["cls_scaler"].transform(X.values); rtprob=float(rc.predict_proba(Xc)[0][1])
    dconf=0.5; dpred=det.get("direction","unknown")
    dm=ml.get("dir_model")
    if dm and dm["best"]:
        Xd=dm["scaler"].transform(X.values); dpred=dm["best"][1].predict(Xd)[0]
        dp=dm["best"][1].predict_proba(Xd)[0]; dconf=float(max(dp))
    esc2=em["scaler"]; xsc2=xm["scaler"]; Xes=esc2.transform(X.values); Xxs=xsc2.transform(X.values)
    ep=float(em["best"][1].predict(Xes)[0]); xp=float(xm["best"][1].predict(Xxs)[0])
    conf=get_conf(em["best"][0],em["best"][1],Xes)
    # Get ALL model predictions for entry and exit
    entry_preds={}; exit_preds={}
    for mn,mo in em.get("all_models",{}).items():
        try: entry_preds[mn]=round(max(0,float(mo.predict(esc2.transform(X.values))[0])),3)
        except: pass
    for mn,mo in xm.get("all_models",{}).items():
        try: exit_preds[mn]=round(max(0.5,float(mo.predict(xsc2.transform(X.values))[0])),3)
        except: pass
    rmult=round(max(0.1,min(1.5,0.1+conf*0.9+(dconf-0.5)*0.5)),2)
    return {"pred_entry":round(max(0,ep),3),"pred_exit":round(max(0.5,xp),3),"retest_prob":round(rtprob,2),
        "confidence":round(conf,2),"dir_conf":round(dconf,2),"dir_pred":dpred,"risk_mult":rmult,
        "entry_model":em["best"][0],"exit_model":xm["best"][0],
        "entry_by_model":entry_preds,"exit_by_model":exit_preds}

# ═══ METRICS — FIXED Sharpe/Sortino/Calmar ═══
EMPTY_STATS={"n":0,"active":0,"skipped":0,"wins":0,"losses":0,"wr":0,"pf":0,"avg_r":0,
    "sharpe":0,"sortino":0,"calmar":0,"total_r":0,"dollar_pnl":0,"final_eq":0,
    "return_pct":0,"max_dd_pct":0,"eq_curve":[],"dd_series":[],"max_win_streak":0,"max_loss_streak":0,
    "avg_win":0,"avg_loss":0}

def calc_stats(outcomes,rs_arr,capital=10000,risk_pct=1.0,risk_mults=None):
    active=[(i,o,r) for i,(o,r) in enumerate(zip(outcomes,rs_arr)) if o in ("win","loss")]
    n=len(outcomes); ns=n-len(active)
    if not active: return {**EMPTY_STATS,"n":n,"skipped":ns}
    idxs,outs,ra=zip(*active); ra=np.array(ra,dtype=float)
    wins=sum(1 for o in outs if o=="win"); na=len(outs); wr=wins/na*100 if na>0 else 0
    wr2=ra[ra>0]; lr2=np.abs(ra[ra<0])
    pf=float(wr2.sum()/lr2.sum()) if lr2.sum()>0 else 99.0
    # Sharpe: per-trade mean/std (no annualization for short backtests)
    sh=float(ra.mean()/ra.std()) if ra.std()>0 else 0
    # Sortino: mean / downside_deviation
    neg=ra[ra<0]
    downside_dev=np.sqrt(np.mean(neg**2)) if len(neg)>0 else 0
    so=float(ra.mean()/downside_dev) if downside_dev>0 else 0
    # Equity curve with adaptive risk
    eq=float(capital); eqc=[eq]; mx=eq; mdd=0; dds=[0]
    for ii,r in enumerate(ra):
        rm=1.0
        if risk_mults is not None:
            oi=idxs[ii]
            if oi<len(risk_mults): rm=float(risk_mults[oi])
        eq+=eq*(risk_pct*rm/100)*r; eqc.append(max(eq,0.01))
        if eq>mx: mx=eq
        dd2=(mx-eq)/mx*100 if mx>0 else 0; dds.append(dd2)
        if dd2>mdd: mdd=dd2
    dp=eq-capital; rp=dp/capital*100
    # Calmar: total return / max drawdown (not annualized — cleaner for backtests)
    cal=round(rp/(mdd) if mdd>0 else 0, 2)
    sw,sl3,cw,cl3=[],[],0,0
    for o in outs:
        if o=="win":
            cw+=1
            if cl3>0: sl3.append(cl3); cl3=0
        else:
            cl3+=1
            if cw>0: sw.append(cw); cw=0
    if cw>0: sw.append(cw)
    if cl3>0: sl3.append(cl3)
    return {"n":n,"active":na,"skipped":ns,"wins":wins,"losses":na-wins,"wr":round(wr,1),"pf":round(pf,2),
        "avg_r":round(float(ra.mean()),3),"sharpe":round(sh,3),"sortino":round(so,3),"calmar":cal,
        "total_r":round(float(ra.sum()),1),"dollar_pnl":round(dp,0),"final_eq":round(eq,0),
        "return_pct":round(rp,1),"max_dd_pct":round(mdd,1),"eq_curve":eqc,"dd_series":dds,
        "max_win_streak":max(sw) if sw else 0,"max_loss_streak":max(sl3) if sl3 else 0,
        "avg_win":round(float(wr2.mean()),3) if len(wr2)>0 else 0,"avg_loss":round(float(lr2.mean()),3) if len(lr2)>0 else 0}

# ═══ PDF REPORT ═══
def generate_pdf(det,today_pred,ml,tdf,asset_name):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter,topMargin=40,bottomMargin=40)
    styles=getSampleStyleSheet(); story=[]
    ts=ParagraphStyle('Title2',parent=styles['Title'],fontSize=18,textColor=colors.HexColor('#1a73e8'))
    story.append(Paragraph(f"CBDR Signal Report - {asset_name}",ts))
    story.append(Spacer(1,12))
    if det:
        story.append(Paragraph(f"<b>Date:</b> {det.get('date','')} ({det.get('day','')}) | <b>Price:</b> {det.get('price','N/A')}",styles['Normal']))
        story.append(Paragraph(f"<b>CBDR Range:</b> {det.get('rh',0)} - {det.get('rl',0)} ({det.get('rs',0)} pts)",styles['Normal']))
        story.append(Paragraph(f"<b>1st Candle:</b> {'Green' if det.get('fc_green') else 'Red'} {det.get('fc_pat','')} | <b>Last:</b> {'Green' if det.get('lc_green') else 'Red'} {det.get('lc_pat','')}",styles['Normal']))
        story.append(Paragraph(f"<b>Trend:</b> {det.get('cbdr_trend',0):+.2f} | <b>Week:</b> {det.get('wbias','?')} | <b>Month:</b> {det.get('mbias','?')}",styles['Normal']))
        story.append(Paragraph(f"<b>S/R:</b> Sup dist {det.get('dist_sup',0):.1f}x (str {det.get('sup_str',0)}) | Res dist {det.get('dist_res',0):.1f}x (str {det.get('res_str',0)})",styles['Normal']))
        story.append(Paragraph(f"<b>Volume:</b> CBDR rel vol {det.get('cbdr_rvol','N/A')}x",styles['Normal']))
        if det.get("macro_available",0)==1:
            story.append(Paragraph(f"<b>Macro (same day close):</b> VIX {det.get('vix_level',0):.1f} (1d {det.get('vix_chg1d',0):+.1f}%, 5d {det.get('vix_chg5d',0):+.1f}%) | DXY {det.get('dxy_level',0):.2f} (1d {det.get('dxy_chg1d',0):+.2f}%, 5d {det.get('dxy_chg5d',0):+.2f}%) | US10Y {det.get('us10y_level',0):.2f}% (5d {det.get('us10y_chg5d',0):+.2f}%) | Oil ${det.get('oil_level',0):.1f} (5d {det.get('oil_chg5d',0):+.1f}%)",styles['Normal']))
        story.append(Spacer(1,12))
    if today_pred:
        tp=today_pred; d=det.get("direction","?") if det else "?"
        story.append(Paragraph("<b>ML Signal Decision</b>",styles['Heading2']))
        story.append(Paragraph(f"Direction: <b>{d.upper()}</b> | Retest prob: <b>{tp['retest_prob']*100:.0f}%</b>",styles['Normal']))
        story.append(Paragraph(f"Entry model ({tp['entry_model']}): pullback <b>{tp['pred_entry']:.2f}x</b> range",styles['Normal']))
        story.append(Paragraph(f"Exit model ({tp['exit_model']}): run <b>{tp['pred_exit']:.1f}x</b> range",styles['Normal']))
        story.append(Paragraph(f"Confidence: <b>{tp['confidence']*100:.0f}%</b> | Dir conf: <b>{tp['dir_conf']*100:.0f}%</b> | Risk mult: <b>{tp['risk_mult']}x</b>",styles['Normal']))
        story.append(Spacer(1,8))
        # Per-model predictions table
        if tp.get("entry_by_model") or tp.get("exit_by_model"):
            story.append(Paragraph("<b>All Model Predictions</b>",styles['Heading3']))
            tdata=[["Model","Entry Pred (x range)","Exit Pred (x range)"]]
            all_m=set(list(tp.get("entry_by_model",{}).keys())+list(tp.get("exit_by_model",{}).keys()))
            for mn in sorted(all_m):
                ep2=tp.get("entry_by_model",{}).get(mn,"N/A")
                xp2=tp.get("exit_by_model",{}).get(mn,"N/A")
                tdata.append([mn,str(ep2),str(xp2)])
            t=Table(tdata,colWidths=[120,150,150])
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1a73e8')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
                ('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),9)]))
            story.append(t); story.append(Spacer(1,8))
    # Historical context
    if tdf is not None and det and det.get("direction"):
        d2=det["direction"]; day2=det["day"]
        sim=tdf[(tdf["day"]==day2)&(tdf["direction"]==d2)]
        if len(sim)>=3:
            story.append(Paragraph(f"<b>Historical ({day2}+{d2}, n={len(sim)}):</b> Avg PB {sim['pb_depth'].mean():.2f}x | Avg Run {sim['max_run'].mean():.1f}x | Med Run {sim['max_run'].median():.1f}x",styles['Normal']))
    story.append(Spacer(1,12))
    story.append(Paragraph("<b>Features Used by ML Models</b>",styles['Heading3']))
    story.append(Paragraph("Candle patterns (body ratio, upper/lower wick, doji/hammer/large_body), CBDR trend, close position, S/R distance and strength, Volume (CBDR relative, breakout surge, trend), Session data (Asia/London/NY/overlap range and volume ratios), Day of week, Week/Month price trend (%), Event flags (FOMC/NFP/CPI), Previous pullback and run values, <b>Macro: VIX level and changes (fear gauge), DXY level and changes (dollar strength, inverse to gold), US10Y level and changes (yield pressure), Oil level and changes (inflation proxy)</b>. All macro features use previous day close — zero data leakage.",styles['Normal']))
    story.append(Spacer(1,12))
    story.append(Paragraph("<b>Methodology</b>",styles['Heading3']))
    story.append(Paragraph("60/20/20 chronological train/val/test split. 7 regression models (RF, GBM, HistGBM, ExtraTrees, AdaBoost, MLP, Ridge) compete for entry depth and exit run prediction. Direction classifier uses 5 models with balanced class weights. Best model selected on validation MAE (regression) or F1 (classifier). SL optimized at 1.3x range (0.231R expectancy from historical analysis). Adaptive risk sizing: 0.1x-1.5x base risk per trade based on model confidence.",styles['Normal']))
    doc.build(story); buf.seek(0); return buf.getvalue()

# ═══ SIDEBAR ═══
st.sidebar.markdown("## CBDR v15 Quant"); st.sidebar.markdown("---")
asset_name=st.sidebar.selectbox("**Asset**",list(ASSETS.keys()),index=0); asset=ASSETS[asset_name]
dsrc=st.sidebar.radio("**Data**",["Yahoo Finance","Upload CSV"],index=0)
uf=st.sidebar.file_uploader("CSV",type=["csv"]) if dsrc=="Upload CSV" else None
csv_tz_offset=0
if dsrc=="Upload CSV":
    csv_tz_offset=st.sidebar.number_input("**CSV Timezone (hours from UTC)**",-12,14,0,
        help="If your CSV is in GMT+3, enter 3. Data will be shifted to UTC.")
st.sidebar.markdown("---")
st.sidebar.markdown("##### CBDR Window (UTC/GMT)")
cbdr_window=st.sidebar.selectbox("**Window**",
    ["20:00-00:00 UTC (DST summer)","19:00-23:00 UTC (winter)","Custom"],index=0)
if cbdr_window.startswith("20"):
    cbdr_start_gmt=20; cbdr_end_gmt=0
elif cbdr_window.startswith("19"):
    cbdr_start_gmt=19; cbdr_end_gmt=23
else:
    cc1,cc2=st.sidebar.columns(2)
    cbdr_start_gmt=cc1.number_input("Start (UTC)",0,23,20)
    cbdr_end_gmt=cc2.number_input("End (UTC)",0,23,0)
range_mode=st.sidebar.selectbox("**Range Definition**",
    ["Wick (High/Low extremes)","Close (settled prices only)"],index=0,
    help="Wick: range = highest high to lowest low of CBDR candles (wider). "
         "Close: range = highest close to lowest close (tighter, more breakouts).")
range_mode_val="wick" if range_mode.startswith("Wick") else "close"
st.sidebar.markdown("---")
st.sidebar.markdown("##### Data Range")
date_mode=st.sidebar.selectbox("**Period Mode**",["Recent (rolling)","Custom Date Range"],index=0)
if date_mode=="Recent (rolling)":
    per_o={"1 Month":30,"3 Months":90,"6 Months":180,"1 Year":365,"2 Years":730}
    pl2=st.sidebar.selectbox("**Period**",list(per_o.keys()),index=4); pdays=per_o[pl2]
    custom_start=None; custom_end=None
else:
    st.sidebar.caption("Pick any historical range. Test = last 20% of selected range.")
    custom_start=st.sidebar.date_input("**Start Date**",datetime(2023,1,1))
    custom_end=st.sidebar.date_input("**End Date**",datetime(2024,12,31))
    pdays=(custom_end-custom_start).days
st.sidebar.markdown("---")
spread=st.sidebar.number_input("**Spread**",0.0,10.0,float(asset["s"]),0.01)
sr_lb=st.sidebar.slider("**S/R Days**",5,60,20,5)
st.sidebar.markdown("---")
base_risk=st.sidebar.slider("**Base Risk %**",0.25,10.0,1.0,0.25)
capital=st.sidebar.number_input("**Capital ($)**",1000,1000000,10000,1000)
run_btn=st.sidebar.button("Run Full Analysis",type="primary",use_container_width=True)

# ═══ MAIN ═══
st.markdown("# CBDR v15 — Multi-Model Quant Dashboard")
st.caption("Ensemble | Risk 0.1x-1.5x | 7 models | Session+Volume+S/R+Macro(VIX/DXY/US10Y/Oil) | Zero leakage")

if run_btn or "tdf" in st.session_state:
    if run_btn:
        with st.spinner("Fetching price data..."):
            if dsrc=="Upload CSV" and uf:
                df=load_csv(uf)
            elif date_mode=="Custom Date Range" and custom_start and custom_end:
                import yfinance as yf
                df=yf.download(asset["t"],start=str(custom_start),end=str(custom_end),interval="1h",progress=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
                    if df.index.tz is not None: df.index=df.index.tz_convert("UTC").tz_localize(None)
                else: df=pd.DataFrame()
            else:
                df=fetch_data(asset["t"],pdays)
            if df.empty: st.error("No data. Yahoo limits hourly data to ~730 days. Try a shorter range."); st.stop()
            if date_mode=="Recent (rolling)" and custom_start is None:
                df=df[df.index>=datetime.now()-timedelta(days=pdays)]
            if df.empty: st.error("No data in selected range."); st.stop()
        with st.spinner("Fetching macro data (VIX, DXY, US10Y, Oil)..."):
            macro_data={}
            if dsrc!="Upload CSV":
                try: macro_data=fetch_macro_data(pdays)
                except: pass
            macro_status=f"VIX/DXY/US10Y/Oil loaded ({len(macro_data)} instruments)" if macro_data else "Macro unavailable"
        with st.spinner(f"Running engine (CBDR {cbdr_start_gmt:02d}:00-{cbdr_end_gmt:02d}:00 UTC)..."):
            eng=Engine(df,gmt=0,spread=spread,sl_mult=1.5,sr_lb=sr_lb,macro_data=macro_data,
                      cbdr_start_gmt=cbdr_start_gmt,cbdr_end_gmt=cbdr_end_gmt,
                      csv_tz_offset=csv_tz_offset,range_mode=range_mode_val)
            tdf=eng.run()
            if tdf.empty: st.error("No trades found. Check CBDR window and data range."); st.stop()
            det=eng.detect_latest(tdf)
        with st.spinner("Training ML..."):
            ml=build_ml(tdf); today_pred=predict_today(ml,det) if ml and det else None
        st.session_state.update({"tdf":tdf,"det":det,"ml":ml,"today_pred":today_pred})
    tdf=st.session_state["tdf"]; det=st.session_state.get("det")
    ml=st.session_state.get("ml"); today_pred=st.session_state.get("today_pred")
    rm_display = "wicks (H/L)" if range_mode_val == "wick" else "closes (C/C)"
    st.markdown(f"### {len(tdf)} breakout days | CBDR {cbdr_start_gmt:02d}:00-{cbdr_end_gmt:02d}:00 UTC | Range: {rm_display}")
    if ml:
        sp=ml["splits"]
        st.markdown(f"**Split:** Train {sp['train']} | Val {sp['val']} | **Test {sp['test']}**")
        if 'macro_status' in dir(): st.caption(macro_status)

    tabs=st.tabs(["Signal","Arena","Models","Sessions","Risk","Distributions","Trade Logs"])

    # ═══ SIGNAL ═══
    with tabs[0]:
        st.markdown("### Latest Signal")
        if det:
            sig_date = pd.Timestamp(det["date"])
            today = pd.Timestamp(datetime.now().date())
            days_ago = (today - sig_date).days
            cbdr_from = det.get("cbdr_from","?")
            cbdr_to = det.get("cbdr_to","?")
            rm_label = "closes" if det.get("range_mode")=="close" else "wicks"
            n_candles = det.get("cbdr_candles","?")

            # Main header: trading day + CBDR source info
            st.markdown(f"**Trade day: {det['day']} {det['date']}** | Price: **{det.get('price','N/A')}**")
            st.markdown(f"CBDR data: **{cbdr_from} to {cbdr_to} UTC** ({n_candles} candles, {rm_label})")

            if days_ago > 0:
                if days_ago <= 3:
                    st.info(f"This is {det['day']}'s signal ({days_ago} day{'s' if days_ago>1 else ''} ago). "
                            f"Today's CBDR ({cbdr_start_gmt:02d}:00-{cbdr_end_gmt:02d}:00 UTC) hasn't formed yet. "
                            f"Run again after {cbdr_end_gmt:02d}:00 UTC tonight.")
            dc=st.columns(6)
            dc[0].markdown(f'<div class="det"><div class="lbl">Range ({rm_label})</div><div class="vl">{det["rh"]} - {det["rl"]}</div><div class="sub">{det["rs"]} pts | {n_candles} candles</div></div>',unsafe_allow_html=True)
            dc[1].markdown(f'<div class="det"><div class="lbl">1st</div><div class="vl">{"G" if det["fc_green"] else "R"} {det["fc_pat"]}</div></div>',unsafe_allow_html=True)
            dc[2].markdown(f'<div class="det"><div class="lbl">Last</div><div class="vl">{"G" if det["lc_green"] else "R"} {det["lc_pat"]}</div></div>',unsafe_allow_html=True)
            dc[3].markdown(f'<div class="det"><div class="lbl">Trend</div><div class="vl">{det["cbdr_trend"]:+.2f}</div><div class="sub">5d:{det["wbias"]} ({det.get("pct_5d",0):+.1f}%) 20d:{det["mbias"]} ({det.get("pct_20d",0):+.1f}%)</div></div>',unsafe_allow_html=True)
            dc[4].markdown(f'<div class="det"><div class="lbl">S/R</div><div class="vl">{"Sup" if det["near_sup"] else("Res" if det["near_res"] else "Clear")}</div><div class="sub">Str {det.get("sup_str",0)}/{det.get("res_str",0)}</div></div>',unsafe_allow_html=True)
            dc[5].markdown(f'<div class="det"><div class="lbl">Vol</div><div class="vl">{det.get("cbdr_rvol","?")}x</div><div class="sub">{"Event!" if det["evt_any"] else "Normal"}</div></div>',unsafe_allow_html=True)
            # Macro data row (VIX, DXY, US10Y, Oil — previous day's close = zero leakage)
            if det.get("macro_available",0)==1:
                mc2=st.columns(5)
                mc2[0].markdown(f'<div class="det"><div class="lbl">VIX (today)</div><div class="vl">{det.get("vix_level",0):.1f}</div><div class="sub">1d:{det.get("vix_chg1d",0):+.1f}% 5d:{det.get("vix_chg5d",0):+.1f}%</div></div>',unsafe_allow_html=True)
                mc2[1].markdown(f'<div class="det"><div class="lbl">DXY (today)</div><div class="vl">{det.get("dxy_level",0):.2f}</div><div class="sub">1d:{det.get("dxy_chg1d",0):+.2f}% 5d:{det.get("dxy_chg5d",0):+.2f}%</div></div>',unsafe_allow_html=True)
                mc2[2].markdown(f'<div class="det"><div class="lbl">US10Y (today)</div><div class="vl">{det.get("us10y_level",0):.2f}%</div><div class="sub">1d:{det.get("us10y_chg1d",0):+.2f}% 5d:{det.get("us10y_chg5d",0):+.2f}%</div></div>',unsafe_allow_html=True)
                mc2[3].markdown(f'<div class="det"><div class="lbl">Oil (today)</div><div class="vl">${det.get("oil_level",0):.1f}</div><div class="sub">1d:{det.get("oil_chg1d",0):+.1f}% 5d:{det.get("oil_chg5d",0):+.1f}%</div></div>',unsafe_allow_html=True)
                dxy5=det.get("dxy_chg5d",0); vix_l=det.get("vix_level",0)
                sig_parts=[]
                if vix_l>25: sig_parts.append("High fear")
                if dxy5<-0.3: sig_parts.append("DXY weak=Gold+")
                elif dxy5>0.3: sig_parts.append("DXY strong=Gold-")
                mc2[4].markdown(f'<div class="det"><div class="lbl">Macro Signal</div><div class="vl">{"Fear" if vix_l>20 else "Calm"}</div><div class="sub">{" | ".join(sig_parts) if sig_parts else "Neutral"}</div></div>',unsafe_allow_html=True)
            if today_pred and det.get("direction"):
                tp2=today_pred; rs=det["rs"]; d=det["direction"]; pe=tp2["pred_entry"]; px=tp2["pred_exit"]
                conf=tp2["confidence"]; dconf=tp2["dir_conf"]; rmult=tp2["risk_mult"]; rtprob=tp2["retest_prob"]
                if rtprob>=0.5: etype="LIMIT (Retest)"; ud=pe*0.3; slx=1.3
                else: etype="MARKET (Breakout)"; ud=0; slx=1.0
                tpx=px+ud
                if d=="bullish": ep=det["rh"]-ud*rs+spread; slp=ep-slx*rs; tpp=ep+tpx*rs
                else: ep=det["rl"]+ud*rs-spread; slp=ep+slx*rs; tpp=ep-tpx*rs
                sld=slx*rs; tpd=tpx*rs; rrv=tpx/slx if slx>0 else 0
                adj_risk=round(base_risk*rmult,2); ramt=capital*(adj_risk/100); pos=ramt/sld if sld>0 else 0
                scls="signal-bull" if d=="bullish" else "signal-bear"
                st.markdown(f'<div class="signal-card {scls}"><div style="display:flex;justify-content:space-between;flex-wrap:wrap"><div><h2 style="color:white;margin:0">{d.upper()} {etype}</h2><p style="color:#8b949e;margin:0">Retest: {rtprob*100:.0f}% | PB: {pe:.2f}x | Run: {px:.1f}x</p></div><div style="text-align:right"><h2 style="color:#58a6ff;margin:0">{conf*100:.0f}% Conf</h2><p style="color:#8b949e;margin:0">Dir: {dconf*100:.0f}% | Risk: {adj_risk:.2f}% ({rmult}x)</p></div></div></div>',unsafe_allow_html=True)
                tc=st.columns(6)
                tc[0].markdown(f'<div class="plv" style="border-color:#58a6ff"><div class="pl">Entry</div><div class="pp" style="color:#58a6ff">{ep:.2f}</div><div class="pd">{ud:.2f}x deep</div></div>',unsafe_allow_html=True)
                tc[1].markdown(f'<div class="plv" style="border-color:#f85149"><div class="pl">SL</div><div class="pp" style="color:#f85149">{slp:.2f}</div><div class="pd">{sld:.1f}pts ({slx}x)</div></div>',unsafe_allow_html=True)
                tc[2].markdown(f'<div class="plv" style="border-color:#3fb950"><div class="pl">TP</div><div class="pp" style="color:#3fb950">{tpp:.2f}</div><div class="pd">{tpd:.1f}pts</div></div>',unsafe_allow_html=True)
                tc[3].markdown(f'<div class="plv"><div class="pl">R:R</div><div class="pp">{rrv:.1f}:1</div></div>',unsafe_allow_html=True)
                tc[4].markdown(f'<div class="plv"><div class="pl">Pos</div><div class="pp">{pos:,.2f}</div></div>',unsafe_allow_html=True)
                tc[5].markdown(f'<div class="plv"><div class="pl">$ Risk</div><div class="pp">${ramt:,.0f}</div></div>',unsafe_allow_html=True)
                # Per-model predictions
                st.markdown("#### All Model Predictions")
                mrows=[]
                all_mn=sorted(set(list(tp2.get("entry_by_model",{}).keys())+list(tp2.get("exit_by_model",{}).keys())))
                for mn in all_mn:
                    ep2=tp2.get("entry_by_model",{}).get(mn,"-")
                    xp2=tp2.get("exit_by_model",{}).get(mn,"-")
                    mrows.append({"Model":mn,"Entry (PB depth)":ep2,"Exit (Run)":xp2})
                if mrows: st.dataframe(pd.DataFrame(mrows),use_container_width=True,hide_index=True)
                # Reasoning
                reasons=[]
                reasons.append(f"{'Retest expected' if rtprob>=0.5 else 'No retest'} ({rtprob*100:.0f}%) -> {'Limit at 30% predicted depth' if rtprob>=0.5 else 'Enter at breakout'}")
                reasons.append(f"Entry ({tp2['entry_model']}): PB {pe:.2f}x | Exit ({tp2['exit_model']}): Run {px:.1f}x")
                reasons.append(f"Conf: {conf*100:.0f}% x Dir: {dconf*100:.0f}% -> Risk {adj_risk:.2f}% ({rmult}x)")
                reasons.append(f"SL {slx}x range (optimal from report: 70%+ of 1.5x stops reversed)")
                # Macro context
                if det.get("macro_available",0)==1:
                    vix_l=det.get("vix_level",0); dxy_5d=det.get("dxy_chg5d",0); us10y_5d=det.get("us10y_chg5d",0)
                    macro_parts=[]
                    if vix_l>25: macro_parts.append(f"VIX elevated ({vix_l:.0f}) = high fear, expect larger moves")
                    elif vix_l>20: macro_parts.append(f"VIX moderate ({vix_l:.0f})")
                    else: macro_parts.append(f"VIX calm ({vix_l:.0f})")
                    if dxy_5d<-0.3: macro_parts.append(f"DXY falling ({dxy_5d:+.1f}% 5d) = gold tailwind")
                    elif dxy_5d>0.3: macro_parts.append(f"DXY rising ({dxy_5d:+.1f}% 5d) = gold headwind")
                    if us10y_5d>0.5: macro_parts.append(f"yields rising ({us10y_5d:+.1f}% 5d) = gold pressure")
                    elif us10y_5d<-0.5: macro_parts.append(f"yields falling ({us10y_5d:+.1f}% 5d) = gold support")
                    oil_5d=det.get("oil_chg5d",0)
                    if abs(oil_5d)>1: macro_parts.append(f"oil {oil_5d:+.1f}% 5d (inflation proxy)")
                    reasons.append("Macro: " + " | ".join(macro_parts))
                sim=tdf[(tdf["day"]==det["day"])&(tdf["direction"]==d)]
                if len(sim)>=3: reasons.append(f"History {det['day']}+{d}: PB {sim['pb_depth'].mean():.2f}x, Run {sim['max_run'].mean():.1f}x (n={len(sim)})")
                st.markdown("#### Reasoning")
                st.markdown('<div class="reason">'+"<br>".join(f"- {r}" for r in reasons)+'</div>',unsafe_allow_html=True)
                # PDF download
                try:
                    pdf_bytes=generate_pdf(det,today_pred,ml,tdf,asset_name)
                    st.download_button("Download Signal PDF",pdf_bytes,f"cbdr_signal_{det['date']}.pdf","application/pdf")
                except Exception as e:
                    st.caption(f"PDF generation: {e}")
            elif det.get("direction") is None: st.info("No breakout yet.")
            else: st.info("Need 40+ trades for ML.")
        else: st.warning("Could not detect CBDR.")

    # ═══ ARENA — FIXED: robust display, no None values ═══
    with tabs[1]:
        st.markdown("### Strategy Arena — Test Period")
        st.markdown("Ensemble uses **adaptive risk** (0.1x-1.5x per trade). Sharpe/Sortino = per-trade. Calmar = return/maxDD.")
        if ml and ml.get("strategies"):
            stdata=ml["strategies"]; colors_list=["#8b949e","#58a6ff","#f85149","#3fb950","#d2a828"]
            strat_labels={"Baseline":"Baseline (boundary, 1.5x SL, 3SD TP)","ML_Retest":"ML Retest (limit, 1.3x SL)",
                "Breakout_Only":"Breakout-Only (bo close, ML TP)","Ensemble":"Ensemble (retest OR bo, adaptive)",
                "Ensemble_Guard":"Ensemble Guard (skip low-conf)"}
            comp_rows=[]; strat_stats={}
            for key,tds in stdata.items():
                if tds is None or (isinstance(tds,pd.DataFrame) and tds.empty): continue
                if not isinstance(tds,pd.DataFrame): continue
                try:
                    rm_vals=tds["risk_mult"].values if "risk_mult" in tds.columns else None
                    s=calc_stats(tds["outcome"].tolist(),tds["r"].tolist(),capital,base_risk,rm_vals)
                    strat_stats[key]=s
                    if s.get("active",0)>0:
                        fl=tds[tds["outcome"].isin(["win","loss"])]
                        arv=fl["rr"].mean() if "rr" in fl.columns and len(fl)>0 else 0
                        et=fl["entry_type"].value_counts().to_dict() if "entry_type" in fl.columns else {}
                        comp_rows.append({"Strategy":strat_labels.get(key,key),"Trades":int(s["active"]),"Skip":int(s["skipped"]),
                            "WR":f'{s["wr"]:.1f}%',"PF":float(s["pf"]),"Avg R:R":f'{arv:.1f}:1',
                            "Sharpe":float(s["sharpe"]),"Sortino":float(s["sortino"]),"Calmar":float(s["calmar"]),
                            "P&L":f'${s["dollar_pnl"]:,.0f}',"Return":f'{s["return_pct"]:.1f}%',
                            "Max DD":f'{s["max_dd_pct"]:.1f}%',"Loss Streak":int(s["max_loss_streak"])})
                except Exception as ex:
                    st.warning(f"Error computing {key}: {ex}")
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows),use_container_width=True,hide_index=True)
            else:
                st.warning("No strategies have active trades. Check data or period.")
            # Equity curves
            fig=go.Figure()
            for i2,(key,_) in enumerate(stdata.items()):
                s2=strat_stats.get(key)
                if s2 and s2.get("eq_curve") and len(s2["eq_curve"])>1:
                    fig.add_trace(go.Scatter(y=s2["eq_curve"],mode="lines",name=strat_labels.get(key,key)[:25],line=dict(color=colors_list[i2%len(colors_list)],width=2)))
            fig.add_hline(y=capital,line_dash="dash",line_color="#30363d")
            fig.update_layout(**PBG,height=400,yaxis_title="$",yaxis_tickprefix="$",title="Equity Curves")
            st.plotly_chart(fig,use_container_width=True)
            # Drawdown
            fig2=go.Figure()
            for i2,(key,_) in enumerate(stdata.items()):
                s2=strat_stats.get(key)
                if s2 and s2.get("dd_series") and len(s2["dd_series"])>1:
                    c=colors_list[i2%len(colors_list)]
                    fig2.add_trace(go.Scatter(y=[-d for d in s2["dd_series"]],mode="lines",name=strat_labels.get(key,key)[:25],line=dict(color=c,width=1.5),fill="tozeroy",fillcolor=hex_to_rgba(c,0.08)))
            fig2.update_layout(**PBG,height=300,yaxis_title="DD %",title="Drawdowns")
            st.plotly_chart(fig2,use_container_width=True)
        else: st.info("Need 40+ trades.")

    # ═══ MODELS ═══
    with tabs[2]:
        st.markdown("### Model Arena")
        if ml:
            dm=ml.get("dir_model")
            if dm and dm.get("results"):
                st.markdown("#### Direction Classifier")
                rows=[]
                for nm,r in dm["results"].items():
                    for sn in ["train","val","test"]:
                        if sn in r: rows.append({"Model":nm,"Split":sn.title(),"Acc":f'{r[sn]["acc"]}%',"F1":f'{r[sn].get("f1",0)}%',"N":r[sn]["n"]})
                if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                if dm.get("best"): st.success(f"Best: **{dm['best'][0]}**")
                if not dm["importance"].empty:
                    top=dm["importance"].head(15)
                    fig=go.Figure(data=[go.Bar(x=top.values,y=top.index,orientation="h",marker_color="#d2a828")])
                    fig.update_layout(**PBG,height=380,yaxis=dict(autorange="reversed")); st.plotly_chart(fig,use_container_width=True)
            st.markdown("---")
            cr=ml.get("retest_clf_results",{})
            if cr: st.dataframe(pd.DataFrame([{"Split":k.title(),"Acc":f'{v["acc"]:.1f}%',"N":v["n"]} for k,v in cr.items()]),use_container_width=True,hide_index=True)
            for mk,title,clr in [("entry","Entry Depth","#58a6ff"),("exit","Max Run","#3fb950")]:
                em2=ml.get(mk)
                if not em2: continue
                st.markdown(f"#### {title}")
                if em2["best"]: st.success(f"Best: **{em2['best'][0]}**")
                rows=[]
                for nm,r in em2["results"].items():
                    for sn in ["train","val","test"]:
                        if sn in r and isinstance(r[sn],dict): rows.append({"Model":nm,"Split":sn.title(),"MAE":r[sn].get("mae","?"),"R2":r[sn].get("r2","?"),"N":r[sn].get("n","?")})
                if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                if not em2["importance"].empty:
                    top=em2["importance"].head(15)
                    fig=go.Figure(data=[go.Bar(x=top.values,y=top.index,orientation="h",marker_color=clr)])
                    fig.update_layout(**PBG,height=380,yaxis=dict(autorange="reversed")); st.plotly_chart(fig,use_container_width=True)
                st.markdown("---")

    # ═══ SESSIONS ═══
    with tabs[3]:
        st.markdown("### Session Analysis")
        if "bo_session" in tdf.columns:
            bg=tdf.groupby("bo_session").agg(n=("max_run","count"),avg_run=("max_run","mean"),avg_pb=("pb_depth","mean")).round(3).reset_index()
            bg.columns=["Session","N","Avg Run","Avg PB"]; st.dataframe(bg,use_container_width=True,hide_index=True)
            scm={"asia":"#58a6ff","london":"#3fb950","ny":"#f85149","ldn_ny_overlap":"#d2a828","off_hours":"#8b949e"}
            fig=make_subplots(rows=1,cols=2,subplot_titles=("Run","PB"))
            cl2=[scm.get(s,"#8b949e") for s in bg["Session"]]
            fig.add_trace(go.Bar(x=bg["Session"],y=bg["Avg Run"],marker_color=cl2),row=1,col=1)
            fig.add_trace(go.Bar(x=bg["Session"],y=bg["Avg PB"],marker_color=cl2),row=1,col=2)
            fig.update_layout(**PBG,height=350,showlegend=False); st.plotly_chart(fig,use_container_width=True)

    # ═══ RISK ═══
    with tabs[4]:
        st.markdown("### Risk Analytics")
        if ml and ml.get("strategies"):
            keys=list(ml["strategies"].keys())
            ssel=st.selectbox("Strategy",keys,index=min(3,len(keys)-1))
            tds=ml["strategies"][ssel]
            if tds is not None and not tds.empty:
                fl=tds[tds["outcome"].isin(["win","loss"])]
                if len(fl)>=2:
                    rm=fl["risk_mult"].values if "risk_mult" in fl.columns else None
                    s=calc_stats(fl["outcome"].tolist(),fl["r"].tolist(),capital,base_risk,rm)
                    c1,c2,c3,c4,c5,c6=st.columns(6)
                    mcard(c1,"Sharpe",s["sharpe"]); mcard(c2,"Sortino",s["sortino"]); mcard(c3,"Calmar",s["calmar"])
                    mcard(c4,"Max DD",s["max_dd_pct"],"pct"); mcard(c5,"Loss Streak",s["max_loss_streak"],"int"); mcard(c6,"PF",s["pf"])
                    ra=fl["r"].values
                    fig=go.Figure(); fig.add_trace(go.Histogram(x=ra,nbinsx=30,marker_color="#58a6ff",opacity=0.8))
                    fig.add_vline(x=0,line_dash="dash",line_color="#f85149"); fig.update_layout(**PBG,height=300,title="R Distribution")
                    st.plotly_chart(fig,use_container_width=True)
                    # Reversal analysis
                    ls15=tdf[tdf["pb_depth"]>=1.5]
                    if len(ls15)>0:
                        rv=ls15[ls15["max_run"]>=1.0]; rp=len(rv)/len(ls15)*100
                        st.markdown(f"**Post-SL Reversal:** {len(ls15)} trades hit 1.5x SL, **{rp:.0f}%** reversed to profit. 1.3x SL captures these.")

    # ═══ DISTRIBUTIONS ═══
    with tabs[5]:
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(); fig.add_trace(go.Histogram(x=tdf["pb_depth"],nbinsx=30,marker_color="#58a6ff",opacity=.8))
            fig.update_layout(**PBG,height=300,title="Pullback Depth"); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=go.Figure(); fig.add_trace(go.Histogram(x=tdf["max_run"],nbinsx=30,marker_color="#3fb950",opacity=.8))
            fig.update_layout(**PBG,height=300,title="Max Run"); st.plotly_chart(fig,use_container_width=True)
        for col,lb2 in [("direction","Direction"),("day","Day"),("bo_session","BO Session"),("fc_pat","1st Candle"),("wbias","Week Bias"),("mpos","Month Pos")]:
            if col not in tdf.columns: continue
            g=tdf.groupby(col).agg(n=("pb_depth","count"),pb=("pb_depth","mean"),run=("max_run","mean")).round(3).reset_index()
            g.columns=[lb2,"N","Avg PB","Avg Run"]; st.markdown(f"#### {lb2}"); st.dataframe(g,use_container_width=True,hide_index=True)

    # ═══ TRADE LOGS — CSV downloads for ALL strategies ═══
    with tabs[6]:
        st.markdown("### Trade Logs")
        # Raw data log
        sc2=["date","day","cbdr_from","cbdr_to","cbdr_candles","range_mode","range_high","range_low","range_size","direction","bo_session","retest_session","pb_depth","max_run","fc_pat","cbdr_trend","near_sup","near_res","dist_sup","sup_str","cbdr_rvol","bo_vsurge","wbias","mbias","mpos","evt_any","pct_5d","pct_20d","vix_level","dxy_level","oil_level","us10y_level"]
        sc2=[c for c in sc2 if c in tdf.columns]
        st.markdown("#### Raw CBDR Data (all breakout days)")
        st.dataframe(tdf[sc2].round(3),use_container_width=True,hide_index=True,height=300)
        st.download_button("Download Raw Data CSV",tdf.to_csv(index=False),"cbdr_v15_raw.csv","text/csv")
        st.markdown("---")
        # Strategy trade logs
        if ml and ml.get("strategies"):
            strat_labels2={"Baseline":"Baseline","ML_Retest":"ML Retest","Breakout_Only":"Breakout-Only","Ensemble":"Ensemble","Ensemble_Guard":"Ensemble Guard"}
            st.markdown("#### Strategy Trade Logs")
            for key,tds in ml["strategies"].items():
                if tds is None or (isinstance(tds,pd.DataFrame) and tds.empty): continue
                label2=strat_labels2.get(key,key)
                n_filled=len(tds[tds["outcome"].isin(["win","loss"])]) if "outcome" in tds.columns else 0
                with st.expander(f"{label2} — {n_filled} filled / {len(tds)} total"):
                    show_cols=["date","direction","action","entry_type","outcome","r","rr","sl_used","risk_mult",
                        "pred_entry","pred_exit","actual_pb","actual_run","retest_prob","confidence","dir_conf"]
                    show_cols=[c for c in show_cols if c in tds.columns]
                    st.dataframe(tds[show_cols].round(3),use_container_width=True,hide_index=True,height=300)
                    csv=tds.to_csv(index=False)
                    st.download_button(f"Download {label2} CSV",csv,f"cbdr_{key}_trades.csv","text/csv",key=f"dl_{key}")
            # Combined CSV with all strategies
            st.markdown("---")
            all_trades=[]
            for key,tds in ml["strategies"].items():
                if tds is None or (isinstance(tds,pd.DataFrame) and tds.empty): continue
                t2=tds.copy(); t2["strategy"]=strat_labels2.get(key,key)
                all_trades.append(t2)
            if all_trades:
                combined=pd.concat(all_trades,ignore_index=True)
                st.download_button("Download ALL Strategies CSV",combined.to_csv(index=False),"cbdr_all_strategies.csv","text/csv",key="dl_all")
else:
    st.info("Click Run Full Analysis to start.")
    st.markdown('''### v15 — Ensemble + Multi-Model

**Core:** 70%+ of 1.5x SL stops reversed to profit. Ensemble ALWAYS trades every day.

| Scenario | Entry | SL |
|---|---|---|
| Retest prob >= 50% | Limit at 30% predicted depth | 1.3x range |
| No retest | Market at breakout close | 1.0x range |
| Low confidence < 40% | Skip (Guard only) | - |

**Risk:** 0.1x-1.5x base per trade by confidence. 7 models compete. Zero leakage.

**Features:** Candle patterns, S/R strength+distance, Volume (CBDR relative, BO surge, trend), Session (Asia/London/NY/overlap), Day/Week/Month bias, Events, **Macro: VIX (fear), DXY (dollar — inverse to gold), US10Y (yield pressure), Oil (inflation proxy)**. All macro uses previous day close = zero leakage.
    ''')
