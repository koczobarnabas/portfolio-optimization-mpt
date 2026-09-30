import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
import datetime

# 1. Page settings
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Interactive Portfolio Optimization & Backtest")

st.error("**Disclaimer:** This is a student project. Use for your investments only at your own discretion.")

# 2. Tutorial
with st.expander("How to use"):
    st.markdown("""
    * **Input:** Type your tickers on the left side separated by commas (yahoofinance.com)** 
    * **Optimization:** Minimum volatility (blue) and the Tangency Portfolio (red) is calculated based on the Markowitz Modern Portfolio Theory.
    * **Backtest:** You can see how your portfolio would have performed in the specified period (you can specify custom periods aswell) if you put $1 into the portfolio.
    """)

# 3. Inputs
st.sidebar.header("Parameters")
ticker_input = st.sidebar.text_input("Tickers (separated by commas)", "AAPL, MSFT, JNJ, GLD")
tickers = [t.strip().upper() for t in ticker_input.split(',')]

benchmark_ticker = st.sidebar.text_input("Benchmark", "SPY").strip().upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Time Period")

# Pre-defined time periods
period = st.sidebar.selectbox(
    "Select one:", 
    ["1 Month (1M)", "6 Months (6M)", "YTD", "1 Year (1Y)", "5 Years (5Y)", "From 2000 (Max)", "Custom date"], 
    index=4  # 5Y is automatic
)

today = datetime.date.today()

# Date calc
if period == "1 Month (1M)":
    start_date = today - datetime.timedelta(days=30)
    end_date = today
elif period == "6 Months (6M)":
    start_date = today - datetime.timedelta(days=180)
    end_date = today
elif period == "YTD":
    start_date = datetime.date(today.year, 1, 1)
    end_date = today
elif period == "1 Year (1Y)":
    start_date = today - datetime.timedelta(days=365)
    end_date = today
elif period == "5 Years (5Y)":
    start_date = today - datetime.timedelta(days=5*365)
    end_date = today
elif period == "From 2000 (Max)":
    start_date = datetime.date(2000, 1, 1) # Visszamenőleg 2000-ig
    end_date = today
else:
    # Calendar
    start_date = st.sidebar.date_input("Start date", today - datetime.timedelta(days=5*365))
    end_date = st.sidebar.date_input("End date", today)

# 4. Functions
@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    end_yf = end + datetime.timedelta(days=1)
    data = yf.download(tickers, start=start, end=end_yf, threads=False, progress=False, auto_adjust=True)['Close']
    
    # Series to df
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        
    returns = data.pct_change().dropna()
    return data, returns 

def portfolio_stats(weights, mean_returns, cov_matrix):
    port_return = np.sum(weights * mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = port_return / port_volatility
    return port_return, port_volatility, sharpe_ratio

def min_func_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_stats(weights, mean_returns, cov_matrix)[2]

def min_func_variance(weights, mean_returns, cov_matrix):
    return portfolio_stats(weights, mean_returns, cov_matrix)[1]

# Calc
if len(tickers) < 2:
    st.error("Specify atleast 2 tickers for optimization!")
else:
    with st.spinner("Downloading data and optimizing..."):
        prices, returns = load_data(tickers, start_date, end_date)

        if returns.empty:
            st.error("Error: Yahoo Finance returned no data, please try another asset or date.")
            st.stop()
        
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(tickers)

        # Constraints here
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]

        # Max Sharpe
        opt_sharpe = sco.minimize(min_func_sharpe, init_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        sharpe_weights = opt_sharpe['x']
        sharpe_stats = portfolio_stats(sharpe_weights, mean_returns, cov_matrix)

        # GMV
        opt_var = sco.minimize(min_func_variance, init_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        var_weights = opt_var['x']
        var_stats = portfolio_stats(var_weights, mean_returns, cov_matrix)

        # 0% weight
        has_zero = any(w < 0.0001 for w in sharpe_weights) or any(w < 0.0001 for w in var_weights)
        
        if has_zero:
            st.markdown(
                "Why is it showing 0 percent?", 
                help="The Markowitz Model excluded that asset because the portfolio performs the best without it."
            )

        # 5. Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tangency Portfolio (Max Sharpe)")
            sharpe_df = pd.DataFrame({"Asset": tickers, "Weight": sharpe_weights})
            sharpe_df['Weight'] = (sharpe_df['Weight'] * 100).round(2).astype(str) + '%'
            st.dataframe(sharpe_df, hide_index=True)
            st.write(f"**Expected return:** {sharpe_stats[0]*100:.2f}% | **Risk:** {sharpe_stats[1]*100:.2f}% | **Sharpe:** {sharpe_stats[2]:.2f}")

        with col2:
            st.subheader("Minimum Volatility Portfolio")
            var_df = pd.DataFrame({"Asset": tickers, "Weight": var_weights})
            var_df['Weight'] = (var_df['Weight'] * 100).round(2).astype(str) + '%'
            st.dataframe(var_df, hide_index=True)
            st.write(f"**Expected return:** {var_stats[0]*100:.2f}% | **Risk:** {var_stats[1]*100:.2f}% | **Sharpe:** {var_stats[2]:.2f}")

        # Math
        with st.expander("How does the calculation work?"):
            st.markdown("""
            The system uses **Markowitz's Modern Portfolio Theory**. By diversifying our portfolio we can eliminate "asset-specific" risks and achieve a portfolio with less overall risk if we allocate them smartly.

            * **Max Sharpe (Tangency Portfolio):** It calculates the exact allocation where you get the highest possible return for every unit of risk taken.
            * **Minimum Volatility:** Here, the algorithm doesn't care about maximizing profit. It exclusively looks for the weighting that results in the lowest price fluctuation (risk).
            """)

        st.markdown("---")

        # Monte Carlo
        st.subheader("Monte Carlo Simulation")
        
        num_portfolios = 10000
        mc_results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            w = np.random.random(num_assets)
            w /= np.sum(w)
            p_ret, p_vol, p_sr = portfolio_stats(w, mean_returns, cov_matrix)
            mc_results[0,i] = p_vol
            mc_results[1,i] = p_ret
            mc_results[2,i] = p_sr

        from scipy.optimize import minimize
        
        target_returns = np.linspace(var_stats[0], max(mean_returns), 50)
        frontier_volatility = []
        
        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.sum(w * mean_returns) - target}
            )
            
            res = minimize(
                lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                len(tickers) * [1. / len(tickers)],
                method='SLSQP',
                bounds=tuple((0, 1) for _ in tickers),
                constraints=constraints
            )
            frontier_volatility.append(res.fun)

        fig_ef = go.Figure()

        # Monte Carlo visualization
        fig_ef.add_trace(go.Scatter(
            x=mc_results[0,:], 
            y=mc_results[1,:], 
            mode='markers', 
            marker=dict(
                color=mc_results[2,:], 
                colorscale='Viridis', 
                showscale=True, 
                size=5, 
                opacity=0.5, 
                colorbar=dict(title='Sharpe-re', x=1.02)
            ),
            name='Random portfolios', 
            hoverinfo='none',
            showlegend=False  # Hide this cuz I don't like it
        ))

        # EF curve
        fig_ef.add_trace(go.Scatter(
            x=frontier_volatility, 
            y=target_returns, 
            mode='lines', 
            line=dict(color='white', width=3, dash='dash'), 
            name='Efficient Frontier',
            showlegend=True
        ))
        
        # Max Sharpe
        fig_ef.add_trace(go.Scatter(x=[sharpe_stats[1]], y=[sharpe_stats[0]], mode='markers', 
                                    marker=dict(color='#ff4b4b', size=16, line=dict(color='white', width=2), symbol='circle'), 
                                    name='Max Sharpe'))
        
        # GMV
        fig_ef.add_trace(go.Scatter(x=[var_stats[1]], y=[var_stats[0]], mode='markers', 
                                    marker=dict(color='#0068c9', size=16, line=dict(color='white', width=2), symbol='circle'), 
                                    name='Min Volatility'))
        
        fig_ef.update_layout(
            xaxis_title='Annual Standard Deviation (Risk)', 
            yaxis_title='Annual Expected Return', 
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        # Monte Carlo explanation
        with st.expander("What does the Monte Carlo Simulation show?"):
            st.markdown("""
            This chart visualizes thousands of randomly generated portfolio combinations using your selected assets. 
            
            * **The Dots:** Each dot represents a unique mix (weighting) of your assets. Its position shows the expected return (vertical axis) versus the risk involved (horizontal axis, standard deviation).
            * **The Color Scale:** The color indicates the Sharpe Ratio. Brighter colors mean a better risk-adjusted return.
            * **The Efficient Frontier:** The outer upper edge of this cloud is called the "Efficient Frontier." It represents the absolute best possible return you can get for any given level of risk. You always want your portfolio to sit on this edge, which is exactly where our Max Sharpe and Minimum Volatility portfolios are located.
            """)

        st.markdown("---")
        
       # 7. Backtest
        st.subheader("Backtest")
        st.write("The backtest shows how your portfolio would have performed if you invested $1, compared to the benchmark.")
        
        # Daily returns
        returns['Max Sharpe'] = returns[tickers].dot(sharpe_weights)
        returns['Min Volatility'] = returns[tickers].dot(var_weights)
        returns['Equal Weight'] = returns[tickers].dot(init_guess)

        cumulative_returns = (1 + returns[['Max Sharpe', 'Min Volatility', 'Equal Weight']]).cumprod()

        # Timezone stuff
        if cumulative_returns.index.tz is not None:
            cumulative_returns.index = cumulative_returns.index.tz_localize(None)

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['Max Sharpe'], mode='lines', name='Max Sharpe',
                                    line=dict(color='red')))
        fig_bt.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['Min Volatility'], mode='lines', name='Min Volatility',
                                    line=dict(color='blue')))
        fig_bt.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['Equal Weight'], mode='lines', name='Equal Weight',
                                    line=dict(color='gray', dash='dash')))
        
# Benchmark
        try:
            end_date_yf = end_date + datetime.timedelta(days=1)
            bench_data = yf.download(benchmark_ticker, start=start_date, end=end_date_yf, threads=False, progress=False)['Close']
            
            if isinstance(bench_data, pd.DataFrame):
                bench_data = bench_data.squeeze()
                
            bench_returns = bench_data.pct_change().dropna()
            bench_cumulative = (1 + bench_returns).cumprod()
            
            asset_prices_md = "**Calculation is based on the following opening & closing prices:**\n"
            
            for col in prices.columns:
                start_p = prices[col].dropna().iloc[0]
                end_p = prices[col].dropna().iloc[-1]
                asset_prices_md += f"* **{col}**: ${start_p:.2f} ➔ ${end_p:.2f}\n"
            
            bench_start = bench_data.iloc[0]
            bench_end = bench_data.iloc[-1]
            asset_prices_md += f"* **{benchmark_ticker} (Benchmark)**: ${bench_start:.2f} ➔ ${bench_end:.2f}"
            
            st.info(asset_prices_md)
            
            if bench_cumulative.index.tz is not None:
                bench_cumulative.index = bench_cumulative.index.tz_localize(None)
                
            fig_bt.add_trace(go.Scatter(x=bench_cumulative.index, y=bench_cumulative, mode='lines', name=f'Benchmark ({benchmark_ticker})',
                                        line=dict(color='white', width=2, dash='dot')))
        except Exception as e:
            st.warning(f"Failed to load benchmark ({benchmark_ticker}): {e}")
        
        fig_bt.update_layout(yaxis_title="Cumulative value (1= Starting capital)", xaxis_title="Date", hovermode="x unified")
        st.plotly_chart(fig_bt, use_container_width=True)

        # Footer
st.markdown(
    """
    <div style="text-align: right; font-size: 12px; color: gray; margin-top: 50px;">
        Created by: <b>Koczó Barnabás</b>
    </div>
    """, 
    unsafe_allow_html=True
)

# Created by: Koczó Barnabás. Gemini AI was used here and there
