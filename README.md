# Portfolio Suggestion Dashboard

A decision-support tool that analyzes an existing set of holdings and ranks
candidate additions using several classical portfolio-construction methods.

> **Not investment advice.** This describes what would have been efficient given
> historical data and explicit assumptions. It does not forecast returns and does
> not account for taxes, costs, liquidity, or your personal circumstances.

## Quick start

Needs **Python 3.10+**.

**First time only** — create the virtual environment and install dependencies:

```bash
git clone https://github.com/scarryskeleton/Investment.git
cd Investment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Every time** — activate the environment, then launch:

```bash
cd Investment
source .venv/bin/activate
streamlit run app.py
```

The dashboard opens automatically at <http://localhost:8501>. Press `Ctrl+C` in
the terminal to stop it.

If you get `command not found: streamlit`, the environment isn't active — re-run
`source .venv/bin/activate` (your shell prompt should start with `(.venv)`).

## Currency

The sidebar has a **Currency** picker (26 options, EUR by default) that applies
across every mode. Prices come back from Yahoo Finance in each security's own
listing currency (a US stock in USD, a Paris listing in EUR, a London one in
pence); the app converts them into your chosen currency using **historical**
EUR-cross exchange rates (via Yahoo's `BASEQUOTE=X` pairs), so a currency's
moves are part of your returns, not hidden — the same way they'd hit a real
account. Explore's two-fund mix, Analyze's portfolio value and weights, and
the whole Practice portfolio all convert this way; Research shows a security's
own figures (market cap, EPS, etc.) in *that security's* currency, since that's
the currency they're actually reported in.

## Four modes

The sidebar has a **Mode** switch:

### 🎮 Practice portfolio (paper trading)

Start with fake cash (default 10,000) and **buy / sell any real ticker at its
latest close**. Everything is derived from the trade log:

- **Headline** — total value, cash, amount in the market, total P&L (amount
  and %), and what the same starting cash would be worth if you'd just bought
  the S&P 500
- **Trade ticket** — ticker, buy/sell, size in shares or your currency; shows
  the security's country, its native price and currency, and the fee on the
  trade; validates that you have the cash (incl. fee) / the shares — plus a
  **price-movement chart** for the ticker (hourly bars over the last few days
  when Yahoo has them, daily otherwise) so you can see how it's been moving
  before you trade
- **🏆 Leaderboard** — always on screen, right under your own numbers. Every
  profile's practice account, ranked by return on its own starting cash
  (currency-neutral, so it's a fair fight across accounts in different
  currencies), with value, start date and trade count. Visible to everyone —
  that's the point of a shared, competitive sandbox; it just says so plainly
  until a second person has traded
- **Trading costs** — a switchable cost model (none, a few broker presets, or
  custom): a flat charge and/or commission per trade, plus an FX conversion fee
  on securities not quoted in your account currency. Buy fees go into cost
  basis; sell fees come off the proceeds; both show in the equity curve
- **Holdings** — shares, average cost, current price, value, unrealized P&L,
  weight, plus business market (sector), country of origin and currency
- **Price movement** — pick any holding to see its recent hourly (or daily
  fallback) chart
- **Portfolio overview** — where your money sits by business market (with the
  tickers in each), by country, and by currency, plus the share of it exposed
  to foreign-currency (FX) risk
- **Equity curve** — your account value day-by-day (a real reconstruction from
  your trades against historical prices, each leg FX-converted at that day's
  rate) next to "all-in S&P 500"
- **📝 Trade journal** — every trade (with fees, in your currency) in a
  read-only overview table with a one-line note preview. Pick any trade below
  it to open a full-width, multi-line editor — room to actually write a
  thesis, not a spreadsheet cell — plus ±45 days of that ticker's own price
  history with a marker at exactly where you traded, so you can judge the
  entry or exit with hindsight. Notes are visible to everyone, editable only
  with the profile's password. Also where undo-last and reset live.

Saved per profile, so you can trade over weeks and watch how your picks do —
and see how everyone else's picks are doing too. Every price is converted
from the security's native quote to your account currency at historical
exchange rates — the FX fee is on top of that, standing in for the real cost
of a broker converting currency for you. Spreads, slippage, dividends and
taxes are still not modelled — a learning sandbox, not a broker.

### 🔎 Research a stock or bond (search or browse, focus one)

**Search by name** — type "Apple", "Coca-Cola", "Novo Nordisk" and pick from the
matches (handles foreign listings too). Or pick a **list** to scan — *Stocks &
bonds (curated ETFs)*, *Popular stocks*, *S&P 500*, or your own tickers — for a
sortable table: 1-year return,
return vs the benchmark, volatility, max drawdown, distance from the 12-month
high, P/E, yield, beta. Click any row (or type a ticker in *jump straight to*)
to focus that name, and the full write-up appears below:

- **The business** — the company/fund description, sector, industry, country,
  headcount, website (from Yahoo Finance)
- **Key stats** — market cap, trailing & forward P/E, price/book, dividend
  yield, profit margin, revenue growth, ROE, beta, 52-week range, analyst
  target and consensus. Add up to two more tickers to compare side by side.
- **In plain language** — the stats turned into sentences ("P/E of 36, well
  above the ~20 market average", "bond fund — price moves opposite to rates")
- **What's expected next** — the next earnings date, and analyst-consensus
  revenue & EPS (with growth vs a year ago, low–high range, analyst count) for
  the coming quarter and fiscal year, plus whether estimates are being revised
  up or down. Flagged as consensus estimates, not company guidance.
- **The company's own plans — straight from the source** — direct links to the
  investor-relations site, the latest annual report (10-K / 20-F) and quarterly
  filing, the next earnings-call date, and recent (ticker-filtered) headlines.
  The dashboard can't read management's strategy for you; this is where it's
  written down.
- **Against \<benchmark\>** — total return vs the benchmark, excess return,
  beta, correlation, up/down-market capture, a rebased price chart and a
  relative-strength line
- **Where it stands** — trend vs the 200-day average and a 1-year scenario
  range, both flagged as *not a forecast*

### 🌱 Explore from cash (for beginners, no holdings needed)
- **Risk slider** — one control for % stocks vs % bonds; historical return,
  volatility, worst drop and worst 12-month stretch update live, with a
  plain-language read-out and a risk/return trade-off chart
- **Growth of 10,000** (in your account currency) for your mix vs 100% stocks /
  100% bonds / cash — the two funds are converted to that currency at
  historical rates first, so FX moves show up in the numbers
- **Monthly savings simulator** — "X/month for Y years" as a bad / typical /
  good outcome band, 1,000 paths. Two engines you can toggle: *resample real
  history* (block bootstrap, the trustworthy default) or an **experimental
  neural generator** — a tiny NumPy mixture-density network ([pa/neural.py](pa/neural.py))
  that learns volatility clustering and fat tails from ~200 months, shown with
  an honest read-out of how faithful it is
- **Fee comparison** — the same plan at 0.15%/yr vs 1.5%/yr, in euros lost

### 📈 Analyze a portfolio

The **Action plan** tab is where the analysis becomes something you can act on
(no forecasting — just arithmetic on a target mix you choose):
- your portfolio's euro value, and how far each holding has **drifted** from
  target (equal weight, risk parity, an optimizer's output, or your own numbers)
- **Option A** — the buy/sell trades to rebalance now, with a note on the
  capital-gains cost of selling winners
- **Option B** — if you're adding new money, how to split it across the
  underweight holdings to drift back toward target *without selling*
- a rebalancing **band** (the standard 5-point "5/25 rule") so you only trade
  when it's worth it

The full optimization workflow below. **No sign-in needed** — fill the holdings
table and press Analyze. A **Beginner mode** checkbox (on by default) hides the
advanced optimizer knobs. Every metric has a `?` tooltip, and the Overview tab
opens with a "What this means" paragraph interpreting your own numbers. Saving
portfolios to a named profile is optional (the *💾 Save & load* panel).

The **Outlook** tab shows a per-asset *range* for watchlist tickers and market
anchors — three textbook expected-return estimates side by side (they disagree,
which is the point) plus a distribution of what-if outcomes over a chosen
horizon. Framed hard as "not a prediction": individual stocks and bonds can't
be reliably forecast, and the tab is built to show that, not hide it.

The **Timing check** tab is the honest answer to "when should I invest?":
- **trend context** — price vs its 50/200-day averages, drawdown from peak, and
  a plain-language status line
- **buy/sell markers + projection cone** — every golden/death cross the
  50/200-day rule would have fired, plotted on the chart, with the current
  signal state; plus a resampled cone of where the price could go 0–36 months
  out (a widening band, explicitly *not* a forecast)
- **200-day rule backtest** — the classic "hold above the 200-day average, sell
  below" rule vs buy-and-hold: historically smoother (smaller drawdowns) but
  *lower* returns, with dozens of trades and lots of missed upside
- **buy-signal track record** — the forward 3/6/12-month return after every time
  the rule flashed "buy", shown next to the same horizons from a random day
  (for SPY: 12-month median +16.7% after a signal vs +16.4% after any day — the
  signal barely moves the needle)
- **invest-now vs wait-for-a-dip** — over historical windows, how often
  investing immediately beat sitting in cash waiting for a 5–30% dip (usually),
  and how often the dip never came

The **💡 Idea worksheet** tab scaffolds a single-stock brief (built for an
*Investment Idea Generation* style assignment). Enter a ticker and a Long/Short
stance and it auto-fills the quantitative context — sector, beta, volatility,
drawdown history, trend vs the 200-day average, distance from the 12-month high,
where the textbook return estimates disagree, and a scenario range — then turns
those numbers into **risk flags** and **research questions**. You write the
pitch, the catalyst and the drivers (the tool has no fundamentals or news); it
exports the whole thing as a filled-in Markdown template you can download.

## What it does

1. **Prices your holdings** and measures annualized return/volatility, Sharpe,
   Sortino, beta, max drawdown, sector exposure, and concentration (HHI /
   effective number of names).
2. **Scans a candidate universe** — your watchlist plus a curated set of ~35
   broad ETFs (sectors, factors, international, bonds, gold, REITs), and
   optionally the full S&P 500 — scoring each name on how it would *complement*
   what you already hold: correlation to your portfolio, volatility impact,
   Sharpe impact, and change in diversification ratio, combined into a single
   0–100 **fit score**.
3. **Runs four optimizers** over your holdings + the candidates you choose, and
   shows suggested target weights next to your current ones:
   - **Mean-Variance (MPT)** — Markowitz efficient frontier via PyPortfolioOpt,
     with Ledoit-Wolf covariance shrinkage and a blended (CAPM + EMA) expected
     return to dampen estimation noise. Optional floor to keep existing
     positions.
   - **Risk Parity** — equal risk contribution from each holding (SLSQP).
   - **Hierarchical Risk Parity** — clustering-based, no return estimates.
   - **Black-Litterman** — starts from market-cap-implied equilibrium returns,
     optionally blended with your own views.

## Profiles & saved portfolios

The dashboard has a lightweight **account layer** — the data lives only on
this machine, in `userdata/portfolios.db`, which is git-ignored:

- **Profile** — pick or create a named profile in the sidebar. Each profile is
  its own namespace of saved portfolios.
- **Editable holdings grid** — edit tickers and amounts inline, add/remove rows.
  "Import from text / CSV" fills the grid from a pasted block.
- **Save / Load / Copy / Delete** — store the grid (plus its amount mode and
  watchlist) under a name; load it back later; duplicate or remove it.
- The active grid is what **Analyze** runs on.

Delete a profile from the `⋯` button next to its name (removes all its
portfolios). To wipe everything, delete `userdata/portfolios.db` (or, on
Turso, drop the tables — see below).

**Practice-portfolio passwords.** Anyone can view any profile's practice
account (pick it from the **Profile** dropdown in 🎮 Practice mode) — there's
no login. A profile can optionally set a password (when it's created, or later
from the sidebar's **🔓 Add a password**), which is then required to place a
trade, undo, or reset *that* profile — but never to view it. Passwords are
stored as a salted SHA-256 hash, not plaintext, and unlocking lasts for the
browser session. A profile with no password stays fully open, as before.

## Using the dashboard

See **[Quick start](#quick-start)** above for the run commands. Once it's open:

- Pick a **Mode** at the top of the sidebar — *🌱 Explore from cash*,
  *🔎 Research a stock or bond*, or *📈 Analyze a portfolio*.
- In Analyze mode, fill the **holdings table** (type tickers + amounts, or use
  *Paste a list instead*), choose whether the amounts are shares / dollars /
  weights, add any tickers to the watchlist, then press **📊 Analyze my
  portfolio**. No sign-in required.
- Results appear across the tabs: Portfolio overview, Diversification scan,
  Optimizers, Action plan, Outlook, Timing check, 💡 Idea worksheet,
  Suggestion summary, 📖 Learn.

## Share it online (Streamlit Community Cloud — free)

1. Put the project on **GitHub** (a public or private repo — both work).
2. Go to <https://share.streamlit.io>, sign in with GitHub, **New app**, pick the
   repo / branch / `app.py`. Under *Advanced settings* choose **Python 3.12**.
3. **Deploy** → you get a URL like `your-app.streamlit.app` to send to peers.

Caveats to tell them:

- **Viewing is always open.** Anyone with the link can open any profile by
  name. An optional per-profile password only gates *changes* (trade, undo,
  reset) in 🎮 Practice mode — see [Profiles & saved portfolios](#profiles--saved-portfolios).
- The app **sleeps after inactivity**; the first visit then takes ~30s to wake.
- Yahoo Finance occasionally rate-limits shared cloud IPs — a reload usually fixes it.

### Durable storage (Turso)

By default `pa/store.py` uses a local SQLite file, which Streamlit Cloud wipes
on every redeploy and when the app wakes from sleep — fine solo, not once
other people's practice-trading history is riding on it. Point it at a free
[Turso](https://turso.tech) database instead and it survives both:

1. Sign up at turso.tech (GitHub login, no card) and create a database.
2. From its dashboard, grab the **Database URL** (`libsql://...`) and generate
   an **Auth Token**.
3. On Streamlit Cloud: your app → **⋮ → Settings → Secrets** →
   ```toml
   TURSO_DATABASE_URL = "libsql://your-db-name.turso.io"
   TURSO_AUTH_TOKEN = "your-token-here"
   ```
   Reboot the app. `pa/store.py` detects these automatically and switches
   backends — no code change needed. Without them (e.g. running locally) it
   keeps using the local SQLite file.
4. For local development against the same Turso database instead of a local
   file, put the same two keys in `.streamlit/secrets.toml` (already
   git-ignored) or export them as environment variables.

Turso *is* SQLite (the same engine, hosted), so the schema and every query in
`pa/store.py` are unchanged — only the connection differs.

Other hosts that work the same way: **Hugging Face Spaces** (Streamlit SDK),
**Render**, **Railway**. For a quick live demo from your own machine instead,
run `streamlit run app.py` and expose it with `cloudflared tunnel` or `ngrok`.

## Run from the command line

For a quick text-only analysis without the dashboard:

```bash
cd Investment
source .venv/bin/activate
python cli.py --holdings sample_portfolio.csv --watchlist "COST,JNJ,XOM,BRK-B" --top 15
```

## Layout

| File | Purpose |
| --- | --- |
| `app.py` | Entry point — page config, sidebar (mode switch, currency), and the 📈 Analyze mode (not yet split out) |
| `ui/common.py` | Shared formatting, currency state, and cached data helpers used by 2+ modes |
| `ui/explore.py` | 🌱 Explore from cash |
| `ui/research.py` | 🔎 Research a stock or bond |
| `ui/practice.py` | 🎮 Practice portfolio — trading, leaderboard, trade journal |
| `tests/` | Offline unit tests (`python -m unittest discover -s tests`) for the pure-logic modules — `paper.py`, `fx.py`, `portfolio.py`, `store.py`. No network calls, runs in well under a second |
| `cli.py` | Terminal version of the analysis |
| `pa/explore.py` | 'Explore from cash' — risk-slider mix + contribution simulator |
| `pa/neural.py` | Tiny NumPy mixture-density network — the experimental path generator |
| `pa/forecast.py` | Per-asset expected-return estimates + scenario ranges (Outlook tab) |
| `pa/timing.py` | Trend context, 200-day-rule backtest, invest-now-vs-wait study |
| `pa/plan.py` | Drift vs target, rebalancing trades, contribution steering (Action plan tab) |
| `pa/ideacard.py` | Single-stock idea worksheet — auto context, risk flags, template export |
| `pa/research.py` | Research mode — business profile, key stats, vs-benchmark stats, plain-language insights |
| `pa/paper.py` | Practice-portfolio simulator — replay trade log → holdings, P&L, equity curve |
| `pa/education.py` | Glossary, method notes, plain-language result interpretation |
| `pa/store.py` | Persistence for profiles, saved portfolios, practice accounts — local SQLite, or Turso when configured |
| `pa/data.py` | yfinance price + fundamentals access, disk-cached |
| `pa/fx.py` | Currency conversion — spot and historical FX rates via Yahoo, reused for the account-currency setting everywhere |
| `pa/universe.py` | Curated ETF list and S&P 500 constituents |
| `pa/portfolio.py` | Parse holdings input → normalized weights |
| `pa/metrics.py` | Return/risk/exposure metrics |
| `pa/optimize.py` | MPT, risk parity, HRP, Black-Litterman |
| `pa/diversification.py` | Correlation-based candidate scoring + rationale |
| `pa/suggest.py` | Orchestration |

## Learn tab

New to investing? The **📖 Learn** tab (also shown on the start screen as
"New to investing? Start here") has:

- a plain-language "starting from cash" primer — the building blocks, why broad
  index funds come up so often for beginners, the knobs that matter, what to be
  wary of, and free non-commercial resources
- a searchable glossary of every metric on the dashboard plus general terms
- what each optimizer assumes and where it breaks down

Every metric tile also has a `?` tooltip with a one-line explanation. All of it
is general education — the tool is not a financial adviser and can't tell you
what to buy.

## Known limitations

- **Expected-return estimation is the weak link.** MPT output is sensitive to it;
  risk parity and HRP sidestep it entirely. When methods agree on a name, that
  agreement is worth more than any single number.
- Correlations are measured in-sample and rise toward 1 in crises.
- yfinance data is free and occasionally patchy; failed tickers are dropped with
  a warning. Sector info is best-effort and cached for a week.
- No transaction costs, taxes, position limits, or borrowing.
