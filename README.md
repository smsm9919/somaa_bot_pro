# DOGE/USDT Trading Bot — Render 24/7 (Pro)
- 15m strategy: EMA200 trend filter + EMA20/EMA50 momentum + RSI band (52–72 for longs / 28–48 for shorts) + ADX≥25 + Supertrend direction + pullback entry (re-cross EMA20).
- Risk: TP=1.2×ATR, SL=0.8×ATR, ATR-based breakeven after +1×ATR, trailing SL by ATR. Cooldown 10m, spike filter, low-range filter, daily loss cap.
- Position size = 60% من الرصيد × الرافعة.
- Healthcheck `/healthz` + `PORT` env + Dockerfile EXPOSE 8080.
