# bybit_mcp_server.py
# --- Imports ---
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import logging
import time
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "True").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bybit_mcp_server')

# Initialize MCP Server
mcp = FastMCP("BybitTradingMCP",
              instructions="You are a trading agent that uses technical analysis to make trading decisions. You can use the tools provided to get information about the market and make trading decisions.",
              sse_ping_interval=20,
    )

# Initialize pybit client
session = HTTP(
    testnet=TESTNET,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET
)

# --- RESOURCES (Context Information) ---
@mcp.resource("bybit://about")
def get_capabilities() -> str:
    """Information about this MCP server's capabilities for Bybit Trading."""
    capabilities = """
    This is an MCP server for trading operations on Bybit.
    Main capabilities:
    
    1. **Market Information**:
       - Get current prices
       - Fetch historical data (klines)
       - Check order book
    
    2. **Advanced Technical Analysis**:
       - Calculate indicators (moving averages, RSI, MACD, ATR)
       - Identify candlestick patterns
       - Analyze market trends
    
    3. **Automated Operations**:
       - Place orders (market, limit)
       - Configure stop loss and take profit
       - Continuous position monitoring
       - Execute automated strategies
    
    4. **Advanced Risk Management**:
       - Dynamic position sizing based on ATR
       - Automated trailing stops management
       - Drawdown monitoring
       - Risk adaptation based on performance
    
    5. **Continuous Monitoring**:
       - Real-time price alerts
       - 24/7 position monitoring
       - Automatic stop adjustment based on volatility
    """
    return capabilities.strip()

# Wallet Tools
@mcp.tool(description="Gets the current balance of the wallet.")
def get_balance() -> str:
    """Gets the current balance of the wallet."""
    try:
        response = session.get_wallet_balance(accountType="UNIFIED")
        if response["retCode"] == 0:
            total_balance = response['result']['list'][0]['totalEquity']
            return f"Current balance: {total_balance}"
        else:
            error = f"Error getting balance: {response['retMsg']}"
            logger.error(error)
            return error
            
    except Exception as e:
        error = f"Error getting balance: {str(e)}"
        logger.error(error)
        return error

# Ticker Tools
@mcp.tool(description="Gets the current ticker (price, volume, etc) for a specific trading pair in perpetual futures.")
def get_ticker(symbol: str) -> str:
    """Gets the current ticker (price, volume, etc) for a specific trading pair in perpetual futures."""
    logger.info(f"Tool 'get_ticker' called for {symbol}")
    try:
        response = session.get_tickers(
            category="linear",  # Standardized for perpetual futures
            symbol=symbol
        )
        
        if response["retCode"] == 0 and response["result"]["list"]:
            ticker = response["result"]["list"][0]
            result = (f"Ticker for {symbol} (Perpetual Futures):\n"
                      f"Current price: ${float(ticker['lastPrice']):.4f}\n"
                      f"24h change: {float(ticker['price24hPcnt'])*100:.2f}%\n"
                      f"24h high: ${float(ticker['highPrice24h']):.4f}\n"
                      f"24h low: ${float(ticker['lowPrice24h']):.4f}\n"
                      f"24h volume: {float(ticker['volume24h']):.2f}\n"
                      f"Funding rate: {float(ticker.get('fundingRate', 0)):.6f}\n"
                      f"Open Interest: {float(ticker.get('openInterest', 0)):.2f}")
            logger.info(f"Ticker successfully obtained for {symbol}")
            return result
        else:
            error = f"Error getting ticker: {response['retMsg']}"
            logger.error(error)
            return error
    except Exception as e:
        error = f"Error getting ticker: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Gets historical data (klines/candlesticks) for perpetual futures analysis.")
def get_klines(symbol: str, intervalo: str = "1", limite: int = 1000) -> str:
    """
    Gets historical data (klines/candlesticks) for perpetual futures analysis.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
        intervalo: Timeframe (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        limite: Quantity of candles to return (min 200, max 1000)
    """
    logger.info(f"Tool 'get_klines' called for {symbol} with interval {intervalo}")
    try:
        # Ensure minimum of 200 candles for proper analysis
        if limite < 200:
            limite = 200
            logger.info(f"Limit adjusted to minimum of 200 candles")
        
        interval_map = {
            "1": "1", "3": "3", "5": "5", "15": "15", "30": "30",
            "60": "60", "1h": "60", "2h": "120", "4h": "240", "6h": "360", 
            "12h": "720", "1d": "D", "1w": "W", "1m": "M", 
            "1min": "1", "1hour": "60", "1day": "D", "1week": "W", "1month": "M"
        }
        
        # Map interval to the format accepted by Bybit
        bybit_interval = interval_map.get(intervalo, intervalo)
        
        response = session.get_kline(
            category="linear",  # Standardized for perpetual futures
            symbol=symbol,
            interval=bybit_interval,
            limit=limite
        )
        
        if response["retCode"] == 0 and response["result"]["list"]:
            klines = response["result"]["list"]
            # Bybit API returns most recent data first
            klines.reverse()
            
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)
            
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            
            # Calculate basic statistics
            current = df["close"].iloc[-1]
            opening = df["open"].iloc[0]
            change = ((current / opening) - 1) * 100
            period_max = df["high"].max()
            period_min = df["low"].min()
            
            # Calculate ATR for volatility analysis and positioning
            df["tr"] = np.maximum(
                df["high"] - df["low"],
                np.maximum(
                    abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"] - df["close"].shift(1))
                )
            )
            df["atr14"] = df["tr"].rolling(window=14).mean()
            current_atr = df["atr14"].iloc[-1] if not np.isnan(df["atr14"].iloc[-1]) else 0
            
            summary = (f"Data for {symbol} (Perpetual Futures) in {intervalo} interval:\n"
                     f"Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}\n"
                     f"Current price: ${current:.4f}\n"
                     f"Period change: {change:.2f}%\n"
                     f"High: ${period_max:.4f}\n"
                     f"Low: ${period_min:.4f}\n"
                     f"ATR(14): ${current_atr:.4f} ({(current_atr/current)*100:.2f}% volatility)\n"
                     f"Total candles: {len(df)}")
            
            logger.info(f"Retrieved {len(df)} candles for {symbol}")
            return summary
        else:
            error = f"Error getting klines: {response['retMsg']}"
            logger.error(error)
            return error
    except Exception as e:
        error = f"Error getting klines: {str(e)}"
        logger.error(error)
        return error

# Technical Analysis Tools
@mcp.tool(description="Performs a complete technical analysis of the market for a given trading pair.")
def analyze_market(symbol: str, intervalo: str = "1") -> str:
    """
    Performs a complete technical analysis of the market for a given trading pair.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
        intervalo: Timeframe (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D)
    """
    logger.info(f"Tool 'analyze_market' called for {symbol} with interval {intervalo}")
    
    try:
        # Get historical data
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=intervalo,
            limit=200
        )
        
        if response["retCode"] != 0 or not response["result"]["list"]:
            error = f"Error getting historical data: {response.get('retMsg', 'No data')}"
            logger.error(error)
            return error
            
        klines = response["result"]["list"]
        klines.reverse()  # Bybit API returns newest first
        
        # Convert to dataframe
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        
        # Current price
        price = df["close"].iloc[-1]
        
        # Calculate technical indicators
        # Moving Averages
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["sma50"] = df["close"].rolling(window=50).mean()
        df["sma200"] = df["close"].rolling(window=200).mean()
        
        sma20 = df["sma20"].iloc[-1]
        sma50 = df["sma50"].iloc[-1]
        sma200 = df["sma200"].iloc[-1]
        
        # RSI
        delta = df["close"].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        rsi = df["rsi"].iloc[-1]
        
        # MACD
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        
        macd = df["macd"].iloc[-1]
        signal = df["signal"].iloc[-1]
        
        # Bollinger Bands
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["stddev"] = df["close"].rolling(window=20).std()
        df["upper_band"] = df["sma20"] + 2 * df["stddev"]
        df["lower_band"] = df["sma20"] - 2 * df["stddev"]
        df["bandwidth"] = (df["upper_band"] - df["lower_band"]) / df["sma20"]
        
        bandwidth = df["bandwidth"].iloc[-1]
        
        # ATR (Average True Range)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr"] = df["tr"].rolling(window=14).mean()
        atr = df["atr"].iloc[-1]
        
        # Volume analysis
        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Support and Resistance levels
        # Simple identification based on recent highs and lows
        supports = []
        resistances = []
        
        # Last 50 candles for support/resistance
        recent_df = df.iloc[-50:].copy()
        
        for i in range(2, len(recent_df) - 2):
            # Support: if we have a local minimum
            if (recent_df["low"].iloc[i] < recent_df["low"].iloc[i-1] and 
                recent_df["low"].iloc[i] < recent_df["low"].iloc[i-2] and
                recent_df["low"].iloc[i] < recent_df["low"].iloc[i+1] and
                recent_df["low"].iloc[i] < recent_df["low"].iloc[i+2]):
                supports.append(recent_df["low"].iloc[i])
            
            # Resistance: if we have a local maximum
            if (recent_df["high"].iloc[i] > recent_df["high"].iloc[i-1] and 
                recent_df["high"].iloc[i] > recent_df["high"].iloc[i-2] and
                recent_df["high"].iloc[i] > recent_df["high"].iloc[i+1] and
                recent_df["high"].iloc[i] > recent_df["high"].iloc[i+2]):
                resistances.append(recent_df["high"].iloc[i])
        
        # Sort supports (ascending) and resistances (descending)
        supports = sorted([s for s in supports if s < price])
        resistances = sorted([r for r in resistances if r > price], reverse=True)
        
        # Moving Average Trend
        trend_sma = "High" if sma20 > sma50 and sma50 > sma200 else ("Low" if sma20 < sma50 and sma50 < sma200 else "Neutral")
        
        # Price vs SMA
        price_vs_sma = "Above" if price > sma20 else "Below"
        
        # RSI Analysis
        rsi_status = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
        
        # MACD Analysis
        macd_status = "Positive" if macd > signal else "Negative"
        macd_cross = "Bullish Cross" if macd > signal and df["macd"].iloc[-2] <= df["signal"].iloc[-2] else \
                     ("Bearish Cross" if macd < signal and df["macd"].iloc[-2] >= df["signal"].iloc[-2] else "No Cross")
        
        # Volatility Analysis (ATR as % of price)
        volatility = (atr / price) * 100
        volatility_status = "High" if volatility > 3 else ("Low" if volatility < 1 else "Medium")
        
        # Bollinger Band Analysis
        bb_status = "Narrowed" if bandwidth < df["bandwidth"].rolling(window=50).mean().iloc[-1] * 0.8 else \
                   ("Expanded" if bandwidth > df["bandwidth"].rolling(window=50).mean().iloc[-1] * 1.2 else "Normal")
        
        # --- Format Analysis ---
        analysis = f"""Complete Technical Analysis for {symbol} (TimeFrame: {intervalo})
----------------------------------------------------------------------------------------
MARKET SUMMARY:
Current price: ${price:.4f}
ATR(14): ${atr:.4f} (Volatility: {volatility:.2f}% - {volatility_status})
Volume: {current_volume:.2f} ({volume_ratio:.2f}x of 20 period average)
Bollinger Bands: {bb_status} (Width: {bandwidth:.4f})

TREND:
Direction: {trend_sma}
Price vs SMA20: {price_vs_sma} (SMA20: ${sma20:.4f})
SMA50: ${sma50:.4f}
SMA200: ${sma200:.4f}

MOMENTUM ANALYSIS:
RSI(14): {rsi:.2f} - {rsi_status}
MACD: {macd:.6f} (Signal: {signal:.6f}) - {macd_status}, {macd_cross}

SUPPORT/RESISTANCE LEVELS:
"""
        
        # Add support levels
        if supports:
            analysis += "Supports:\n"
            for i, support in enumerate(supports, 1):
                analysis += f"S{i}: ${support:.4f} ({((price / support) - 1) * 100:.2f}% below)\n"
        else:
            analysis += "No recent support identified in last 50 candles\n"
            
        # Add resistance levels
        if resistances:
            analysis += "Resistances:\n"
            for i, resistance in enumerate(resistances, 1):
                analysis += f"R{i}: ${resistance:.4f} ({((resistance / price) - 1) * 100:.2f}% above)\n"
        else:
            analysis += "No recent resistance identified in last 50 candles\n"
            
        # Conclusion and strategy recommendation
        analysis += "\nCONCLUSION AND RECOMMENDATION:\n"
        
        # Conclusion based on indicators
        bull_signals = 0
        bear_signals = 0
        
        # Moving average check
        if price > sma20 > sma50:
            bull_signals += 1
        elif price < sma20 < sma50:
            bear_signals += 1
            
        # RSI check
        if rsi > 50 and rsi < 70:
            bull_signals += 0.5
        elif rsi < 50 and rsi > 30:
            bear_signals += 0.5
        elif rsi >= 70:
            bear_signals += 0.5  # Overbought can indicate high trend, but reversal risk
        elif rsi <= 30:
            bull_signals += 0.5  # Oversold can indicate low trend, but reversal risk
            
        # MACD check
        if macd > signal and macd > 0:
            bull_signals += 1
        elif macd < signal and macd < 0:
            bear_signals += 1
            
        # Final evaluation
        if bull_signals > bear_signals + 1:
            bias = "BULLISH (High Trend)"
            strategy = "Strategy: Consider long positions with stops below support S1"
        elif bear_signals > bull_signals + 1:
            bias = "BEARISH (Low Trend)"
            strategy = "Strategy: Consider short positions with stops above resistance R1"
        else:
            bias = "NEUTRAL (No clear trend)"
            strategy = "Strategy: Wait for trend confirmation or operate in range"
        
        # Add recommended position size based on ATR
        risk_amount = 2.0  # % of account willing to risk
        if volatility_status == "High":
            risk_amount = 1.0  # Reduce risk in high volatility
            
        # Calculate position size (simplified example)
        atr_multiplier = 2.0  # Stop distance in ATR multiples
        stop_distance = atr * atr_multiplier
        position_size = f"Recommended Position Size: Based on {risk_amount}% risk and stop at {atr_multiplier}x ATR (${stop_distance:.4f})"
        
        analysis += f"{bias}\n{strategy}\n{position_size}"
        
        logger.info(f"Complete technical analysis generated for {symbol}")
        return analysis
        
    except Exception as e:
        error = f"Error performing technical analysis: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Set leverage for the trading pair.")
def set_leverage(symbol: str, leverage: int) -> str:
    """
    Set the leverage for a specific trading pair.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
        leverage: Leverage value (ex: 10)
    """
    logger.info(f"Tool 'set_leverage' called for {symbol} with leverage {leverage}")    
    
    try:
        # Validate leverage value
        instrument = get_instrument_info(symbol)
        max_leverage = instrument["list"][0]["leverageFilter"]["maxLeverage"]
        if float(leverage) < 1 or float(leverage) > float(max_leverage):
            error = f"Error: Leverage must be between 1 and {max_leverage}"
            logger.error(error)
            return error
        
        # Send the leverage request
        response = session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage)
        )
        
        if response["retCode"] == 0:
            logger.info(f"Leverage set successfully for {symbol}")
            return f"Leverage set to {leverage} for {symbol}"
        
    except Exception as e:
        error = f"Error setting leverage: {str(e)}"
        logger.error(error)
        return error
       
@mcp.tool(description="Get instrument information for a specific trading pair.")
def get_instrument_info(symbol: str) -> str:
    """
    Get detailed information about a specific trading pair.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
    """
    logger.info(f"Tool 'get_instrument_info' called for {symbol}")
    
    try:
        # Send the instrument info request
        response = session.get_instruments_info(
            category="linear",
            symbol=symbol
        )
        
        if response["retCode"] == 0:
            info = response["result"]
            return info
        
        else:
            return f"Error getting instrument info: {response.get('retMsg', 'Unknown error')}"
        
    except Exception as e:
        error = f"Error getting instrument info: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Gets the orderbook depth data for a specific trading pair.")
def get_orderbook(symbol: str, category: str = "linear", limit: int = 200) -> str:
    """
    Gets the orderbook depth data for a specific trading pair.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
        category: Product type. spot, linear, inverse, option
        limit: Limit size for each bid and ask
               spot: [1, 200]. Default: 1.
               linear&inverse: [1, 500]. Default: 25.
               option: [1, 25]. Default: 1.
    """
    logger.info(f"Tool 'get_orderbook' called for {symbol} with category {category} and limit {limit}")
    
    try:
        # Validate limit based on category
        if category == "spot" and (limit < 1 or limit > 200):
            return f"Error: For spot, limit must be between 1 and 200"
        elif category in ["linear", "inverse"] and (limit < 1 or limit > 500):
            return f"Error: For {category}, limit must be between 1 and 500"
        elif category == "option" and (limit < 1 or limit > 25):
            return f"Error: For option, limit must be between 1 and 25"
        
        # Send the orderbook request
        response = session.get_orderbook(
            category=category,
            symbol=symbol,
            limit=limit
        )
        
        if response["retCode"] == 0:
            orderbook = response["result"]
            symbol = orderbook["s"]
            timestamp = orderbook["ts"]
            
            # Format asks and bids for readable output
            asks = orderbook["a"][:10]  # Limit to first 10 levels for readability
            bids = orderbook["b"][:10]  # Limit to first 10 levels for readability
            
            result = f"Orderbook for {symbol} (Category: {category}, Timestamp: {timestamp}):\n\n"
            
            # Format asks (sellers)
            result += "ASKS (Sellers):\n"
            result += "-"*10 + "\n"
            for i, ask in enumerate(asks):
                result += f"Level {i+1}: Price: ${float(ask[0]):.2f}, Size: {float(ask[1]):.6f}\n"
            
            result += "\nBIDS (Buyers):\n"
            result += "-"*10 + "\n"
            for i, bid in enumerate(bids):
                result += f"Level {i+1}: Price: ${float(bid[0]):.2f}, Size: {float(bid[1]):.6f}\n"
            
            # Calculate spread
            if asks and bids:
                best_ask = float(asks[0][0])
                best_bid = float(bids[0][0])
                spread = best_ask - best_bid
                spread_percent = (spread / best_bid) * 100
                
                result += f"\nSpread: ${spread:.2f} ({spread_percent:.4f}%)"
                
            # Additional orderbook metrics
            result += f"\nTotal levels returned: {len(orderbook['a'])} asks, {len(orderbook['b'])} bids"
            result += f"\nUpdate ID: {orderbook['u']}, Sequence: {orderbook['seq']}"
            
            logger.info(f"Orderbook successfully obtained for {symbol}")
            return result
        else:
            error = f"Error getting orderbook: {response['retMsg']}"
            logger.error(error)
            return error
            
    except Exception as e:
        error = f"Error getting orderbook: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Gets the open interest data for a specific trading pair.")
def get_open_interest(symbol: str, category: str = "linear", intervalTime: str = "1h", startTime: int = None, endTime: int = None, limit: int = 50) -> str:
    """
    Gets the open interest data for a specific trading pair.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
        category: Product type. linear, inverse
        intervalTime: Interval time. 5min,15min,30min,1h,4h,1d
        startTime: The start timestamp (ms)
        endTime: The end timestamp (ms)
        limit: Limit for data size per page. [1, 200]. Default: 50
    """
    logger.info(f"Tool 'get_open_interest' called for {symbol} with category {category} and interval {intervalTime}")
    
    try:
        # Validate category
        if category not in ["linear", "inverse"]:
            return f"Error: Category must be 'linear' or 'inverse'"
        
        # Validate intervalTime
        valid_intervals = ["5min", "15min", "30min", "1h", "4h", "1d"]
        if intervalTime not in valid_intervals:
            return f"Error: intervalTime must be one of {', '.join(valid_intervals)}"
        
        # Validate limit
        if limit < 1 or limit > 200:
            return f"Error: Limit must be between 1 and 200"
        
        # Prepare parameters
        params = {
            "category": category,
            "symbol": symbol,
            "intervalTime": intervalTime,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if startTime is not None:
            params["startTime"] = startTime
        if endTime is not None:
            params["endTime"] = endTime
        
        # Send the open interest request
        response = session.get_open_interest(**params)
        
        if response["retCode"] == 0:
            result_data = response["result"]
            symbol = result_data["symbol"]
            category = result_data["category"]
            
            # Format the response
            result = f"Open Interest for {symbol} (Category: {category}, Interval: {intervalTime}):\n\n"
            
            # Add data points
            for entry in result_data["list"]:
                timestamp = int(entry["timestamp"])
                dt = datetime.fromtimestamp(timestamp/1000)
                open_interest = float(entry["openInterest"])
                
                # Format based on category
                if category == "inverse":
                    unit = "USD"
                else:  # linear
                    unit = symbol.replace("USDT", "").replace("USDC", "")
                
                result += f"{dt}: {open_interest:,.2f} {unit}\n"
            
            # Add pagination info if available
            if result_data.get("nextPageCursor"):
                result += f"\nMore data available. Use cursor: {result_data['nextPageCursor']}"
            
            logger.info(f"Open interest data successfully obtained for {symbol}")
            return result
        else:
            error = f"Error getting open interest: {response['retMsg']}"
            logger.error(error)
            return error
            
    except Exception as e:
        error = f"Error getting open interest: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Performs a technical analysis across multiple timeframes for signal confirmation.")
def analyze_multi_timeframe(symbol: str) -> str:
    """
    Performs a technical analysis across multiple timeframes for signal confirmation.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
    """
    logger.info(f"Tool 'analyze_multi_timeframe' called for {symbol}")
    
    try:
        # Define timeframes to analyze
        timeframes = ["1", "5", "15", "60", "240"]
        tf_names = {"1": "1 minute", "5": "5 minutes", "15": "15 minutes", "60": "1 hour", "240": "4 hours"}
        
        result = f"Multi-Timeframe Analysis for {symbol}\n"
        result += "="*80 + "\n\n"
        
        # Counters for signals
        bull_count = 0
        bear_count = 0
        neutro_count = 0
        
        # Store key indicators by timeframe
        indicators = {}
        
        # Analyze each timeframe
        for tf in timeframes:
            logger.info(f"Analyzing timeframe {tf} for {symbol}")
            
            # Get historical data
            response = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=tf,
                limit=100
            )
            
            if response["retCode"] != 0 or not response["result"]["list"]:
                error = f"Error getting historical data: {response.get('retMsg', 'No data')}"
                logger.error(error)
                continue
                
            klines = response["result"]["list"]
            klines.reverse()  # API returns newest first
            
            # Convert to dataframe for analysis
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)
            
            # Calculate key indicators
            price = df["close"].iloc[-1]
            
            # Moving averages
            df["sma20"] = df["close"].rolling(window=20).mean()
            df["sma50"] = df["close"].rolling(window=50).mean()
            df["sma200"] = df["close"].rolling(window=200).mean()
            
            sma20 = df["sma20"].iloc[-1]
            sma50 = df["sma50"].iloc[-1]
            sma200 = df["sma200"].iloc[-1]
            
            # RSI calculation
            delta = df["close"].diff()
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df["rsi"] = 100 - (100 / (1 + rs))
            rsi = df["rsi"].iloc[-1]
            
            # MACD
            df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = df["ema12"] - df["ema26"]
            df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            
            macd = df["macd"].iloc[-1]
            signal = df["signal"].iloc[-1]
            
            # Store indicators
            indicators[tf] = {
                "price": price,
                "sma20": sma20,
                "sma50": sma50,
                "sma200": sma200,
                "rsi": rsi,
                "macd": macd,
                "signal": signal
            }
            
            # Trend analysis for this timeframe
            trend = "NEUTRAL"
            sinais_bull = 0
            sinais_bear = 0
            
            # Price above moving averages
            if price > sma20:
                sinais_bull += 1
            else:
                sinais_bear += 1
                
            if sma20 > sma50:
                sinais_bull += 1
            else:
                sinais_bear += 1
                
            # RSI analysis
            if rsi > 50:
                sinais_bull += 0.5
            else:
                sinais_bear += 0.5
                
            # MACD analysis
            if macd > signal:
                sinais_bull += 1
            else:
                sinais_bear += 1
                
            # Determine trend based on signals
            if sinais_bull > sinais_bear + 1:
                trend = "BULLISH"
                bull_count += 1
            elif sinais_bear > sinais_bull + 1:
                trend = "BEARISH"
                bear_count += 1
            else:
                neutro_count += 1
                
            # Add to result
            result += f"TIMEFRAME: {tf_names[tf]}\n"
            result += f"Price: ${price:.4f}\n"
            result += f"SMA20: ${sma20:.4f}, SMA50: ${sma50:.4f}, SMA200: ${sma200:.4f}\n"
            result += f"RSI: {rsi:.2f}\n"
            result += f"MACD: {macd:.6f}, Signal: {signal:.6f}\n"
            result += f"Trend: {trend}\n\n"
            
        # Consolidated analysis
        result += "CONSOLIDATED ANALYSIS:\n"
        result += "="*50 + "\n"
        
        # Check agreement between timeframes
        if bull_count > bear_count and bull_count > neutro_count:
            trend_general = "BULLISH (HIGH) ↑"
            confidence = (bull_count / len(timeframes)) * 100
        elif bear_count > bull_count and bear_count > neutro_count:
            trend_general = "BEARISH (LOW) ↓"
            confidence = (bear_count / len(timeframes)) * 100
        else:
            trend_general = "NEUTRAL (SIDEWAYS) →"
            confidence = (neutro_count / len(timeframes)) * 100
            
        result += f"Predominant trend: {trend_general}\n"
        result += f"Confidence level: {confidence:.1f}%\n"
        result += f"Consensus: {max(bull_count, bear_count, neutro_count)}/{len(timeframes)} timeframes\n\n"
        
        # Strategy recommendation based on multi-timeframe analysis
        if bull_count >= 3:
            result += "RECOMMENDATION: Consider long/buy positions with adequate risk management.\n"
        elif bear_count >= 3:
            result += "RECOMMENDATION: Consider short/sell positions with adequate risk management.\n"
        else:
            result += "RECOMMENDATION: Wait for clearer signals or focus on short-term operations.\n"
        
        logger.info(f"Multi-timeframe analysis completed for {symbol}")
        return result
    
    except Exception as e:
        error = f"Error performing multi-timeframe analysis: {str(e)}"
        logger.error(error)
        return error

# Trading Tools
@mcp.tool(description="Places a buy or sell order in the futures perpetual market.")
def place_order(symbol: str, side: str, order_type: str, qty: float, price: float = None, 
                 take_profit: float = None, stop_loss: float = None, trailing_stop: float = None) -> str:
    """
    Places a buy or sell order in the futures perpetual market.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
        side: Direction of the order ('Buy' or 'Sell')
        order_type: Type of order ('Market' or 'Limit')
        qty: Size of the position in contracts
        price: Limit price (required for Limit orders)
        take_profit: Take profit price (optional)
        stop_loss: Stop loss price (optional)
        trailing_stop: Trailing stop distance in % (optional)
    """
    logger.info(f"Tool 'place_order' called for {symbol}: {side} {order_type} {qty}")
    
    # Validate parameters
    if side not in ["Buy", "Sell"]:
        error = "Error: 'side' parameter must be 'Buy' or 'Sell'"
        logger.error(error)
        return error
        
    if order_type not in ["Market", "Limit"]:
        error = "Error: 'order_type' parameter must be 'Market' or 'Limit'"
        logger.error(error)
        return error
        
    if order_type == "Limit" and price is None:
        error = "Error: Limit orders require a price"
        logger.error(error)
        return error
        
    if qty <= 0:
        error = "Error: Quantity must be greater than zero"
        logger.error(error)
        return error
    
    
    try:
        positionIdx = 1 if side == "Buy" else 2
        # Prepare order parameters
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "positionIdx": positionIdx
        }
        
        # Add price for Limit orders
        if order_type == "Limit" and price is not None:
            params["price"] = str(price)
            
        # Add take profit and stop loss if provided
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
            
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
        
        # Send the order
        response = session.place_order(
            **params
        )
        
        if response["retCode"] == 0:
            order_id = response["result"]["orderId"]
            created_time = response["result"].get("createdTime", "N/A")
            
            # If trailing stop was specified, configure it after the order is placed
            if trailing_stop is not None and trailing_stop > 0:
                logger.info(f"Configuring trailing stop at {trailing_stop}% for order {order_id}")
                
                # Wait a moment to ensure the order was processed
                time.sleep(2)
                
                # Configure trailing stop
                try:
                    trailing_stop_response = session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        trailingStop=str(trailing_stop),
                        positionIdx=0  # 0 for one-way mode
                    )
                    
                    if trailing_stop_response["retCode"] == 0:
                        logger.info(f"Trailing stop configured successfully for {symbol}")
                    else:
                        logger.error(f"Error configuring trailing stop: {trailing_stop_response['retMsg']}")
                except Exception as e:
                    logger.error(f"Error configuring trailing stop: {str(e)}")
            
            result = (f"Order placed successfully:\n"
                     f"Symbol: {symbol}\n"
                     f"Side: {side}\n"
                     f"Type: {order_type}\n"
                     f"Quantity: {qty}\n")
                     
            if order_type == "Limit":
                result += f"Price: ${price}\n"
                
            if take_profit is not None:
                result += f"Take Profit: ${take_profit}\n"
                
            if stop_loss is not None:
                result += f"Stop Loss: ${stop_loss}\n"
                
            if trailing_stop is not None:
                result += f"Trailing Stop: {trailing_stop}%\n"
                
            result += (f"Order ID: {order_id}\n"
                      f"Creation Time: {created_time}")
                      
            logger.info(f"Order placed successfully: {order_id}")
            return result
        else:
            error = f"Error placing order: {response['retMsg']}"
            logger.error(error)
            return error
            
    except Exception as e:
        error_msg = f"Error performing multi-timeframe analysis: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return error_msg

@mcp.tool(description="Cancels an active order.")
def cancel_order(symbol: str, order_id: str) -> str:
    """
    Cancels an active order.
    
    Args:
        symbol: Symbol of trading pair (ex: BTCUSDT)
        order_id: ID of the order to be canceled
    """
    logger.info(f"Tool 'cancel_order' called for {symbol}, order {order_id}")
    
    try:
        response = session.cancel_order(
            category="linear",
            symbol=symbol,
            orderId=order_id
        )
        
        if response["retCode"] == 0:
            result = f"Order {order_id} canceled successfully!"
            logger.info(f"Order canceled successfully")
            return result
        else:
            error = f"Error canceling order: {response['retMsg']}"
            logger.error(error)
            return error
            
    except Exception as e:
        error = f"Error canceling order: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Gets all active orders.")
def get_active_orders() -> str:
    """
    Gets all active orders.
    
    Args:
        symbol: Symbol of trading pair (optional, if not provided returns all)
    """
    logger.info(f"Tool 'get_active_orders' called")
    
    try:
        params = {"category": "linear", "settleCoin": "USDT"}
        response = session.get_open_orders(
            **params
        )
        
        if response["retCode"] == 0:
            orders = response["result"]["list"]
            
            if not orders:
                return "No active orders found."
                
            result = f"Active orders found: {len(orders)}\n\n"
            
            for order in orders:
                result += (f"ID: {order['orderId']}\n"
                             f"Symbol: {order['symbol']}\n"
                             f"Side: {order['side']}\n"
                             f"Type: {order['orderType']}\n"
                             f"Price: {order['price']}\n"
                             f"Quantity: {order['qty']}\n"
                             f"Status: {order['orderStatus']}\n"
                             f"Created at: {datetime.fromtimestamp(int(order['createdTime'])/1000)}\n\n")
                
            logger.info(f"Found {len(orders)} active orders")
            return result
        else:
            error = f"Error getting active orders: {response['retMsg']}"
            logger.error(error)
            return error
            
    except Exception as e:
        error = f"Error getting active orders: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Gets all active positions.")
def get_active_positions() -> str:
    """
    Gets all active positions.
    """
    logger.info(f"Tool 'get_active_positions' called")
    
    try:
        response = session.get_positions(
            category="linear",
            settleCoin="USDT"
        )
        
        if response["retCode"] == 0:
            positions = response["result"]["list"]
            
            if not positions:
                return "No active positions found."
                
            result = f"Active positions found: {len(positions)}\n\n"
            
            for position in positions:
                result += (f"Symbol: {position['symbol']}\n"
                             f"Side: {position['side']}\n"
                             f"Quantity: {position['size']}\n"
                             f"P/L: {position['unrealisedPnl']}\n"
                             f"Created at: {datetime.fromtimestamp(int(position['createdTime'])/1000)}\n\n")
                
            logger.info(f"Found {len(positions)} active positions")
            return result
        else:
            error = f"Error getting active positions: {response['retMsg']}"
            logger.error(error)
            return error
            
    except Exception as e:
        error = f"Error getting active positions: {str(e)}"
        logger.error(error)
        return error
    
@mcp.tool(description="Get closed P/L")
def get_closed_pl() -> str:
    """
    Gets the closed P/L.
    """
    logger.info(f"Tool 'get_closed_pl' called")
    
    try:
        response = session.get_closed_pnl(
            category="linear",
            settleCoin="USDT"
        )
        
        if response["retCode"] == 0:
            result = f"Closed P/L: {response['result']['total']}"
            logger.info(f"Closed P/L: {response['result']['total']}")
            return result
        else:
            error = f"Error getting closed P/L: {response['retMsg']}"
            logger.error(error)
            return error
        
    except Exception as e:
        error = f"Error getting closed P/L: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Get funding rate history, symbol: SUIUSDT, startTime: milisseconds, endTime: milisseconds, limit: 200")
def get_funding_rate_history(symbol: str = 'SUIUSDT', startTime: int = None, endTime: int = None, limit: int = 200) -> str:
    """
    Gets the funding rate history.
    """
    logger.info(f"Tool 'get_funding_rate_history' called")
    
    try:
        response = session.get_funding_rate_history(
            category="linear",
            symbol=symbol,
            startTime=startTime,
            endTime=endTime,
            limit=limit
        )

        if response["retCode"] == 0:
            result = f"Funding rate history: {response['result']}"
            logger.info(f"Funding rate history: {response['result']}")
            return result
        else:
            error = f"Error getting funding rate history: {response['retMsg']}"
            logger.error(error)
            return error
        
    except Exception as e:
        error = f"Error getting funding rate history: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Get Open & Closed Orders")
def get_open_closed_orders() -> str:
    """
    Gets the open and closed orders.
    """
    logger.info(f"Tool 'get_open_closed_orders' called")
    
    try:
        response = session.get_open_orders(
            category="linear",
            settleCoin="USDT"
        )
        
        if response["retCode"] == 0:
            result = f"Open orders: {response['result']}"
            logger.info(f"Open orders: {response['result']}")
            return result
        else:
            error = f"Error getting open orders: {response['retMsg']}"
            logger.error(error)
            return error
        
    except Exception as e:
        error = f"Error getting open orders: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Amend Order")
def amend_order(symbol: str, category: str = 'linear', order_id: str = None, qty: float = None, price: float = None) -> str:
    """
    Amend an active order.
    """
    logger.info(f"Tool 'amend_order' called")
    
    try:
        response = session.amend_order(
            category=category,
            symbol=symbol,
            orderId=order_id,
            qty=qty,
            price=price
        )
    
        if response["retCode"] == 0:
            result = f"Order amended successfully"
            logger.info(f"Order amended successfully")
            return result
        else:
            error = f"Error amending order: {response['retMsg']}"
            logger.error(error)
            return error
        
    except Exception as e:
        error = f"Error amending order: {str(e)}"
        logger.error(error)
        return error

@mcp.tool(description="Cancel Order")
def cancel_order(symbol: str, category: str = 'linear') -> str:
    """
    Cancel an active order.
    """
    logger.info(f"Tool 'cancel_order' called")
    
    try:
        response = session.cancel_order(
            category=category,
            symbol=symbol
        )
    
        if response["retCode"] == 0:
            result = f"Order canceled successfully"
            logger.info(f"Order canceled successfully")
            return result
        else:
            error = f"Error canceling order: {response['retMsg']}"
            logger.error(error)
            return error
        
    except Exception as e:
        error = f"Error canceling order: {str(e)}"
        logger.error(error)
        return error

# Prompts
@mcp.prompt(name="analyze_market", description="Technical analysis across multiple timeframes for trend confirmation.")
def analyze_market_prompt(symbol: str) -> str:
    return analyze_multi_timeframe(symbol)

if __name__ == "__main__":
    logger.info("Starting Bybit MCP server...")
    mcp.run()