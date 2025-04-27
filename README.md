# 🤖 Solarium Quantum

<div align="center">

![Version](https://img.shields.io/badge/version-0.0.1-purple)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

</div>

<p align="center">
  <b>AI-Powered Trading Infrastructure for Bybit Cryptocurrency Exchange</b><br>
  <sub>An advanced AI Agent for automated cryptocurrency trading on the Bybit exchange<sub>
</p>

## 🌟 Overview

Solarium Quantum is a powerful AI Agent which performs trades [Bybit API](https://bybit-exchange.github.io/docs/v5/intro) through the Machine Context Protocol (MCP). It transforms raw market data into actionable insights, enabling sophisticated algorithmic trading strategies, real-time market analysis, and automated execution systems.

### What Makes This Project Special

- **AI-Ready Architecture**: Built from the ground up to integrate with Large Language Models
- **Comprehensive Trading Tools**: 15+ specialized tools for market analysis and trade execution
- **Multi-timeframe Analysis**: Advanced technical indicators across different time horizons
- **Risk Management**: Volatility-based position sizing and automated risk controls
- **Production-Grade Infrastructure**: Docker-based deployment with n8n workflow automation
- **Telegram Integration**: Real-time trade notifications and performance reports delivered to your device

## 🚀 Features

<table>
  <tr>
    <td width="33%">
      <h3 align="center">📊 Market Analysis</h3>
      <ul>
        <li>Real-time price and ticker data</li>
        <li>Historical candlestick retrieval</li>
        <li>Advanced technical indicators (RSI, MACD, ATR)</li>
        <li>Multi-timeframe trend analysis</li>
        <li>Order book depth visualization</li>
      </ul>
    </td>
    <td width="33%">
      <h3 align="center">💹 Trading Operations</h3>
      <ul>
        <li>Place market and limit orders</li>
        <li>Configure take profit, stop loss & trailing stops</li>
        <li>Position monitoring and management</li>
        <li>Leverage and margin configuration</li>
        <li>Order cancellation and modification</li>
      </ul>
    </td>
    <td width="33%">
      <h3 align="center">🛡️ Risk Management</h3>
      <ul>
        <li>Volatility-based position sizing</li>
        <li>Dynamic stop-loss calculation</li>
        <li>Risk-reward analysis</li>
        <li>Portfolio balance tracking</li>
        <li>Exposure monitoring</li>
      </ul>
    </td>
  </tr>
</table>

## 📋 Requirements

- Python 3.12+
- Bybit API credentials
- Docker & Docker Compose (for containerized deployment)

## ⚙️ Installation

### Method 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/andrefigueira93/solarium.git
cd solarium

# Create .env file with your API credentials
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=True  # Set to False for production

# Build and start with Docker Compose
docker-compose build
docker-compose up -d
```

### Method 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/andrefigueira93/solarium.git
cd solarium

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Create .env file with your API credentials
# (Same as in Docker method)

# Start the MCP server
mcp run main.py -t sse
```

> In any case, after you spin up the server, you will be able to access the n8n instance on http://localhost:5678. And then you should:
>
> - Setup your user
> - Setup your preferred model credentials
> - Setup your Telegram Bot credentials for trade notifications
> - Import Solarium_Quantum.json Workflow
> - Customize prompts as you want. Default trades are on SUIUSDT

## 🔌 Architecture

```
         ┌─────────────┐           ┌─────────────┐
         │             │           │             │
         │  LLM Agent  │◄────────►│ MCP Client  │
         │             │           │             │
         └─────────────┘           └──────┬──────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                      Bybit MCP Server                       │
│                                                             │
├─────────────┬─────────────┬──────────────┬─────────────────┤
│  Technical  │   Market    │   Trading    │      Risk       │
│  Analysis   │    Data     │  Operations  │   Management    │
└─────┬───────┴──────┬──────┴───────┬──────┴────────┬────────┘
      │              │              │               │
      ▼              ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                        Bybit API                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Usage Examples

### Interactive Usage with an MCP Client

```python
from mcp.client import Client

# Connect to the MCP server
client = Client("http://localhost:8000")

# Get current Bitcoin price
ticker = client.call("get_ticker", symbol="BTCUSDT")
print(ticker)

# Perform technical analysis
analysis = client.call("analyze_market", symbol="ETHUSDT", intervalo="60")
print(analysis)

# Place a market order with stop loss
order = client.call("place_order",
                    symbol="BTCUSDT",
                    side="Buy",
                    order_type="Market",
                    qty=0.001,
                    stop_loss=25000)
print(order)

# Get all active positions
positions = client.call("get_active_positions")
print(positions)
```

### Integration with AI Agents

```python
import openai
from mcp.client import Client

# Initialize MCP client
mcp_client = Client("http://localhost:8000")

# Function to let LLM interact with MCP
def analyze_trading_opportunity(symbol):
    # Get market data via MCP
    market_data = mcp_client.call("analyze_multi_timeframe", symbol=symbol)

    # Let LLM analyze the data
    response = openai.Completion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a crypto trading expert."},
            {"role": "user", "content": f"Analyze this market data and suggest a trading strategy:\n{market_data}"}
        ]
    )

    return response.choices[0].message.content

# Example usage
trading_advice = analyze_trading_opportunity("BTCUSDT")
print(trading_advice)
```

## 📦 Docker Services

The included docker-compose.yml file sets up a complete trading infrastructure:

- **bybit_mcp_server**: The core MCP server for Bybit trading
- **n8n**: Workflow automation for trade scheduling
- **postgres**: Database for storing trading data and backtesting results
- **redis**: Caching layer for high-performance data access

## 🔧 Available Tools

| Tool                      | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `get_balance`             | Retrieves current wallet balance                 |
| `get_ticker`              | Gets current price information for a symbol      |
| `get_klines`              | Gets historical candlestick data                 |
| `analyze_market`          | Performs comprehensive technical analysis        |
| `analyze_multi_timeframe` | Analyzes trends across multiple timeframes       |
| `set_leverage`            | Sets leverage for trading                        |
| `get_instrument_info`     | Gets trading pair specifications                 |
| `get_orderbook`           | Gets order book depth data                       |
| `place_order`             | Places a buy/sell order with optional parameters |
| `cancel_order`            | Cancels an active order                          |
| `get_active_orders`       | Lists all active orders                          |
| `get_active_positions`    | Lists all open positions                         |

## 🔐 Environment Variables

| Variable           | Description                                      | Default  |
| ------------------ | ------------------------------------------------ | -------- |
| `BYBIT_API_KEY`    | Your Bybit API key                               | Required |
| `BYBIT_API_SECRET` | Your Bybit API secret                            | Required |
| `BYBIT_TESTNET`    | Whether to use testnet (True) or mainnet (False) | `True`   |

## 👨‍💻 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Trading cryptocurrencies involves significant risk and can result in the loss of your invested capital. This software is delivered "as is" and is not financial advice. Always do your own research before trading.**

---

<p align="center">
    Made with ❤️ by <a href="https://github.com/andrefigueira93">André Figueira</a>
    <br><br>
    <a href="https://www.buymeacoffee.com/andrefigueira"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support-yellow.svg?style=for-the-badge" height="35"></a>
</p>
