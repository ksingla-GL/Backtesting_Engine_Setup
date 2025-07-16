# Zero-Code Backtesting System

A flexible, database-driven backtesting system for trading strategies that requires no coding to create or modify strategies.

## Features

- **True Zero-Code Strategy Creation**: All strategy logic stored in database
- **Flexible Rule Expressions**: Use expressions like `SMA(ES, 50) > PRICE` instead of hardcoded logic
- **Multi-Timeframe Support**: Automatically uses highest frequency data available
- **Generic Calculation Methods**: Reusable indicators (SMA, EMA, RSI, etc.) for any symbol/period
- **Performance Tracking**: Comprehensive metrics including Sharpe ratio, drawdown, etc.

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Create the database**:
```bash
python flexible_database_schema.py
```

This will:
- Create a fresh SQLite database
- Set up all tables with proper schema
- Migrate your historical price data
- Configure the Quick Panic ES strategy
- Create a `.gitignore` file

3. **Run a backtest**:
```bash
python run_backtest.py --strategy_id=1 --start=2023-01-01 --end=2023-12-31
```

## Project Structure

```
backtesting_system/
├── flexible_database_schema.py  # Database setup and data migration
├── expression_parser.py         # Flexible rule expression parser
├── calculate_indicators.py      # On-the-fly indicator calculations
├── backtesting_engine.py        # Main backtesting engine
├── run_backtest.py             # CLI script to run backtests
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file (auto-generated)
└── README.md                   # This file
```

## Database Schema

### Key Tables

1. **calculation_methods**: Generic calculation definitions (SMA, EMA, RSI, etc.)
2. **strategy**: Strategy definitions with auto frequency detection
3. **strategy_rules**: Flexible rule expressions
4. **price_data**: Historical OHLCV data
5. **market_indicator**: Market-wide indicators (VIX, TRIN, etc.)

### Strategy Rule Examples

Instead of hardcoding indicators, use flexible expressions:

```sql
-- Old way (inflexible)
rule_components: {"left": "CALC.ES_SMA_50", "operator": ">", "right": "PRICE.ES.CLOSE"}

-- New way (flexible)
rule_expression: "PRICE > SMA(PRIMARY, 50)"
```

## Creating New Strategies

1. Insert strategy record:
```sql
INSERT INTO strategy (name, strategy_type, primary_symbol, use_highest_frequency)
VALUES ('My Strategy', 'trend_following', 'SPY', 1);
```

2. Add flexible rules:
```sql
INSERT INTO strategy_rules (strategy_id, rule_type, rule_expression)
VALUES 
  (2, 'filter', 'PRICE > SMA(PRIMARY, 200)'),
  (2, 'entry', 'RSI(PRIMARY, 14) > 70'),
  (2, 'exit', 'CROSSES_BELOW(EMA(PRIMARY, 9), EMA(PRIMARY, 21))');
```

## Available Functions

### Indicators
- `SMA(symbol, period)` - Simple Moving Average
- `EMA(symbol, period)` - Exponential Moving Average  
- `RSI(symbol, period)` - Relative Strength Index
- `DECLINE_FROM_PEAK(symbol, period)` - % decline from rolling peak
- `SPIKE(symbol, period)` - % spike from rolling minimum

### Signals
- `CROSSES_ABOVE(series1, series2)` - Crossover detection
- `CROSSES_BELOW(series1, series2)` - Crossunder detection

### Special Variables
- `PRIMARY` - Primary symbol of strategy
- `PRICE` - Current close price
- `INDICATOR(name)` - Market indicator value
- `PARAM(name)` - Strategy parameter value

## Data Requirements

Place your data files in the configured data directory:
- Intraday data: `1 min/`, `5 min/`, etc. folders
- Daily data: `Daily/` folder
- Market indicators: Root directory or appropriate folders

## Command Line Options

```bash
python run_backtest.py [options]

Options:
  --strategy_id   Strategy ID to backtest (default: 1)
  --start         Start date YYYY-MM-DD (default: 2023-01-01)
  --end           End date YYYY-MM-DD (default: 2023-12-31)
  --capital       Starting capital (default: 100000)
  --db            Database file path (default: backtesting_v2.db)
```

## Notes

- Database file (`*.db`) is excluded from git by default
- Only code files are tracked in the repository
- System automatically detects and uses highest frequency data
- All calculations done on-the-fly, not stored in database

## Performance Tips

1. **Data Loading**: The system loads only required data for the backtest period
2. **Indicator Caching**: Indicators are calculated once and reused
3. **Expression Parsing**: Rule expressions are parsed once at startup
4. **Database Indexes**: Proper indexes ensure fast data retrieval

## Troubleshooting

1. **No data found error**: Check that your data files are in the correct folders
2. **Missing indicators**: Ensure market indicator files are properly named
3. **Expression errors**: Verify your rule expressions follow the correct syntax
4. **Memory issues**: For very long backtests, consider chunking by year
