"""
Flexible Database Schema and Setup for Backtesting System
Fixed version with better error handling
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Configuration
DB_PATH = r'C:\Users\kshit\Desktop\Upwork\Kyrg_Mean_Reversion\backtesting_v2.db'
DATA_ROOT = r'C:\Users\kshit\Desktop\Upwork\Kyrg_Mean_Reversion\Backtesting_data'

# Delete existing database to start fresh
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"Deleted existing database: {DB_PATH}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FlexibleDatabaseSetup:
    def __init__(self, db_path, data_root):
        self.db_path = db_path
        self.data_root = data_root
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Enable optimizations
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        
    def create_flexible_schema(self):
        """Create a more flexible and scalable schema"""
        logger.info("Creating flexible database schema...")
        
        # 1. SYMBOL table (unchanged)
        self.cursor.execute("""
            CREATE TABLE symbol (
                symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker VARCHAR(10) UNIQUE NOT NULL,
                exchange VARCHAR(20) DEFAULT 'CME',
                symbol_type VARCHAR(20) DEFAULT 'future',
                is_tradeable BOOLEAN DEFAULT 1
            )
        """)
        
        # 2. PRICE_DATA table (unchanged but with index)
        self.cursor.execute("""
            CREATE TABLE price_data (
                datetime DATETIME NOT NULL,
                symbol_id INTEGER NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER DEFAULT 0,
                PRIMARY KEY (datetime, symbol_id, timeframe),
                FOREIGN KEY (symbol_id) REFERENCES symbol(symbol_id)
            )
        """)
        
        # 3. MARKET_INDICATOR table (unchanged)
        self.cursor.execute("""
            CREATE TABLE market_indicator (
                datetime DATETIME NOT NULL,
                indicator_name VARCHAR(50) NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (datetime, indicator_name)
            )
        """)
        
        # 4. CALCULATION_METHODS - Generic calculation definitions
        self.cursor.execute("""
            CREATE TABLE calculation_methods (
                method_id INTEGER PRIMARY KEY AUTOINCREMENT,
                method_name VARCHAR(50) UNIQUE NOT NULL,
                method_category VARCHAR(50),
                input_type VARCHAR(50),
                required_params TEXT,
                optional_params TEXT,
                min_lookback INTEGER DEFAULT 0,
                description TEXT,
                formula TEXT
            )
        """)
        
        # 5. STRATEGY table with auto-detect highest frequency
        self.cursor.execute("""
            CREATE TABLE strategy (
                strategy_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL,
                version VARCHAR(20) DEFAULT 'v1.0',
                strategy_type VARCHAR(50),
                strategy_level VARCHAR(20) DEFAULT 'main',
                parent_strategy_id INTEGER,
                primary_symbol VARCHAR(10),
                secondary_symbols TEXT,
                use_highest_frequency BOOLEAN DEFAULT 1,
                fallback_timeframe VARCHAR(10) DEFAULT '1min',
                min_data_quality_score REAL DEFAULT 0.95,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                modified_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                description TEXT,
                FOREIGN KEY (parent_strategy_id) REFERENCES strategy(strategy_id)
            )
        """)
        
        # 6. STRATEGY_RULES - More flexible rule components
        self.cursor.execute("""
            CREATE TABLE strategy_rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                rule_type VARCHAR(50) NOT NULL,
                rule_name VARCHAR(100),
                rule_logic TEXT,
                rule_expression TEXT,
                priority INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (strategy_id) REFERENCES strategy(strategy_id)
            )
        """)
        
        # 7. STRATEGY_PARAMETERS (unchanged)
        self.cursor.execute("""
            CREATE TABLE strategy_parameters (
                param_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                param_name VARCHAR(100) NOT NULL,
                param_value TEXT,
                param_type VARCHAR(50) DEFAULT 'numeric',
                min_value REAL,
                max_value REAL,
                description TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategy(strategy_id),
                UNIQUE(strategy_id, param_name)
            )
        """)
        
        # 8. POSITION_SIZING_RULES (unchanged)
        self.cursor.execute("""
            CREATE TABLE position_sizing_rules (
                sizing_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                sizing_type VARCHAR(50),
                sizing_params TEXT,
                max_position_size REAL,
                priority INTEGER DEFAULT 1,
                FOREIGN KEY (strategy_id) REFERENCES strategy(strategy_id)
            )
        """)
        
        # 9. TIME_FILTERS table
        self.cursor.execute("""
            CREATE TABLE time_filters (
                filter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                filter_type VARCHAR(50),
                filter_params TEXT,
                exclude_or_include VARCHAR(10) DEFAULT 'include',
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (strategy_id) REFERENCES strategy(strategy_id)
            )
        """)
        
        # 10. BACKTEST and related tables (unchanged)
        self.cursor.execute("""
            CREATE TABLE backtest (
                backtest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                config TEXT,
                data_frequency_used VARCHAR(10),
                FOREIGN KEY (strategy_id) REFERENCES strategy(strategy_id)
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE trade (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER NOT NULL,
                symbol_id INTEGER NOT NULL,
                entry_datetime DATETIME NOT NULL,
                exit_datetime DATETIME,
                entry_price REAL NOT NULL,
                exit_price REAL,
                position_size INTEGER DEFAULT 1,
                pnl REAL,
                pnl_percent REAL,
                entry_conditions TEXT,
                exit_reason TEXT,
                max_profit REAL,
                max_loss REAL,
                FOREIGN KEY (backtest_id) REFERENCES backtest(backtest_id),
                FOREIGN KEY (symbol_id) REFERENCES symbol(symbol_id)
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE backtest_metrics (
                backtest_id INTEGER PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                total_return REAL,
                total_pnl REAL,
                avg_pnl REAL,
                avg_win REAL,
                avg_loss REAL,
                avg_duration_hours REAL,
                final_capital REAL,
                commissions_paid REAL,
                FOREIGN KEY (backtest_id) REFERENCES backtest(backtest_id)
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX idx_price_symbol_date ON price_data(symbol_id, datetime)",
            "CREATE INDEX idx_price_timeframe ON price_data(timeframe, symbol_id)",
            "CREATE INDEX idx_indicator_date ON market_indicator(datetime)",
            "CREATE INDEX idx_indicator_name ON market_indicator(indicator_name, datetime)",
            "CREATE INDEX idx_trade_backtest ON trade(backtest_id)",
            "CREATE INDEX idx_strategy_active ON strategy(is_active)",
            "CREATE INDEX idx_rules_strategy ON strategy_rules(strategy_id, rule_type)"
        ]
        
        for idx in indexes:
            self.cursor.execute(idx)
        
        self.conn.commit()
        logger.info("Flexible schema created successfully")
    
    def setup_calculation_methods(self):
        """Setup generic calculation methods that can be reused"""
        methods = [
            {
                'name': 'SMA',
                'category': 'moving_average',
                'input_type': 'price',
                'required_params': ['period'],
                'optional_params': ['column'],
                'min_lookback': 'period',
                'description': 'Simple Moving Average',
                'formula': 'SUM(values) / period'
            },
            {
                'name': 'EMA',
                'category': 'moving_average',
                'input_type': 'price',
                'required_params': ['period'],
                'optional_params': ['column'],
                'min_lookback': 'period * 2',
                'description': 'Exponential Moving Average',
                'formula': 'Price * multiplier + EMA_prev * (1 - multiplier)'
            },
            {
                'name': 'RSI',
                'category': 'momentum',
                'input_type': 'price',
                'required_params': ['period'],
                'optional_params': ['column'],
                'min_lookback': 'period + 1',
                'description': 'Relative Strength Index',
                'formula': '100 - (100 / (1 + RS))'
            },
            {
                'name': 'DECLINE_FROM_PEAK',
                'category': 'trend',
                'input_type': 'price',
                'required_params': ['period'],
                'optional_params': ['column'],
                'min_lookback': 'period',
                'description': 'Percentage decline from rolling peak',
                'formula': '(current - peak) / peak'
            },
            {
                'name': 'SPIKE',
                'category': 'volatility',
                'input_type': 'price',
                'required_params': ['period'],
                'optional_params': ['column'],
                'min_lookback': 'period',
                'description': 'Percentage spike from rolling minimum',
                'formula': '(current - min) / min'
            },
            {
                'name': 'CROSSES_ABOVE',
                'category': 'signal',
                'input_type': 'indicator',
                'required_params': ['series1', 'series2'],
                'optional_params': [],
                'min_lookback': 2,
                'description': 'Series 1 crosses above Series 2',
                'formula': 'series1[t-1] <= series2[t-1] AND series1[t] > series2[t]'
            },
            {
                'name': 'CROSSES_BELOW',
                'category': 'signal',
                'input_type': 'indicator',
                'required_params': ['series1', 'series2'],
                'optional_params': [],
                'min_lookback': 2,
                'description': 'Series 1 crosses below Series 2',
                'formula': 'series1[t-1] >= series2[t-1] AND series1[t] < series2[t]'
            }
        ]
        
        for method in methods:
            if isinstance(method['min_lookback'], str):
                lookback = 0
            else:
                lookback = method['min_lookback']
                
            self.cursor.execute("""
                INSERT OR IGNORE INTO calculation_methods 
                (method_name, method_category, input_type, required_params, 
                 optional_params, min_lookback, description, formula)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                method['name'],
                method['category'],
                method['input_type'],
                json.dumps(method['required_params']),
                json.dumps(method['optional_params']),
                lookback,
                method['description'],
                method['formula']
            ))
        
        self.conn.commit()
        logger.info(f"Inserted {len(methods)} calculation methods")
    
    def insert_symbols(self):
        """Insert tradeable symbols with metadata"""
        symbols = [
            ('ES', 'CME', 'future', 1),
            ('SPX', 'INDEX', 'index', 0),
            ('SPY', 'NYSE', 'etf', 1),
            ('VIX', 'CBOE', 'index', 0),
            ('VX', 'CFE', 'future', 1),
        ]
        
        for ticker, exchange, stype, tradeable in symbols:
            self.cursor.execute("""
                INSERT OR IGNORE INTO symbol (ticker, exchange, symbol_type, is_tradeable)
                VALUES (?, ?, ?, ?)
            """, (ticker, exchange, stype, tradeable))
        
        self.conn.commit()
        logger.info(f"Inserted {len(symbols)} symbols")
    
    def setup_strategies(self):
        """Setup strategies with flexible rule expressions"""
        
        # Strategy 1: Quick Panic ES - Auto highest frequency
        self.cursor.execute("""
            INSERT INTO strategy 
            (name, strategy_type, primary_symbol, secondary_symbols, 
             use_highest_frequency, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'Quick Panic ES',
            'mean_reversion',
            'ES',
            json.dumps(['SPX', 'VX', 'VIX']),
            1,
            'Buy ES dips during uptrend with favorable market conditions'
        ))
        quick_panic_id = self.cursor.lastrowid
        
        # Flexible rules using expressions
        rules = [
            # Filters
            ('filter', 'uptrend', 'Price above 50 SMA', 'PRICE > SMA(PRIMARY, 50)'),
            ('filter', 'low_vix', 'VIX below 25', 'INDICATOR(VIX) < 25'),
            ('filter', 'breadth_positive', 'Market breadth above 40%', 'INDICATOR(MARKET_BREADTH_DAILY) > 40'),
            
            # Entry conditions
            ('entry', 'es_decline', 'ES declined 1%+ from 10-day peak', 'DECLINE_FROM_PEAK(PRIMARY, 10) < -0.01'),
            ('entry', 'vx_calm', 'VX spike below 25%', 'SPIKE(VX, 10) < 0.25'),
            ('entry', 'vix_ratio_low', 'VIX/VXV ratio below 1.0', 'INDICATOR(VIX_VXV_RATIO) < 1.0'),
            ('entry', 'oversold_rsi', 'RSI below 30', 'RSI(PRIMARY, 2) < 30'),
            ('entry', 'trin_bullish', 'TRIN below 1.5', 'INDICATOR(TRIN_DAILY) < 1.5'),
            
            # Exit conditions
            ('exit', 'ema_cross', 'EMA 9 crosses below EMA 15', 'CROSSES_BELOW(EMA(PRIMARY, 9), EMA(PRIMARY, 15))'),
            ('exit', 'profit_target', 'Profit target reached', 'PROFIT_PCT >= PARAM(profit_target_percent)'),
            ('exit', 'stop_loss', 'Stop loss hit', 'LOSS_PCT >= PARAM(stop_loss_percent)'),
            ('exit', 'trailing_stop', 'Trailing stop hit', 'DRAWDOWN_FROM_PEAK >= PARAM(trailing_stop_percent)')
        ]
        
        priority = 1
        for rule_type, rule_name, rule_logic, expression in rules:
            self.cursor.execute("""
                INSERT INTO strategy_rules 
                (strategy_id, rule_type, rule_name, rule_logic, rule_expression, priority)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (quick_panic_id, rule_type, rule_name, rule_logic, expression, priority))
            priority += 1
        
        # Parameters
        params = [
            ('position_size', '1', 'numeric', 1, 10, 'Number of contracts'),
            ('stop_loss_percent', '1.0', 'numeric', 0.5, 2.0, 'Stop loss %'),
            ('trailing_stop_percent', '3.0', 'numeric', 1.0, 5.0, 'Trailing stop %'),
            ('profit_target_percent', '5.0', 'numeric', 2.0, 10.0, 'Profit target %'),
        ]
        
        for name, value, ptype, min_val, max_val, desc in params:
            self.cursor.execute("""
                INSERT INTO strategy_parameters 
                (strategy_id, param_name, param_value, param_type, min_value, max_value, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (quick_panic_id, name, value, ptype, min_val, max_val, desc))
        
        # Position sizing
        self.cursor.execute("""
            INSERT INTO position_sizing_rules 
            (strategy_id, sizing_type, sizing_params, max_position_size)
            VALUES (?, 'fixed', '{"contracts": 1}', 5)
        """, (quick_panic_id,))
        
        # Time filters
        self.cursor.execute("""
            INSERT INTO time_filters 
            (strategy_id, filter_type, filter_params)
            VALUES (?, 'trading_hours', '{"start": "09:30", "end": "16:00", "timezone": "America/New_York"}')
        """, (quick_panic_id,))
        
        self.conn.commit()
        logger.info("Strategies setup complete")
    
    def parse_datetime_flexible(self, date_str: str, time_str: str = ''):
        """Parse datetime with flexible format detection"""
        if time_str:
            datetime_str = f"{date_str} {time_str}"
        else:
            datetime_str = date_str
            
        datetime_str = datetime_str.strip()
        
        # List of formats to try (both US and European)
        formats = [
            # With time
            '%m/%d/%Y %H:%M:%S',  # US format
            '%d/%m/%Y %H:%M:%S',  # European format
            '%Y-%m-%d %H:%M:%S',  # ISO format
            '%d.%m.%Y %H:%M:%S',  # European with dots
            '%m.%d.%Y %H:%M:%S',  # US with dots
            # Date only
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d',
            '%d.%m.%Y',
            '%m.%d.%Y',
        ]
        
        # Try each format
        for fmt in formats:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        
        # If all formats fail, try pandas
        try:
            # Try with dayfirst=True for European dates
            dt = pd.to_datetime(datetime_str, dayfirst=True)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
            
        # Try with dayfirst=False for US dates
        try:
            dt = pd.to_datetime(datetime_str, dayfirst=False)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return None
    
    def migrate_price_data(self):
        """Migrate price data"""
        logger.info("Starting price data migration...")
        self._migrate_intraday_data()
        self._migrate_daily_data()
        
    def _migrate_intraday_data(self):
        """Migrate intraday price data"""
        timeframe_map = {
            '1 min': '1min',
            '5 min': '5min',
            '15 min': '15min',
            '30 min': '30min',
            '1 hour': '1hour'
        }
        
        total = 0
        for folder, timeframe in timeframe_map.items():
            folder_path = os.path.join(self.data_root, folder)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('.txt', '.csv')):
                        symbol = file.split('_')[0]
                        if symbol in ['ES', 'SPX', 'VIX', 'VX']:
                            filepath = os.path.join(folder_path, file)
                            count = self._load_price_file(filepath, symbol, timeframe)
                            total += count
        
        logger.info(f"Migrated {total:,} intraday records")
    
    def _migrate_daily_data(self):
        """Migrate daily price data"""
        daily_folder = os.path.join(self.data_root, 'Daily')
        if not os.path.exists(daily_folder):
            return
            
        files = {
            'ES_F_2000-2025.csv': 'ES',
            'SPX_max_1928-2025.csv': 'SPX',
            'VIX_1990-2025.csv': 'VIX',
            'VX_2013-2025.csv': 'VX'
        }
        
        total = 0
        for filename, symbol in files.items():
            filepath = os.path.join(daily_folder, filename)
            if os.path.exists(filepath):
                count = self._load_price_file(filepath, symbol, 'daily')
                total += count
                
        logger.info(f"Migrated {total:,} daily records")
    
    def _load_price_file(self, filepath, symbol, timeframe):
        """Load price file with better error handling"""
        try:
            # Get symbol_id
            self.cursor.execute("SELECT symbol_id FROM symbol WHERE ticker = ?", (symbol,))
            result = self.cursor.fetchone()
            if not result:
                return 0
            symbol_id = result['symbol_id']
            
            # Try reading with different formats
            df = None
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(filepath, sep=sep)
                    if len(df.columns) >= 5:
                        break
                except:
                    continue
            
            if df is None or df.empty:
                return 0
            
            # Parse based on column count
            if len(df.columns) == 7:
                df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
                # Create datetime column with flexible parsing
                df['datetime'] = df.apply(
                    lambda row: self.parse_datetime_flexible(str(row['date']), str(row['time'])), 
                    axis=1
                )
            elif len(df.columns) == 6:
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                df['datetime'] = df['date'].apply(lambda x: self.parse_datetime_flexible(str(x)))
            else:
                df.columns = ['datetime', 'open', 'high', 'low', 'close']
                df['datetime'] = df['datetime'].apply(lambda x: self.parse_datetime_flexible(str(x)))
                df['volume'] = 0
            
            # Remove rows where datetime parsing failed
            df = df.dropna(subset=['datetime'])
            
            # Convert columns to numeric, handling European decimal format
            for col in ['open', 'high', 'low', 'close']:
                # Replace comma with dot for European format
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle volume - convert to int, NaN becomes 0
            df['volume'] = df.get('volume', 0)
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)
                # Try to convert, if it fails set to 0
                try:
                    df['volume'] = df['volume'].astype(float).astype(int)
                except:
                    df['volume'] = 0
            
            # Clean data - remove invalid rows
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Only keep rows with positive prices
            df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
            
            # Batch insert
            batch_size = 10000
            total_inserted = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                records = [
                    (
                        row['datetime'],
                        symbol_id,
                        timeframe,
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row.get('volume', 0))
                    )
                    for _, row in batch.iterrows()
                    if row['datetime'] is not None
                ]
                
                if records:
                    self.cursor.executemany("""
                        INSERT OR IGNORE INTO price_data 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)
                    
                    total_inserted += len(records)
            
            self.conn.commit()
            logger.info(f"  {os.path.basename(filepath)}: {total_inserted:,} records")
            return total_inserted
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return 0
    
    def migrate_indicators(self):
        """Migrate market indicators"""
        logger.info("Starting indicator migration...")
        
        indicators = {
            'NAAIM.xlsx': 'NAAIM',
            'CNN Fear and Greed Index.xlsx': 'CNN_FEAR_GREED',
            'FedReserve stance.xlsx': 'FED_STANCE',
            'Buffett Indicator.xlsx': 'BUFFETT_INDICATOR',
            'VIX_VXV_Ratio.parquet': 'VIX_VXV_RATIO',
            'TRIN_2013-2025.csv': 'TRIN_DAILY',
            'Market breadth (_ above 50SMA)_2007-2025.csv': 'MARKET_BREADTH_DAILY',
            'VXV.xlsx': 'VXV'
        }
        
        search_paths = [
            self.data_root,
            os.path.join(self.data_root, '1 min'),
            os.path.join(self.data_root, 'Daily')
        ]
        
        for filename, indicator_name in indicators.items():
            found = False
            for search_path in search_paths:
                filepath = os.path.join(search_path, filename)
                if os.path.exists(filepath):
                    self._load_indicator_file(filepath, indicator_name)
                    found = True
                    break
            
            if not found:
                logger.warning(f"Indicator file not found: {filename}")
        
        logger.info("Indicator migration complete")
    
    def _load_indicator_file(self, filepath, indicator_name):
        """Load indicator file with improved parsing"""
        try:
            # Read based on extension
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath, engine='openpyxl')
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                # Try multiple CSV formats
                for sep in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(filepath, sep=sep)
                        if len(df.columns) >= 2:
                            break
                    except:
                        continue
            
            if df is None or df.empty:
                logger.warning(f"Empty file: {filepath}")
                return
            
            # Find date and value columns intelligently
            date_col = None
            value_col = None
            
            # Look for date column
            for col in df.columns:
                col_lower = str(col).lower()
                if any(term in col_lower for term in ['date', 'time', 'datetime']):
                    date_col = col
                    break
            
            # Value column is usually the second column or the one after date
            if date_col:
                cols = list(df.columns)
                date_idx = cols.index(date_col)
                if date_idx + 1 < len(cols):
                    value_col = cols[date_idx + 1]
                elif len(cols) == 2:
                    value_col = [c for c in cols if c != date_col][0]
            
            if not date_col or not value_col:
                logger.error(f"Cannot identify columns in {filepath}")
                return
            
            # Clear existing data for this indicator (use REPLACE)
            self.cursor.execute(
                "DELETE FROM market_indicator WHERE indicator_name = ?", 
                (indicator_name,)
            )
            
            # Process and insert
            records = []
            for _, row in df.iterrows():
                try:
                    # Use flexible datetime parsing
                    date_str = str(row[date_col])
                    parsed_date = self.parse_datetime_flexible(date_str)
                    
                    if parsed_date and pd.notna(row[value_col]):
                        records.append((
                            parsed_date,
                            indicator_name,
                            str(row[value_col])
                        ))
                except:
                    continue
            
            if records:
                # Bulk insert with REPLACE to handle duplicates
                self.cursor.executemany("""
                    INSERT OR REPLACE INTO market_indicator VALUES (?, ?, ?)
                """, records)
                
                self.conn.commit()
                logger.info(f"  {indicator_name}: {len(records):,} records")
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    
    def create_gitignore(self):
        """Create .gitignore file for the repository"""
        gitignore_content = """# Database files
*.db
*.db-journal
*.db-wal

# Data files
*.csv
*.xlsx
*.xls
*.parquet
*.txt

# Log files
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Backtest results
results/
output/
*.png
*.jpg
*.pdf

# Keep only code and documentation
!requirements.txt
!README.md
!*.py
"""
        
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content.strip())
        
        logger.info("Created .gitignore file")
    
    def verify_and_summarize(self):
        """Verify setup and show summary"""
        print("\n" + "="*60)
        print("DATABASE SETUP SUMMARY")
        print("="*60)
        
        # Check tables
        self.cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        tables = [row[0] for row in self.cursor.fetchall()]
        print(f"\nTables created: {len(tables)}")
        for table in tables:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = self.cursor.fetchone()[0]
            print(f"  {table}: {count:,} records")
        
        # Data summary
        print("\n" + "-"*40)
        print("PRICE DATA SUMMARY")
        print("-"*40)
        
        self.cursor.execute("""
            SELECT s.ticker, p.timeframe, 
                   COUNT(*) as records,
                   MIN(p.datetime) as start_date,
                   MAX(p.datetime) as end_date
            FROM price_data p
            JOIN symbol s ON p.symbol_id = s.symbol_id
            GROUP BY s.ticker, p.timeframe
            ORDER BY s.ticker, 
                     CASE p.timeframe 
                        WHEN '1min' THEN 1
                        WHEN '5min' THEN 2
                        WHEN '15min' THEN 3
                        WHEN '30min' THEN 4
                        WHEN '1hour' THEN 5
                        WHEN 'daily' THEN 6
                     END
        """)
        
        current_symbol = None
        for row in self.cursor.fetchall():
            if row[0] != current_symbol:
                current_symbol = row[0]
                print(f"\n{current_symbol}:")
            print(f"  {row[1]:>6}: {row[2]:>8,} records ({row[3][:10]} to {row[4][:10]})")
        
        print("\n" + "-"*40)
        print("READY FOR BACKTESTING!")
        print("-"*40)
        print(f"\nDatabase: {self.db_path}")
        print("Strategy: Quick Panic ES (auto highest frequency)")
        print("\nFor Git repository:")
        print("  - .gitignore created")
        print("  - Database files excluded")
        print("  - Only code files will be tracked")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Run complete flexible setup"""
    setup = FlexibleDatabaseSetup(DB_PATH, DATA_ROOT)
    
    try:
        # Create schema
        setup.create_flexible_schema()
        
        # Setup base data
        setup.setup_calculation_methods()
        setup.insert_symbols()
        setup.setup_strategies()
        
        # Migrate data
        setup.migrate_price_data()
        setup.migrate_indicators()
        
        # Create .gitignore
        setup.create_gitignore()
        
        # Verify
        setup.verify_and_summarize()
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        setup.close()


if __name__ == "__main__":
    main()