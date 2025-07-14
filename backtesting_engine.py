"""
Main Backtesting Engine for Zero-Code Strategy Testing
With flexible expression support
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass
from calculate_indicators import IndicatorCalculator
from expression_parser import ExpressionParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade"""
    entry_datetime: datetime
    exit_datetime: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: int
    entry_conditions: Dict
    exit_reason: Optional[str]
    pnl: Optional[float]
    pnl_percent: Optional[float]
    max_profit: float = 0.0
    max_loss: float = 0.0

@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 100000.0
    commission_per_trade: float = 2.0
    slippage_percent: float = 0.01
    data_frequency: str = 'auto'
    timezone: str = 'America/New_York'

class BacktestingEngine:
    """Main backtesting engine for zero-code strategy execution"""
    
    def __init__(self, db_path: str, strategy_id: int, config: BacktestConfig = None):
        self.db_path = db_path
        self.strategy_id = strategy_id
        self.config = config or BacktestConfig()
        
        # Initialize components
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.indicator_calc = IndicatorCalculator()
        self.expression_parser = ExpressionParser(self.indicator_calc)
        
        # Load strategy
        self.strategy = self._load_strategy()
        self.rules = self._load_strategy_rules()
        self.parameters = self._load_strategy_parameters()
        
        # Trading state
        self.current_position = None
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        
        logger.info(f"Initialized backtesting engine for strategy: {self.strategy['name']}")
    
    def _load_strategy(self) -> Dict:
        """Load strategy definition from database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM strategy WHERE strategy_id = ?
        """, (self.strategy_id,))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Strategy {self.strategy_id} not found")
            
        return dict(row)
    
    def _load_strategy_rules(self) -> Dict[str, List[Dict]]:
        """Load and organize strategy rules by type"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM strategy_rules 
            WHERE strategy_id = ? AND is_active = 1
            ORDER BY rule_type, priority
        """, (self.strategy_id,))
        
        rules = {'filter': [], 'entry': [], 'exit': []}
        
        for row in cursor.fetchall():
            rule = dict(row)
            rules[rule['rule_type']].append(rule)
            
        return rules
    
    def _load_strategy_parameters(self) -> Dict[str, Any]:
        """Load strategy parameters"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT param_name, param_value, param_type 
            FROM strategy_parameters 
            WHERE strategy_id = ?
        """, (self.strategy_id,))
        
        params = {}
        for row in cursor.fetchall():
            value = row['param_value']
            if row['param_type'] == 'numeric':
                value = float(value)
            elif row['param_type'] == 'boolean':
                value = value.lower() == 'true'
            params[row['param_name']] = value
            
        # Set defaults if not specified
        defaults = {
            'position_size': 1,
            'stop_loss_percent': 1.0,
            'trailing_stop_percent': 3.0,
            'profit_target_percent': 5.0
        }
        
        for key, default_value in defaults.items():
            if key not in params:
                params[key] = default_value
                
        return params
    
    def _load_price_data(self, symbol: str, timeframe: str, 
                        start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load price data from database"""
        query = """
            SELECT p.datetime, p.open, p.high, p.low, p.close, p.volume
            FROM price_data p
            JOIN symbol s ON p.symbol_id = s.symbol_id
            WHERE s.ticker = ? AND p.timeframe = ?
            AND p.datetime >= ? AND p.datetime <= ?
            ORDER BY p.datetime
        """
        
        df = pd.read_sql_query(
            query, 
            self.conn,
            params=(symbol, timeframe, start_date.strftime('%Y-%m-%d'), 
                   end_date.strftime('%Y-%m-%d')),
            parse_dates=['datetime'],
            index_col='datetime'
        )
        
        return df
    
    def _get_highest_frequency_data(self, symbol: str, start_date: datetime, 
                                   end_date: datetime) -> Tuple[str, pd.DataFrame]:
        """
        Automatically detect and load the highest frequency data available
        
        Returns: (timeframe, data)
        """
        timeframes = ['1min', '5min', '15min', '30min', '1hour', 'daily']
        
        cursor = self.conn.cursor()
        
        for timeframe in timeframes:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM price_data p
                JOIN symbol s ON p.symbol_id = s.symbol_id
                WHERE s.ticker = ? AND p.timeframe = ?
                AND p.datetime >= ? AND p.datetime <= ?
            """, (symbol, timeframe, start_date.strftime('%Y-%m-%d'), 
                 end_date.strftime('%Y-%m-%d')))
            
            count = cursor.fetchone()['count']
            
            if count > 0:
                logger.info(f"Using {timeframe} data for {symbol} ({count} bars)")
                data = self._load_price_data(symbol, timeframe, start_date, end_date)
                return timeframe, data
        
        raise ValueError(f"No data found for {symbol} in date range")
    
    def _load_market_indicators(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load market indicators from database"""
        query = """
            SELECT datetime, indicator_name, value
            FROM market_indicator
            WHERE datetime >= ? AND datetime <= ?
            ORDER BY datetime, indicator_name
        """
        
        df = pd.read_sql_query(
            query,
            self.conn,
            params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            parse_dates=['datetime']
        )
        
        if not df.empty:
            df_pivot = df.pivot(index='datetime', columns='indicator_name', values='value')
            numeric_cols = ['VIX', 'VIX_VXV_RATIO', 'TRIN_DAILY', 'MARKET_BREADTH_DAILY',
                           'NAAIM', 'CNN_FEAR_GREED', 'BUFFETT_INDICATOR']
            for col in numeric_cols:
                if col in df_pivot.columns:
                    df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')
            return df_pivot
        else:
            return pd.DataFrame()
    
    def _evaluate_rule_expression(self, rule: Dict, context: Dict) -> Tuple[bool, Dict]:
        """Evaluate a rule using the expression parser"""
        
        if 'rule_expression' in rule and rule['rule_expression']:
            return self.expression_parser.evaluate(rule['rule_expression'], context)
        else:
            # Fallback to old logic if no expression
            return False, {'error': 'No rule expression'}
    
    def _check_entry_conditions(self, context: Dict) -> Tuple[bool, List[Dict]]:
        """Check if all entry conditions are met"""
        
        # First check filters
        filter_results = []
        for rule in self.rules['filter']:
            passed, details = self._evaluate_rule_expression(rule, context)
            details['rule_name'] = rule['rule_name']
            filter_results.append(details)
            if not passed:
                return False, filter_results
        
        # Then check entry rules
        entry_results = []
        all_passed = True
        for rule in self.rules['entry']:
            passed, details = self._evaluate_rule_expression(rule, context)
            details['rule_name'] = rule['rule_name']
            entry_results.append(details)
            if not passed:
                all_passed = False
                
        return all_passed, filter_results + entry_results
    
    def _check_exit_conditions(self, context: Dict, entry_price: float) -> Tuple[bool, str, float]:
        """
        Check if any exit condition is met
        Returns: (should_exit, exit_reason, exit_price)
        """
        
        row_data = context['current_bar']
        
        # Get bar extremes
        bar_high = row_data['high']
        bar_low = row_data['low']
        bar_open = row_data['open']
        bar_close = row_data['close']
        
        # Calculate exit levels
        stop_loss_pct = self.parameters.get('stop_loss_percent', 1.0) / 100
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        profit_target_pct = self.parameters.get('profit_target_percent', 5.0) / 100
        profit_target_price = entry_price * (1 + profit_target_pct)
        
        # Trailing stop
        trailing_stop_price = None
        if self.current_position and self.current_position.max_profit > 0:
            trailing_stop_pct = self.parameters.get('trailing_stop_percent', 3.0) / 100
            peak_price = entry_price + self.current_position.max_profit / self.current_position.position_size
            trailing_stop_price = peak_price * (1 - trailing_stop_pct)
        
        # Check if both stop and target could be hit in same bar
        stop_hit = bar_low <= stop_loss_price
        target_hit = bar_high >= profit_target_price
        trailing_hit = trailing_stop_price and bar_low <= trailing_stop_price
        
        if (stop_hit or trailing_hit) and target_hit:
            # Both could be hit - estimate sequence
            sequence = self._estimate_intrabar_sequence(row_data, entry_price)
            
            if sequence == 'low_first':
                exit_price = max(stop_loss_price, trailing_stop_price or 0, bar_low)
                return True, "Stop loss (low first)", exit_price
            else:
                exit_price = min(profit_target_price, bar_high)
                return True, f"Profit target at {profit_target_pct*100}%", exit_price
        
        # Single exit condition checks
        if stop_hit:
            exit_price = max(stop_loss_price, bar_low)
            return True, f"Stop loss at {stop_loss_pct*100}%", exit_price
            
        if trailing_hit:
            exit_price = max(trailing_stop_price, bar_low)
            return True, f"Trailing stop at {trailing_stop_pct*100}%", exit_price
            
        if target_hit:
            exit_price = min(profit_target_price, bar_high)
            return True, f"Profit target at {profit_target_pct*100}%", exit_price
        
        # Check exit rules using expressions
        for rule in self.rules['exit']:
            passed, details = self._evaluate_rule_expression(rule, context)
            if passed:
                return True, rule['rule_name'], bar_close
                
        return False, None, None
    
    def _estimate_intrabar_sequence(self, row: pd.Series, entry_price: float) -> str:
        """
        Estimate whether high or low was likely hit first in a bar
        Based on gap analysis and open position within range
        """
        if hasattr(self, 'prev_close') and self.prev_close:
            prev_close = self.prev_close
        else:
            prev_close = row['open']
        
        gap_pct = (row['open'] - prev_close) / prev_close
        gap_up = gap_pct > 0.005
        gap_down = gap_pct < -0.005
        
        bar_range = row['high'] - row['low']
        if bar_range > 0:
            open_position = (row['open'] - row['low']) / bar_range
        else:
            open_position = 0.5
        
        if bar_range > 0:
            entry_position = (entry_price - row['low']) / bar_range
        else:
            entry_position = 0.5
        
        if gap_up and open_position > 0.66:
            return 'high_first'
        elif gap_down and open_position < 0.33:
            return 'low_first'
        elif open_position < 0.2:
            return 'low_first'
        elif open_position > 0.8:
            return 'high_first'
        elif entry_position > 0.7:
            return 'low_first'
        else:
            return 'low_first'
    
    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
        """Run the complete backtest"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Get primary symbol
        primary_symbol = self.strategy.get('primary_symbol')
        if not primary_symbol:
            primary_symbol = 'ES'  # Default
            
        logger.info(f"Using primary symbol: {primary_symbol}")
        
        # Get highest frequency data if configured
        if self.strategy.get('use_highest_frequency', True):
            main_timeframe, main_data = self._get_highest_frequency_data(
                primary_symbol, start_date, end_date
            )
        else:
            main_timeframe = self.strategy.get('fallback_timeframe', '5min')
            main_data = self._load_price_data(primary_symbol, main_timeframe, start_date, end_date)
        
        # Update config
        self.config.data_frequency = main_timeframe
        
        # Build price data dictionary
        main_data_key = f'{primary_symbol}_{main_timeframe}'
        price_data = {
            main_data_key: main_data,
            f'{primary_symbol}_daily': self._load_price_data(primary_symbol, 'daily', 
                                            start_date - timedelta(days=100), end_date)
        }
        
        # Load secondary symbols if needed
        secondary_symbols = json.loads(self.strategy.get('secondary_symbols', '[]'))
        for symbol in secondary_symbols:
            if symbol in ['SPX', 'VX', 'VIX']:
                price_data[f'{symbol}_daily'] = self._load_price_data(
                    symbol, 'daily', start_date - timedelta(days=100), end_date
                )
        
        # Load market indicators
        logger.info("Loading market indicators...")
        market_indicators = self._load_market_indicators(
            start_date - timedelta(days=20), end_date
        )
        
        # Get required indicators from expressions
        all_expressions = []
        for rule_type in self.rules:
            for rule in self.rules[rule_type]:
                if rule.get('rule_expression'):
                    all_expressions.append(rule['rule_expression'])
        
        required_indicators = self.expression_parser.get_required_indicators(all_expressions)
        logger.info(f"Required indicators: {required_indicators}")
        
        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        calculated_indicators = self.indicator_calc.calculate_required_indicators(
            price_data, market_indicators, list(required_indicators)
        )
        
        # Align all indicators to main timeframe
        aligned_indicators = self.indicator_calc.align_indicators_to_timeframe(
            calculated_indicators, main_timeframe, main_data.index
        )
        
        # Combine price data with indicators
        backtest_data = pd.concat([main_data, aligned_indicators], axis=1)
        
        # Remove rows with insufficient data
        required_cols = [col for col in aligned_indicators.columns 
                        if col in required_indicators or col in ['VIX', 'VIX_VXV_RATIO']]
        if required_cols:
            backtest_data = backtest_data.dropna(subset=required_cols)
        
        logger.info(f"Running backtest on {len(backtest_data)} bars at {main_timeframe} frequency")
        
        # Initialize tracking
        self.prev_close = None
        previous_values = {}
        
        # Main backtest loop
        for idx, (timestamp, row) in enumerate(backtest_data.iterrows()):
            
            # Skip if outside market hours
            if self._should_skip_bar(timestamp):
                self.prev_close = row['close']
                continue
            
            # Create context for expression evaluation
            context = {
                'current_bar': row,
                'indicators': row,
                'parameters': self.parameters,
                'position': self._get_position_info() if self.current_position else None,
                'primary_symbol': primary_symbol,
                'previous_values': previous_values
            }
            
            # Check for exit if in position
            if self.current_position:
                should_exit, exit_reason, exit_price = self._check_exit_conditions(
                    context, self.current_position.entry_price
                )
                
                if should_exit:
                    self._exit_position(timestamp, exit_price, exit_reason)
            
            # Check for entry if not in position
            else:
                should_enter, entry_conditions = self._check_entry_conditions(context)
                
                if should_enter:
                    self._enter_position(timestamp, row['close'], entry_conditions)
            
            # Update position tracking
            if self.current_position:
                current_max_pnl = (row['high'] - self.current_position.entry_price) * \
                                self.current_position.position_size
                self.current_position.max_profit = max(
                    self.current_position.max_profit, current_max_pnl
                )
                
                current_min_pnl = (row['low'] - self.current_position.entry_price) * \
                                self.current_position.position_size
                self.current_position.max_loss = min(
                    self.current_position.max_loss, current_min_pnl
                )
            
            # Update equity curve
            position_value = 0
            if self.current_position:
                position_value = (row['close'] - self.current_position.entry_price) * \
                               self.current_position.position_size
            
            total_equity = self.current_capital + position_value
            self.equity_curve.append({
                'datetime': timestamp,
                'equity': total_equity,
                'in_position': self.current_position is not None,
                'cash': self.current_capital,
                'position_value': position_value
            })
            
            # Update tracking
            self.prev_close = row['close']
            # Store previous values for crossover detection
            for ind in required_indicators:
                if ind in row:
                    previous_values[ind] = row[ind]
        
        # Close any open position at end
        if self.current_position:
            final_bar = backtest_data.iloc[-1]
            self._exit_position(
                backtest_data.index[-1], 
                final_bar['close'],
                'End of backtest'
            )
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        results['data_frequency'] = main_timeframe
        results['total_bars_processed'] = len(backtest_data)
        
        # Save results to database
        self._save_results(results, start_date, end_date)
        
        logger.info("Backtest completed successfully")
        return results
    
    def _get_position_info(self) -> Dict:
        """Get current position information for context"""
        if not self.current_position:
            return None
            
        return {
            'entry_price': self.current_position.entry_price,
            'entry_datetime': self.current_position.entry_datetime,
            'position_size': self.current_position.position_size,
            'peak_price': self.current_position.entry_price + 
                         self.current_position.max_profit / self.current_position.position_size
        }
    
    def _should_skip_bar(self, timestamp: datetime) -> bool:
        """Check if bar should be skipped based on time filters"""
        if timestamp.time() < pd.Timestamp('09:30').time() or \
           timestamp.time() > pd.Timestamp('16:00').time():
            return True
            
        if timestamp.weekday() >= 5:
            return True
            
        return False
    
    def _enter_position(self, timestamp: datetime, price: float, conditions: List[Dict]):
        """Enter a new position"""
        position_size = int(self.parameters.get('position_size', 1))
        
        entry_price = price * (1 + self.config.slippage_percent / 100)
        
        self.current_position = Trade(
            entry_datetime=timestamp,
            exit_datetime=None,
            entry_price=entry_price,
            exit_price=None,
            position_size=position_size,
            entry_conditions={'conditions': conditions},
            exit_reason=None,
            pnl=None,
            pnl_percent=None
        )
        
        self.current_capital -= self.config.commission_per_trade
        
        logger.info(f"Entered position at {timestamp}: {position_size} @ {entry_price:.2f}")
    
    def _exit_position(self, timestamp: datetime, exit_price: float, reason: str):
        """Exit current position with specific exit price"""
        if not self.current_position:
            return
            
        exit_price_with_slippage = exit_price * (1 - self.config.slippage_percent / 100)
        
        pnl = (exit_price_with_slippage - self.current_position.entry_price) * \
              self.current_position.position_size
        pnl_percent = (exit_price_with_slippage / self.current_position.entry_price - 1) * 100
        
        self.current_position.exit_datetime = timestamp
        self.current_position.exit_price = exit_price_with_slippage
        self.current_position.exit_reason = reason
        self.current_position.pnl = pnl
        self.current_position.pnl_percent = pnl_percent
        
        self.current_capital += pnl - self.config.commission_per_trade
        
        self.trades.append(self.current_position)
        self.current_position = None
        
        logger.info(f"Exited position at {timestamp}: {exit_price_with_slippage:.2f}, "
                   f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%), Reason: {reason}")
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'message': 'No trades executed'
            }
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.current_capital / self.config.initial_capital - 1) * 100
        
        if len(self.equity_curve) > 1:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['returns'] = equity_df['equity'].pct_change()
            sharpe_ratio = np.sqrt(252) * (equity_df['returns'].mean() / 
                                          equity_df['returns'].std()) if equity_df['returns'].std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        equity_df = pd.DataFrame(self.equity_curve)
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        avg_pnl = np.mean([t.pnl for t in self.trades])
        
        durations = [(t.exit_datetime - t.entry_datetime).total_seconds() / 3600 
                    for t in self.trades if t.exit_datetime]
        avg_duration = np.mean(durations) if durations else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_duration_hours': avg_duration,
            'final_capital': self.current_capital,
            'commissions_paid': total_trades * 2 * self.config.commission_per_trade
        }
        
        return metrics
    
    def _save_results(self, results: Dict, start_date: datetime, end_date: datetime):
        """Save backtest results to database with enhanced metrics"""
        cursor = self.conn.cursor()
        
        config_json = json.dumps({
            'initial_capital': self.config.initial_capital,
            'commission_per_trade': self.config.commission_per_trade,
            'slippage_percent': self.config.slippage_percent,
            'data_frequency': self.config.data_frequency,
            'parameters': self.parameters
        })
        
        cursor.execute("""
            INSERT INTO backtest (strategy_id, start_date, end_date, run_timestamp, 
                                config, data_frequency_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (self.strategy_id, start_date, end_date, datetime.now(), 
              config_json, self.config.data_frequency))
        
        backtest_id = cursor.lastrowid
        
        for trade in self.trades:
            entry_json = json.dumps(trade.entry_conditions)
            
            cursor.execute("""
                INSERT INTO trade (
                    backtest_id, symbol_id, entry_datetime, exit_datetime,
                    entry_price, exit_price, pnl, pnl_percent,
                    entry_conditions, exit_reason
                ) VALUES (
                    ?, (SELECT symbol_id FROM symbol WHERE ticker = ?),
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                backtest_id, self.strategy.get('primary_symbol', 'ES'),
                trade.entry_datetime, trade.exit_datetime,
                trade.entry_price, trade.exit_price, trade.pnl, trade.pnl_percent,
                entry_json, trade.exit_reason
            ))
        
        cursor.execute("""
            INSERT INTO backtest_metrics (
                backtest_id, total_trades, win_rate, profit_factor,
                sharpe_ratio, max_drawdown, total_return,
                avg_pnl, avg_win, avg_loss, avg_duration_hours,
                winning_trades, losing_trades, total_pnl,
                final_capital, commissions_paid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            backtest_id, 
            results.get('total_trades', 0), 
            results.get('win_rate', 0),
            results.get('profit_factor', 0), 
            results.get('sharpe_ratio', 0),
            results.get('max_drawdown', 0), 
            results.get('total_return', 0),
            results.get('avg_pnl', 0),
            results.get('avg_win', 0),
            results.get('avg_loss', 0),
            results.get('avg_duration_hours', 0),
            results.get('winning_trades', 0),
            results.get('losing_trades', 0),
            results.get('total_pnl', 0),
            results.get('final_capital', self.config.initial_capital),
            results.get('commissions_paid', 0)
        ))
        
        self.conn.commit()
        logger.info(f"Results saved to database with backtest_id: {backtest_id}")
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
            
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'entry_time': trade.entry_datetime,
                'exit_time': trade.exit_datetime,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_size': trade.position_size,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason,
                'duration_hours': (trade.exit_datetime - trade.entry_datetime).total_seconds() / 3600
                                if trade.exit_datetime else None
            })
            
        return pd.DataFrame(trade_data)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    DB_PATH = r'C:\Users\kshit\Desktop\Upwork\Kyrg_Mean_Reversion\backtesting_v2.db'
    STRATEGY_ID = 1
    
    config = BacktestConfig(
        initial_capital=100000,
        commission_per_trade=2.0,
        slippage_percent=0.01
    )
    
    engine = BacktestingEngine(DB_PATH, STRATEGY_ID, config)
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    results = engine.run_backtest(start_date, end_date)
    
    print("\n=== Backtest Results ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    engine.close()
