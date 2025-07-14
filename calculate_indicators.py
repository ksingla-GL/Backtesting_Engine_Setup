"""
Technical Indicator Calculator for Backtesting Engine
Calculates indicators on-the-fly without storing in database
Fixed to handle market indicators properly
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Tuple, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """Calculates technical indicators on-demand for backtesting"""
    
    def __init__(self):
        self.cache = {}
        # Market indicators that come from database, not calculated
        self.market_indicators = {
            'VIX', 'VIX_VXV_RATIO', 'TRIN_DAILY', 'MARKET_BREADTH_DAILY',
            'NAAIM', 'CNN_FEAR_GREED', 'FED_STANCE', 'BUFFETT_INDICATOR', 'VXV'
        }
        
    def calculate(self, 
                 indicator_name: str, 
                 data: pd.DataFrame, 
                 params: Dict = None) -> pd.Series:
        """
        Main method to calculate any indicator
        
        Args:
            indicator_name: Name of indicator (e.g., 'ES_SMA_50', 'ES_RSI_2')
            data: DataFrame with OHLCV data
            params: Additional parameters for calculation
        
        Returns:
            pd.Series with calculated indicator values
        """
        parts = indicator_name.split('_')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid indicator name: {indicator_name}")
        
        # Extract calculation type and parameters
        if 'SMA' in indicator_name:
            period = int(parts[-1])
            return self.sma(data['close'], period)
            
        elif 'EMA' in indicator_name:
            period = int(parts[-1])
            return self.ema(data['close'], period)
            
        elif 'RSI' in indicator_name:
            period = int(parts[-1])
            return self.rsi(data['close'], period)
            
        elif 'PEAK' in indicator_name:
            period = params.get('period', 252) if params else 252
            return self.rolling_peak(data['close'], period)
            
        elif 'DECLINE_FROM_PEAK' in indicator_name:
            period = params.get('period', 10) if params else 10
            return self.decline_from_peak(data['close'], period)
            
        elif 'VX_SPIKE' in indicator_name or 'SPIKE' in indicator_name:
            period = params.get('period', 10) if params else 10
            return self.spike(data['close'], period)
            
        else:
            raise ValueError(f"Unknown indicator type: {indicator_name}")
    
    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period, min_periods=period).mean()
    
    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def rsi(self, series: pd.Series, period: int = 2) -> pd.Series:
        """
        Relative Strength Index
        Default period=2 for Quick Panic strategy
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        
        return rsi
    
    def rolling_peak(self, series: pd.Series, period: int) -> pd.Series:
        """Rolling maximum over specified period"""
        return series.rolling(window=period, min_periods=1).max()
    
    def decline_from_peak(self, series: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate percentage decline from rolling peak
        Used for entry signals in Quick Panic strategy
        """
        peak = self.rolling_peak(series, period)
        decline = (series - peak) / peak
        return decline
    
    def spike(self, series: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate spike percentage over period
        Spike = (current - min) / min
        """
        rolling_min = series.rolling(window=period, min_periods=1).min()
        spike = (series - rolling_min) / rolling_min
        return spike
    
    def calculate_required_indicators(self, 
                                    price_data: Dict[str, pd.DataFrame],
                                    market_indicators: pd.DataFrame,
                                    required_indicators: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Calculate specific indicators based on expression requirements
        
        Args:
            price_data: Dict with price data for various symbols/timeframes
            market_indicators: DataFrame with market indicators
            required_indicators: List of indicator names needed
        
        Returns:
            Dict with calculated indicators
        """
        indicators = {}
        
        # Find the main intraday data
        main_intraday_key = None
        for key in price_data.keys():
            if not key.endswith('_daily'):
                main_intraday_key = key
                break
        
        # Separate market indicators from calculated indicators
        calculated_needed = []
        for ind_name in required_indicators:
            if ind_name in self.market_indicators:
                # This is a market indicator, skip calculation
                continue
            else:
                calculated_needed.append(ind_name)
        
        # Process each calculated indicator
        for ind_name in calculated_needed:
            # Handle PRIMARY symbol replacement
            if ind_name.startswith('PRIMARY_'):
                # Extract the actual calculation type
                calc_parts = ind_name.replace('PRIMARY_', '').split('_')
                
                if len(calc_parts) >= 2:
                    calc_type = calc_parts[0]
                    
                    # Find appropriate data (use main intraday data)
                    if main_intraday_key and main_intraday_key in price_data:
                        data = price_data[main_intraday_key]
                        
                        # Calculate indicator based on type
                        if calc_type == 'SMA' and len(calc_parts) == 2:
                            period = int(calc_parts[1])
                            indicators[ind_name] = self.sma(data['close'], period)
                        elif calc_type == 'EMA' and len(calc_parts) == 2:
                            period = int(calc_parts[1])
                            indicators[ind_name] = self.ema(data['close'], period)
                        elif calc_type == 'RSI' and len(calc_parts) == 2:
                            period = int(calc_parts[1])
                            indicators[ind_name] = self.rsi(data['close'], period)
                        elif calc_type == 'DECLINE' and 'FROM' in ind_name and 'PEAK' in ind_name:
                            # Handle DECLINE_FROM_PEAK_10 format
                            period = int(calc_parts[-1])
                            indicators[ind_name] = self.decline_from_peak(data['close'], period)
            
            else:
                # Regular symbol-based indicators
                parts = ind_name.split('_')
                
                if len(parts) >= 3:
                    symbol = parts[0]
                    calc_type = parts[1]
                    
                    # Check if it's a spike calculation
                    if calc_type == 'SPIKE' and len(parts) == 3:
                        period = int(parts[2])
                        # Find appropriate data for the symbol
                        data_key = None
                        for key in price_data.keys():
                            if symbol in key:
                                data_key = key
                                break
                        
                        if data_key and data_key in price_data:
                            data = price_data[data_key]
                            indicators[ind_name] = self.spike(data['close'], period)
                    
                    # Other calculations can be added here
        
        # Add EMA crossover detection if needed
        if main_intraday_key:
            main_data = price_data[main_intraday_key]
            
            # Check if we have EMA indicators calculated
            ema9_key = None
            ema15_key = None
            
            for key in indicators:
                if 'EMA' in key and '9' in key:
                    ema9_key = key
                elif 'EMA' in key and '15' in key:
                    ema15_key = key
            
            if ema9_key and ema15_key and 'EMA_CROSS_DOWN' not in indicators:
                indicators['EMA_CROSS_DOWN'] = (
                    (indicators[ema9_key].shift(1) >= indicators[ema15_key].shift(1)) & 
                    (indicators[ema9_key] < indicators[ema15_key])
                )
        
        # Add market indicators
        indicators['market_indicators'] = market_indicators
        
        return indicators
    
    def align_indicators_to_timeframe(self, 
                                    indicators: Dict[str, Union[pd.Series, pd.DataFrame]], 
                                    target_timeframe: str,
                                    target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align all indicators to the target timeframe for backtesting
        """
        aligned_df = pd.DataFrame(index=target_index)
        
        for name, data in indicators.items():
            if name == 'market_indicators' and isinstance(data, pd.DataFrame):
                # Handle market indicators DataFrame
                for col in data.columns:
                    if col != 'datetime':
                        aligned_df[col] = data.set_index('datetime')[col].reindex(
                            target_index, method='ffill'
                        )
            elif isinstance(data, pd.Series):
                # Handle regular Series indicators
                if isinstance(data.index, pd.DatetimeIndex):
                    # Different frequency - need to align
                    if len(data) != len(target_index) or not data.index.equals(target_index):
                        aligned_df[name] = data.reindex(target_index, method='ffill')
                    else:
                        aligned_df[name] = data
                else:
                    # Same length, just assign
                    if len(data) == len(target_index):
                        aligned_df[name] = data.values
                        
        return aligned_df
    
    def get_lookback_required(self, indicator_expressions: List[str]) -> int:
        """
        Calculate the maximum lookback period required for a set of indicators
        """
        max_lookback = 0
        
        for expr in indicator_expressions:
            # Extract periods from expressions
            import re
            
            # Find SMA/EMA/RSI patterns
            pattern = r'(SMA|EMA|RSI|DECLINE_FROM_PEAK|SPIKE)\([^,]+,\s*(\d+)\)'
            matches = re.findall(pattern, expr)
            
            for func, period in matches:
                period_int = int(period)
                if func in ['SMA', 'DECLINE_FROM_PEAK', 'SPIKE']:
                    max_lookback = max(max_lookback, period_int)
                elif func == 'EMA':
                    max_lookback = max(max_lookback, period_int * 2)
                elif func == 'RSI':
                    max_lookback = max(max_lookback, period_int + 1)
        
        # Add buffer
        return max_lookback + 10

if __name__ == "__main__":
    # Test indicator calculations
    calc = IndicatorCalculator()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Test calculations
    sma_50 = calc.calculate('ES_SMA_50', test_data)
    rsi_2 = calc.calculate('ES_RSI_2', test_data)
    decline = calc.calculate('ES_DECLINE_FROM_PEAK', test_data, {'period': 10})
    
    print(f"SMA(50) last value: {sma_50.iloc[-1]:.2f}")
    print(f"RSI(2) last value: {rsi_2.iloc[-1]:.2f}")
    print(f"Decline from peak: {decline.iloc[-1]:.4f}")