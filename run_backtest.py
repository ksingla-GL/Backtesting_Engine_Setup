"""
Simple script to run backtests using the flexible system
"""

import argparse
import sys
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from backtesting_engine import BacktestingEngine, BacktestConfig
from expression_parser import ExpressionParser
from calculate_indicators import IndicatorCalculator

def main():
    """Main function to run backtests"""
    
    parser = argparse.ArgumentParser(description='Run backtest for a strategy')
    parser.add_argument('--strategy_id', type=int, default=1, 
                       help='Strategy ID to backtest')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Starting capital')
    parser.add_argument('--db', type=str, default='backtesting_v2.db',
                       help='Database file path')
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError:
        print("Error: Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    config = BacktestConfig(
        initial_capital=args.capital,
        commission_per_trade=2.0,
        slippage_percent=0.01
    )
    
    print("="*60)
    print("BACKTESTING SYSTEM")
    print("="*60)
    print(f"Database: {args.db}")
    print(f"Strategy ID: {args.strategy_id}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Starting Capital: ${args.capital:,.2f}")
    print("="*60)
    
    try:
        print("\nInitializing backtesting engine...")
        engine = BacktestingEngine(args.db, args.strategy_id, config)
        
        print("Running backtest...")
        results = engine.run_backtest(start_date, end_date)
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nPerformance Summary:")
        print(f"  Total Trades: {results.get('total_trades', 0)}")
        print(f"  Win Rate: {results.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.1f}%")
        print(f"  Total Return: {results.get('total_return', 0):.1f}%")
        
        print(f"\nTrade Statistics:")
        print(f"  Winning Trades: {results.get('winning_trades', 0)}")
        print(f"  Losing Trades: {results.get('losing_trades', 0)}")
        print(f"  Average P&L: ${results.get('avg_pnl', 0):.2f}")
        print(f"  Average Win: ${results.get('avg_win', 0):.2f}")
        print(f"  Average Loss: ${results.get('avg_loss', 0):.2f}")
        
        print(f"\nCapital Summary:")
        print(f"  Starting Capital: ${config.initial_capital:,.2f}")
        print(f"  Final Capital: ${results.get('final_capital', config.initial_capital):,.2f}")
        print(f"  Total P&L: ${results.get('total_pnl', 0):,.2f}")
        
        trades_df = engine.get_trade_summary()
        if not trades_df.empty:
            print(f"\nLast 5 Trades:")
            print(trades_df.tail(5).to_string())
            
            output_file = f"trades_strategy_{args.strategy_id}_{args.start}_{args.end}.csv"
            trades_df.to_csv(output_file, index=False)
            print(f"\nTrades saved to: {output_file}")
        
        equity_df = engine.get_equity_curve()
        if not equity_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df['datetime'], equity_df['equity'], linewidth=2)
            plt.axhline(y=config.initial_capital, color='gray', 
                       linestyle='--', label='Initial Capital')
            plt.title(f'Equity Curve - Strategy {args.strategy_id}')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plot_file = f"equity_curve_strategy_{args.strategy_id}_{args.start}_{args.end}.png"
            plt.savefig(plot_file, dpi=300)
            print(f"Equity curve saved to: {plot_file}")
            
            plt.show()
        
        engine.close()
        
        print("\nBacktest completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
