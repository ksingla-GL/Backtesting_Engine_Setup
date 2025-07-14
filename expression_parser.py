"""
Expression Parser for Flexible Rule Evaluation
Handles expressions like: SMA(ES, 50) > PRICE
Fixed to handle PRIMARY symbol and market indicators properly
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional
import json

class ExpressionParser:
    """Parse and evaluate flexible trading rule expressions"""
    
    def __init__(self, indicator_calculator):
        self.indicator_calculator = indicator_calculator
        self.cache = {}
        # Market indicators that come from database
        self.market_indicators = {
            'VIX', 'VIX_VXV_RATIO', 'TRIN_DAILY', 'MARKET_BREADTH_DAILY',
            'NAAIM', 'CNN_FEAR_GREED', 'FED_STANCE', 'BUFFETT_INDICATOR', 'VXV'
        }
        
    def parse_expression(self, expression: str) -> Dict:
        """
        Parse expression into components
        Examples:
            'PRICE > SMA(PRIMARY, 50)'
            'RSI(ES, 2) < 30'
            'CROSSES_BELOW(EMA(PRIMARY, 9), EMA(PRIMARY, 15))'
        """
        expression = ' '.join(expression.split())
        
        operators = ['>=', '<=', '>', '<', '==', '!=']
        operator = None
        for op in operators:
            if f' {op} ' in expression:
                operator = op
                break
        
        if operator:
            left, right = expression.split(f' {operator} ', 1)
            return {
                'left': self._parse_component(left.strip()),
                'operator': operator,
                'right': self._parse_component(right.strip())
            }
        else:
            return {
                'component': self._parse_component(expression),
                'operator': 'evaluate'
            }
    
    def _parse_component(self, component: str) -> Dict:
        """Parse a single component of expression"""
        
        func_match = re.match(r'(\w+)\((.*)\)', component)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            args = [arg.strip() for arg in args_str.split(',')]
            
            return {
                'type': 'function',
                'name': func_name,
                'args': args
            }
        
        if component.startswith('INDICATOR('):
            ind_name = component[10:-1]
            return {
                'type': 'indicator',
                'name': ind_name
            }
        
        if component.startswith('PARAM('):
            param_name = component[6:-1]
            return {
                'type': 'parameter',
                'name': param_name
            }
        
        try:
            value = float(component)
            return {
                'type': 'value',
                'value': value
            }
        except ValueError:
            pass
        
        if component in ['PRICE', 'PRIMARY', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
            return {
                'type': 'keyword',
                'name': component
            }
        
        if component in ['PROFIT_PCT', 'LOSS_PCT', 'DRAWDOWN_FROM_PEAK']:
            return {
                'type': 'position_metric',
                'name': component
            }
        
        return {
            'type': 'symbol',
            'name': component
        }
    
    def evaluate(self, expression: str, context: Dict) -> Tuple[bool, Any]:
        """
        Evaluate expression in given context
        
        Args:
            expression: Rule expression to evaluate
            context: Dict containing:
                - current_bar: Current price bar
                - indicators: DataFrame of indicators
                - parameters: Strategy parameters
                - position: Current position info
                - primary_symbol: Primary trading symbol
                
        Returns:
            Tuple of (result, details)
        """
        try:
            parsed = self.parse_expression(expression)
            
            if 'operator' in parsed and parsed['operator'] == 'evaluate':
                value = self._evaluate_component(parsed['component'], context)
                return bool(value), {'value': value}
            
            left_value = self._evaluate_component(parsed['left'], context)
            right_value = self._evaluate_component(parsed['right'], context)
            
            operator = parsed['operator']
            if operator == '>':
                result = left_value > right_value
            elif operator == '<':
                result = left_value < right_value
            elif operator == '>=':
                result = left_value >= right_value
            elif operator == '<=':
                result = left_value <= right_value
            elif operator == '==':
                result = left_value == right_value
            elif operator == '!=':
                result = left_value != right_value
            else:
                result = False
            
            return result, {
                'left_value': left_value,
                'operator': operator,
                'right_value': right_value,
                'result': result
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _evaluate_component(self, component: Dict, context: Dict) -> Any:
        """Evaluate a single component"""
        
        if component['type'] == 'value':
            return component['value']
            
        elif component['type'] == 'keyword':
            keyword = component['name']
            current_bar = context['current_bar']
            
            if keyword == 'PRICE':
                return current_bar['close']
            elif keyword == 'PRIMARY':
                return context['primary_symbol']
            elif keyword in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
                return current_bar[keyword.lower()]
                
        elif component['type'] == 'indicator':
            ind_name = component['name']
            if ind_name in context['indicators']:
                return context['indicators'][ind_name]
            else:
                raise ValueError(f"Indicator {ind_name} not found")
                
        elif component['type'] == 'parameter':
            param_name = component['name']
            if param_name in context['parameters']:
                return context['parameters'][param_name]
            else:
                raise ValueError(f"Parameter {param_name} not found")
                
        elif component['type'] == 'position_metric':
            metric_name = component['name']
            position = context.get('position')
            
            if not position:
                return 0
                
            if metric_name == 'PROFIT_PCT':
                if position.get('entry_price'):
                    current_price = context['current_bar']['close']
                    return (current_price / position['entry_price'] - 1) * 100
                return 0
                
            elif metric_name == 'LOSS_PCT':
                if position.get('entry_price'):
                    current_price = context['current_bar']['close']
                    return (1 - current_price / position['entry_price']) * 100
                return 0
                
            elif metric_name == 'DRAWDOWN_FROM_PEAK':
                if position.get('peak_price') and position.get('entry_price'):
                    current_price = context['current_bar']['close']
                    peak = max(position['peak_price'], position['entry_price'])
                    return (1 - current_price / peak) * 100
                return 0
                
        elif component['type'] == 'function':
            return self._evaluate_function(component, context)
            
        elif component['type'] == 'symbol':
            symbol = component['name']
            if symbol == context['primary_symbol']:
                return context['current_bar']['close']
            else:
                raise ValueError(f"Multi-symbol lookup not implemented yet")
                
        return 0
    
    def _evaluate_function(self, func_component: Dict, context: Dict) -> Any:
        """Evaluate function calls like SMA(ES, 50)"""
        
        func_name = func_component['name']
        args = func_component['args']
        
        # Replace PRIMARY with actual symbol in the indicator name
        processed_args = []
        for arg in args:
            if arg == 'PRIMARY':
                processed_args.append(context['primary_symbol'])
            else:
                processed_args.append(arg)
        
        # Handle calculation functions
        if func_name in ['SMA', 'EMA', 'RSI', 'DECLINE_FROM_PEAK', 'SPIKE']:
            # Build the indicator name that should exist in context
            symbol = processed_args[0]
            period = processed_args[1]
            
            # Special handling for PRIMARY symbol
            if symbol == context['primary_symbol']:
                # Use PRIMARY_ prefix for primary symbol indicators
                if func_name == 'DECLINE_FROM_PEAK':
                    indicator_name = f"PRIMARY_DECLINE_FROM_PEAK_{period}"
                else:
                    indicator_name = f"PRIMARY_{func_name}_{period}"
            else:
                # Regular symbol indicators
                indicator_name = f"{symbol}_{func_name}_{period}"
            
            # Get from context
            if indicator_name in context['indicators']:
                return context['indicators'][indicator_name]
            else:
                # Try without PRIMARY prefix as fallback
                fallback_name = f"{context['primary_symbol']}_{func_name}_{period}"
                if fallback_name in context['indicators']:
                    return context['indicators'][fallback_name]
                else:
                    raise ValueError(f"Indicator {indicator_name} not found in context")
                
        elif func_name == 'CROSSES_ABOVE':
            series1_val = self._evaluate_component(
                self._parse_component(processed_args[0]), context
            )
            series2_val = self._evaluate_component(
                self._parse_component(processed_args[1]), context
            )
            
            prev1 = context.get('previous_values', {}).get(processed_args[0], series1_val)
            prev2 = context.get('previous_values', {}).get(processed_args[1], series2_val)
            
            return prev1 <= prev2 and series1_val > series2_val
            
        elif func_name == 'CROSSES_BELOW':
            series1_val = self._evaluate_component(
                self._parse_component(processed_args[0]), context
            )
            series2_val = self._evaluate_component(
                self._parse_component(processed_args[1]), context
            )
            
            prev1 = context.get('previous_values', {}).get(processed_args[0], series1_val)
            prev2 = context.get('previous_values', {}).get(processed_args[1], series2_val)
            
            return prev1 >= prev2 and series1_val < series2_val
            
        return 0
    
    def get_required_indicators(self, expressions: list) -> set:
        """
        Extract all required indicators from a list of expressions
        
        Returns set of indicator names needed
        """
        required = set()
        
        for expr in expressions:
            # Find all function calls
            func_pattern = r'(\w+)\(([^)]+)\)'
            matches = re.findall(func_pattern, expr)
            
            for func_name, args_str in matches:
                if func_name in ['SMA', 'EMA', 'RSI', 'DECLINE_FROM_PEAK', 'SPIKE']:
                    args = [arg.strip() for arg in args_str.split(',')]
                    # Handle PRIMARY symbol
                    symbol = args[0]
                    if len(args) > 1:
                        period = args[1]
                        if symbol == 'PRIMARY':
                            # Use PRIMARY_ prefix for these
                            if func_name == 'DECLINE_FROM_PEAK':
                                indicator_name = f"PRIMARY_DECLINE_FROM_PEAK_{period}"
                            else:
                                indicator_name = f"PRIMARY_{func_name}_{period}"
                        else:
                            indicator_name = f"{symbol}_{func_name}_{period}"
                        required.add(indicator_name)
            
            # Find direct indicator references
            ind_pattern = r'INDICATOR\(([^)]+)\)'
            ind_matches = re.findall(ind_pattern, expr)
            for ind_name in ind_matches:
                required.add(ind_name)
        
        return required


# Example usage
if __name__ == "__main__":
    parser = ExpressionParser(None)
    
    test_expressions = [
        "PRICE > SMA(PRIMARY, 50)",
        "RSI(ES, 2) < 30",
        "INDICATOR(VIX) < 25",
        "CROSSES_BELOW(EMA(PRIMARY, 9), EMA(PRIMARY, 15))",
        "PROFIT_PCT >= PARAM(profit_target_percent)",
        "DECLINE_FROM_PEAK(PRIMARY, 10) < -0.01"
    ]
    
    print("Expression Parsing Tests:")
    print("="*60)
    
    for expr in test_expressions:
        parsed = parser.parse_expression(expr)
        print(f"\nExpression: {expr}")
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
    
    required = parser.get_required_indicators(test_expressions)
    print(f"\nRequired indicators: {required}")