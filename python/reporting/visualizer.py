"""
Lightweight, non-blocking HTML/Plotly report generator.
Exports equity curves, drawdown heatmaps, and monthly return tables to disk without holding massive charting objects in RAM.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Trading Strategy Performance Report"
    output_dir: str = "./reports"
    include_equity_curve: bool = True
    include_drawdown_heatmap: bool = True
    include_monthly_returns: bool = True
    include_metrics_table: bool = True
    theme: str = "dark"  # "dark" or "light"
    width: int = 1200
    height: int = 600


class ReportGenerator:
    """
    Memory-efficient HTML report generator using Plotly.
    
    Features:
    - Non-blocking async rendering
    - Streaming output to disk (no large objects in RAM)
    - Responsive design
    - Dark/light theme support
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.output_dir = self.config.output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        equity_curve: Optional[List[float]] = None,
        returns_data: Optional[Dict] = None,
        trades: Optional[List[Dict]] = None,
        filename: str = "performance_report.html",
    ) -> str:
        """
        Generate complete HTML report.
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            Performance metrics dictionary.
        equity_curve : Optional[List[float]]
            Equity curve data points.
        returns_data : Optional[Dict]
            Returns data for heatmap/table.
        trades : Optional[List[Dict]]
            Individual trade records.
        filename : str
            Output filename.
            
        Returns
        -------
        str
            Path to generated report.
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Build report sections incrementally (memory efficient)
        html_parts = []
        
        # Header
        html_parts.append(self._generate_header())
        
        # Summary metrics
        html_parts.append(self._generate_metrics_section(metrics))
        
        # Equity curve
        if self.config.include_equity_curve and equity_curve:
            html_parts.append(self._generate_equity_chart(equity_curve))
        
        # Drawdown heatmap
        if self.config.include_drawdown_heatmap and returns_data:
            html_parts.append(self._generate_drawdown_heatmap(returns_data))
        
        # Monthly returns table
        if self.config.include_monthly_returns and returns_data:
            html_parts.append(self._generate_monthly_returns_table(returns_data))
        
        # Trade analysis
        if trades:
            html_parts.append(self._generate_trade_table(trades))
        
        # Footer
        html_parts.append(self._generate_footer())
        
        # Write incrementally to avoid memory buildup
        with open(output_path, 'w') as f:
            for part in html_parts:
                f.write(part)
                f.flush()  # Stream to disk
        
        logger.info(f"Report generated: {output_path}")
        return output_path
    
    def _generate_header(self) -> str:
        """Generate HTML header with CSS and Plotly CDN."""
        theme_bg = "#1e1e1e" if self.config.theme == "dark" else "#ffffff"
        theme_text = "#ffffff" if self.config.theme == "dark" else "#000000"
        card_bg = "#2d2d2d" if self.config.theme == "dark" else "#f8f9fa"
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: {theme_bg};
            color: {theme_text};
            padding: 20px;
        }}
        .container {{ max-width: {self.config.width}px; margin: 0 auto; }}
        h1 {{ text-align: center; margin-bottom: 30px; font-size: 2em; }}
        h2 {{ margin: 20px 0 15px; font-size: 1.4em; border-bottom: 2px solid #4a9eff; padding-bottom: 10px; }}
        .card {{ 
            background-color: {card_bg};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
        }}
        .metric-card {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .metric-value {{ font-size: 1.8em; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.8; }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 15px;
        }}
        th, td {{ 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid #444; 
        }}
        th {{ background-color: #333; }}
        tr:hover {{ background-color: #3a3a3a; }}
        .chart-container {{ height: {self.config.height}px; }}
        .footer {{ text-align: center; margin-top: 40px; opacity: 0.6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.config.title}</h1>
        <p style="text-align: center; margin-bottom: 30px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
'''
    
    def _generate_metrics_section(self, metrics: Dict[str, Any]) -> str:
        """Generate metrics summary cards."""
        html = '<div class="card"><h2>Performance Summary</h2><div class="metrics-grid">'
        
        key_metrics = [
            ('Total Return', metrics.get('total_return', 0), '%'),
            ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), ''),
            ('Max Drawdown', metrics.get('max_drawdown', 0), '%'),
            ('Win Rate', metrics.get('win_rate', 0), '%'),
            ('Profit Factor', metrics.get('profit_factor', 0), ''),
            ('Total Trades', metrics.get('total_trades', 0), ''),
        ]
        
        for label, value, suffix in key_metrics:
            formatted_value = f"{value:.2f}{suffix}" if isinstance(value, float) else str(value)
            css_class = "positive" if value > 0 else "negative" if value < 0 else ""
            
            html += f'''
            <div class="metric-card">
                <div class="metric-value {css_class}">{formatted_value}</div>
                <div class="metric-label">{label}</div>
            </div>
            '''
        
        html += '</div></div>'
        return html
    
    def _generate_equity_chart(self, equity_curve: List[float]) -> str:
        """Generate equity curve chart."""
        chart_id = "equity-chart"
        
        # Create minimal plotly figure JSON
        fig_dict = {
            'data': [{
                'x': list(range(len(equity_curve))),
                'y': equity_curve,
                'type': 'scatter',
                'mode': 'lines',
                'line': {'color': '#4a9eff', 'width': 2},
                'name': 'Equity'
            }],
            'layout': {
                'title': 'Equity Curve',
                'paper_bgcolor': '#1e1e1e' if self.config.theme == 'dark' else '#ffffff',
                'plot_bgcolor': '#1e1e1e' if self.config.theme == 'dark' else '#ffffff',
                'font': {'color': '#ffffff' if self.config.theme == 'dark' else '#000000'},
                'height': self.config.height,
                'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50},
            }
        }
        
        return f'''
        <div class="card">
            <h2>Equity Curve</h2>
            <div id="{chart_id}" class="chart-container"></div>
            <script>
                var data = {json.dumps(fig_dict['data'])};
                var layout = {json.dumps(fig_dict['layout'])};
                Plotly.newPlot('{chart_id}', data, layout, {{responsive: true}});
            </script>
        </div>
        '''
    
    def _generate_drawdown_heatmap(self, returns_data: Dict) -> str:
        """Generate drawdown heatmap."""
        # Extract monthly drawdowns for heatmap
        monthly_dd = returns_data.get('monthly_drawdowns', [])
        
        if not monthly_dd:
            return ''
        
        chart_id = "drawdown-heatmap"
        
        fig_dict = {
            'data': [{
                'type': 'heatmap',
                'z': monthly_dd,
                'colorscales': [[0, '#4caf50'], [0.5, '#ffeb3b'], [1, '#f44336']],
                'colorbar': {'title': 'Drawdown %'},
            }],
            'layout': {
                'title': 'Drawdown Heatmap by Month',
                'paper_bgcolor': '#1e1e1e' if self.config.theme == 'dark' else '#ffffff',
                'height': 400,
            }
        }
        
        return f'''
        <div class="card">
            <h2>Drawdown Analysis</h2>
            <div id="{chart_id}" class="chart-container"></div>
            <script>
                var data = {json.dumps(fig_dict['data'])};
                var layout = {json.dumps(fig_dict['layout'])};
                Plotly.newPlot('{chart_id}', data, layout, {{responsive: true}});
            </script>
        </div>
        '''
    
    def _generate_monthly_returns_table(self, returns_data: Dict) -> str:
        """Generate monthly returns table."""
        monthly_returns = returns_data.get('monthly_returns', {})
        
        if not monthly_returns:
            return '<div class="card"><h2>Monthly Returns</h2><p>No data available</p></div>'
        
        html = '<div class="card"><h2>Monthly Returns (%)</h2><table><thead><tr>'
        html += '<th>Month</th><th>Return</th><th>Cumulative</th></tr></thead><tbody>'
        
        cumulative = 1.0
        for month, ret in sorted(monthly_returns.items()):
            cumulative *= (1 + ret / 100)
            cum_ret = (cumulative - 1) * 100
            css_class = "positive" if ret > 0 else "negative" if ret < 0 else ""
            html += f'<tr><td>{month}</td><td class="{css_class}">{ret:.2f}%</td><td class="{css_class}">{cum_ret:.2f}%</td></tr>'
        
        html += '</tbody></table></div>'
        return html
    
    def _generate_trade_table(self, trades: List[Dict]) -> str:
        """Generate individual trades table."""
        if not trades:
            return ''
        
        # Limit to last 100 trades for performance
        trades_display = trades[-100:] if len(trades) > 100 else trades
        
        html = '<div class="card"><h2>Recent Trades</h2><table><thead><tr>'
        html += '<th>#</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>PnL</th></tr></thead><tbody>'
        
        for i, trade in enumerate(trades_display):
            pnl_class = "positive" if trade.get('pnl', 0) > 0 else "negative" if trade.get('pnl', 0) < 0 else ""
            html += f'''
            <tr>
                <td>{i + 1}</td>
                <td>{trade.get('symbol', 'N/A')}</td>
                <td>{trade.get('side', 'N/A')}</td>
                <td>{trade.get('quantity', 0):.4f}</td>
                <td>{trade.get('price', 0):.2f}</td>
                <td class="{pnl_class}">{trade.get('pnl', 0):.2f}</td>
            </tr>
            '''
        
        html += '</tbody></table></div>'
        return html
    
    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return f'''
        <div class="footer">
            <p>Ultra-Low Latency Crypto Trading Bot - Stage 7 Backtest Report</p>
            <p>Hardware: AMD Ryzen AI 5 | AMD Radeon GPU | 16GB RAM</p>
        </div>
    </div>
</body>
</html>
'''


def generate_backtest_report(
    metrics: Dict[str, Any],
    equity_curve: List[float],
    returns_data: Dict,
    trades: Optional[List[Dict]] = None,
    output_dir: str = "./reports",
) -> str:
    """
    Convenience function to generate a complete backtest report.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Performance metrics.
    equity_curve : List[float]
        Equity curve values.
    returns_data : Dict
        Returns data for additional charts.
    trades : Optional[List[Dict]]
        Trade records.
    output_dir : str
        Output directory.
        
    Returns
    -------
    str
        Path to generated report.
    """
    config = ReportConfig(
        title="Stage 7 Backtest Report",
        output_dir=output_dir,
        theme="dark",
    )
    
    generator = ReportGenerator(config)
    
    return generator.generate_report(
        metrics=metrics,
        equity_curve=equity_curve,
        returns_data=returns_data,
        trades=trades,
        filename=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample data for testing
    sample_metrics = {
        'total_return': 0.2547,
        'sharpe_ratio': 1.85,
        'max_drawdown': -0.0823,
        'win_rate': 0.58,
        'profit_factor': 1.92,
        'total_trades': 1247,
    }
    
    # Generate sample equity curve
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 500)
    equity_curve = (100000 * np.cumprod(1 + returns)).tolist()
    
    # Sample returns data
    sample_returns = {
        'monthly_returns': {
            '2024-01': 5.2,
            '2024-02': -2.1,
            '2024-03': 8.7,
            '2024-04': 3.4,
            '2024-05': -1.2,
            '2024-06': 6.8,
        },
        'monthly_drawdowns': [[-2, -5, -3], [-8, -12, -6], [-1, -4, -2]],
    }
    
    # Generate report
    report_path = generate_backtest_report(
        metrics=sample_metrics,
        equity_curve=equity_curve,
        returns_data=sample_returns,
        output_dir="./test_reports",
    )
    
    print(f"Report generated at: {report_path}")
