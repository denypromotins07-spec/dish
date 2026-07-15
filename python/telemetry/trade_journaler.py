"""
Asynchronous batch writer for trade journal and TCA (Transaction Cost Analysis).
Formats executed trades into JSON/SQLite for frontend "Trade History" and "Journal" tabs.
Strictly caps memory usage with bounded queues and periodic flushing.
"""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from collections import deque
from pathlib import Path
import threading


@dataclass
class TradeRecord:
    """Single trade record for journal."""
    trade_id: str
    timestamp_ms: int
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    fee_currency: str
    order_id: str
    strategy_name: str
    pnl: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    execution_quality_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TCASummary:
    """Transaction Cost Analysis summary for a period."""
    period_start_ms: int
    period_end_ms: int
    total_trades: int
    total_volume: float
    total_fees: float
    avg_slippage_bps: float
    avg_market_impact_bps: float
    avg_execution_quality: float
    total_pnl: float
    win_rate: float
    profit_factor: float


class TradeJournaler:
    """
    Asynchronous batch writer for trade journal and TCA.
    Memory-bounded with periodic flushing to prevent RAM bloat.
    """

    def __init__(
        self,
        db_path: str = "trade_journal.db",
        max_pending_trades: int = 10000,  # Memory bound
        flush_interval_seconds: int = 60,
        max_tca_history_days: int = 30,
    ):
        self.db_path = Path(db_path)
        self.max_pending_trades = max_pending_trades
        self.flush_interval_seconds = flush_interval_seconds
        self.max_tca_history_days = max_tca_history_days

        # Pending trades queue (bounded)
        self.pending_trades: deque[TradeRecord] = deque(maxlen=max_pending_trades)
        
        # Thread-safe lock for queue access
        self._queue_lock = threading.Lock()

        # SQLite connection (created per thread)
        self._local = threading.local()

        # Running state
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

        # Statistics
        self.total_trades_written = 0
        self.last_flush_time = 0

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._init_database(self._local.conn)
        return self._local.conn

    def _init_database(self, conn: sqlite3.Connection):
        """Initialize database schema."""
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                timestamp_ms INTEGER,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                fee REAL,
                fee_currency TEXT,
                order_id TEXT,
                strategy_name TEXT,
                pnl REAL,
                slippage_bps REAL,
                market_impact_bps REAL,
                execution_quality_score REAL,
                metadata TEXT
            )
        ''')
        
        # TCA summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tca_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start_ms INTEGER,
                period_end_ms INTEGER,
                total_trades INTEGER,
                total_volume REAL,
                total_fees REAL,
                avg_slippage_bps REAL,
                avg_market_impact_bps REAL,
                avg_execution_quality REAL,
                total_pnl REAL,
                win_rate REAL,
                profit_factor REAL
            )
        ''')
        
        # Indexes for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp_ms)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name)')
        
        conn.commit()

    def add_trade(self, trade: TradeRecord):
        """Add a trade to the pending queue (thread-safe)."""
        with self._queue_lock:
            self.pending_trades.append(trade)
            
            # Auto-flush if queue is getting full
            if len(self.pending_trades) >= self.max_pending_trades * 0.9:
                self._flush_pending_trades()

    def add_trade_raw(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        order_id: str,
        strategy_name: str,
        timestamp_ms: Optional[int] = None,
        **kwargs,
    ):
        """Convenience method to add a trade with raw parameters."""
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
            
        trade = TradeRecord(
            trade_id=trade_id,
            timestamp_ms=timestamp_ms,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            fee_currency=kwargs.get('fee_currency', 'USD'),
            order_id=order_id,
            strategy_name=strategy_name,
            pnl=kwargs.get('pnl', 0.0),
            slippage_bps=kwargs.get('slippage_bps', 0.0),
            market_impact_bps=kwargs.get('market_impact_bps', 0.0),
            execution_quality_score=kwargs.get('execution_quality_score', 0.0),
            metadata=kwargs.get('metadata', {}),
        )
        self.add_trade(trade)

    def _flush_pending_trades(self):
        """Flush pending trades to SQLite."""
        with self._queue_lock:
            if not self.pending_trades:
                return
                
            trades_to_write = list(self.pending_trades)
            self.pending_trades.clear()

        conn = self._get_connection()
        cursor = conn.cursor()
        
        for trade in trades_to_write:
            cursor.execute('''
                INSERT OR REPLACE INTO trades 
                (trade_id, timestamp_ms, symbol, side, quantity, price, fee, 
                 fee_currency, order_id, strategy_name, pnl, slippage_bps, 
                 market_impact_bps, execution_quality_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.trade_id,
                trade.timestamp_ms,
                trade.symbol,
                trade.side,
                trade.quantity,
                trade.price,
                trade.fee,
                trade.fee_currency,
                trade.order_id,
                trade.strategy_name,
                trade.pnl,
                trade.slippage_bps,
                trade.market_impact_bps,
                trade.execution_quality_score,
                json.dumps(trade.metadata),
            ))
            self.total_trades_written += 1
        
        conn.commit()
        self.last_flush_time = int(time.time() * 1000)

    async def start_background_flush(self):
        """Start background task for periodic flushing."""
        self._running = True
        
        async def flush_loop():
            while self._running:
                await asyncio.sleep(self.flush_interval_seconds)
                self._flush_pending_trades()
                
                # Cleanup old TCA data
                self._cleanup_old_tca_data()
        
        self._flush_task = asyncio.create_task(flush_loop())

    async def stop(self):
        """Stop background tasks and flush remaining trades."""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        self._flush_pending_trades()

    def _cleanup_old_tca_data(self):
        """Remove TCA data older than max_tca_history_days."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_ms = int((time.time() - self.max_tca_history_days * 86400) * 1000)
        cursor.execute('DELETE FROM tca_summaries WHERE period_end_ms < ?', (cutoff_ms,))
        conn.commit()

    def get_recent_trades(
        self,
        limit: int = 100,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[Dict]:
        """Get recent trades from database, optionally filtered."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM trades WHERE 1=1'
        params = []
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if strategy:
            query += ' AND strategy_name = ?'
            params.append(strategy)
        
        query += ' ORDER BY timestamp_ms DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        results = []
        for row in cursor.fetchall():
            trade_dict = dict(zip(columns, row))
            # Parse metadata JSON
            if trade_dict.get('metadata'):
                trade_dict['metadata'] = json.loads(trade_dict['metadata'])
            results.append(trade_dict)
        
        return results

    def get_tca_summary(
        self,
        start_ms: int,
        end_ms: int,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Optional[TCASummary]:
        """Calculate TCA summary for a period."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT 
                COUNT(*) as total_trades,
                SUM(quantity) as total_volume,
                SUM(fee) as total_fees,
                AVG(slippage_bps) as avg_slippage_bps,
                AVG(market_impact_bps) as avg_market_impact_bps,
                AVG(execution_quality_score) as avg_execution_quality,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades
            FROM trades
            WHERE timestamp_ms BETWEEN ? AND ?
        '''
        params = [start_ms, end_ms]
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if strategy:
            query += ' AND strategy_name = ?'
            params.append(strategy)
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if row[0] == 0:
            return None
        
        total_trades = row[0]
        winning_trades = row[7] or 0
        total_pnl = row[6] or 0.0
        losing_trades = total_trades - winning_trades
        
        # Calculate profit factor
        losing_pnl_query = '''
            SELECT ABS(SUM(pnl)) FROM trades 
            WHERE timestamp_ms BETWEEN ? AND ? AND pnl < 0
        '''
        cursor.execute(losing_pnl_query, [start_ms, end_ms])
        losing_amount = cursor.fetchone()[0] or 0.0
        
        profit_factor = (total_pnl / losing_amount) if losing_amount > 0 else float('inf')
        
        return TCASummary(
            period_start_ms=start_ms,
            period_end_ms=end_ms,
            total_trades=total_trades,
            total_volume=row[1] or 0.0,
            total_fees=row[2] or 0.0,
            avg_slippage_bps=row[3] or 0.0,
            avg_market_impact_bps=row[4] or 0.0,
            avg_execution_quality=row[5] or 0.0,
            total_pnl=total_pnl,
            win_rate=winning_trades / total_trades if total_trades > 0 else 0.0,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get current journal statistics."""
        return {
            'pending_trades': len(self.pending_trades),
            'total_trades_written': self.total_trades_written,
            'last_flush_time': self.last_flush_time,
            'db_path': str(self.db_path),
        }

    def export_to_json(self, output_path: str, limit: int = 10000):
        """Export recent trades to JSON file."""
        trades = self.get_recent_trades(limit=limit)
        
        with open(output_path, 'w') as f:
            json.dump(trades, f, indent=2)
        
        return len(trades)


# Example usage
if __name__ == '__main__':
    import tempfile
    
    # Create journaler with temp DB
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    journaler = TradeJournaler(db_path=db_path, max_pending_trades=1000)
    
    # Add some test trades
    journaler.add_trade_raw(
        trade_id='test_001',
        symbol='BTC-PERP',
        side='buy',
        quantity=0.5,
        price=50000.0,
        fee=2.5,
        order_id='ord_123',
        strategy_name='momentum',
        pnl=150.0,
        slippage_bps=5.0,
    )
    
    journaler.add_trade_raw(
        trade_id='test_002',
        symbol='BTC-PERP',
        side='sell',
        quantity=0.5,
        price=50200.0,
        fee=2.5,
        order_id='ord_124',
        strategy_name='momentum',
        pnl=-50.0,
        slippage_bps=3.0,
    )
    
    # Flush to database
    journaler._flush_pending_trades()
    
    # Query recent trades
    trades = journaler.get_recent_trades(limit=10)
    print(f"Retrieved {len(trades)} trades")
    
    # Get TCA summary
    now_ms = int(time.time() * 1000)
    tca = journaler.get_tca_summary(now_ms - 86400000, now_ms)
    if tca:
        print(f"TCA Summary: {asdict(tca)}")
    
    print(f"Statistics: {journaler.get_statistics()}")
