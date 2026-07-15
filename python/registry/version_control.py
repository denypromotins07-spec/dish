"""
Version Control for Strategy Parameters: Git-like versioning system.
Saves parameter snapshots to SQLite for instant rollback via UI.
Memory-efficient with bounded history retention.
"""

import sqlite3
import json
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time


@dataclass
class ParameterSnapshot:
    """A single parameter snapshot."""
    snapshot_id: int
    strategy_name: str
    version: int
    parameters: Dict[str, float]
    risk_limits: Dict[str, float]
    created_at: int  # microseconds
    created_by: str  # 'user' or 'auto'
    description: str
    parent_version: Optional[int]


class StrategyVersionControl:
    """
    SQLite-backed version control for strategy parameters.
    Supports branching, tagging, and instant rollback.
    """
    
    def __init__(self, db_path: str = ":memory:", max_versions_per_strategy: int = 100):
        self.db_path = db_path
        self.max_versions = max_versions_per_strategy
        self._lock = threading.RLock()
        self._init_db()
        
    def _init_db(self):
        """Initialize the database schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    parameters TEXT NOT NULL,
                    risk_limits TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    parent_version INTEGER,
                    is_active BOOLEAN DEFAULT FALSE,
                    UNIQUE(strategy_name, version)
                )
            ''')
            
            # Indexes for fast lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_strategy_version 
                ON snapshots(strategy_name, version)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON snapshots(created_at DESC)
            ''')
            
            # Tags table for named versions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    tag_name TEXT PRIMARY KEY,
                    snapshot_id INTEGER NOT NULL,
                    strategy_name TEXT NOT NULL,
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def create_snapshot(
        self,
        strategy_name: str,
        parameters: Dict[str, float],
        risk_limits: Dict[str, float],
        created_by: str = "user",
        description: str = "",
        parent_version: Optional[int] = None
    ) -> int:
        """
        Create a new parameter snapshot.
        Returns the new version number.
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current max version
            cursor.execute('''
                SELECT COALESCE(MAX(version), 0) FROM snapshots
                WHERE strategy_name = ?
            ''', (strategy_name,))
            
            current_version = cursor.fetchone()[0]
            new_version = current_version + 1
            
            # Enforce max versions limit
            if new_version > self.max_versions:
                # Delete oldest version
                cursor.execute('''
                    DELETE FROM snapshots 
                    WHERE strategy_name = ? AND version = (
                        SELECT MIN(version) FROM snapshots WHERE strategy_name = ?
                    )
                ''', (strategy_name, strategy_name))
            
            # Insert new snapshot
            cursor.execute('''
                INSERT INTO snapshots 
                (strategy_name, version, parameters, risk_limits, created_at, 
                 created_by, description, parent_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_name,
                new_version,
                json.dumps(parameters),
                json.dumps(risk_limits),
                int(time.time() * 1_000_000),
                created_by,
                description,
                parent_version if parent_version else current_version
            ))
            
            # Mark as active
            cursor.execute('''
                UPDATE snapshots SET is_active = FALSE 
                WHERE strategy_name = ?
            ''', (strategy_name,))
            
            cursor.execute('''
                UPDATE snapshots SET is_active = TRUE 
                WHERE strategy_name = ? AND version = ?
            ''', (strategy_name, new_version))
            
            conn.commit()
            conn.close()
            
            return new_version
    
    def get_snapshot(
        self,
        strategy_name: str,
        version: Optional[int] = None
    ) -> Optional[ParameterSnapshot]:
        """
        Get a specific snapshot. If version is None, gets the active one.
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if version is None:
                cursor.execute('''
                    SELECT snapshot_id, strategy_name, version, parameters, 
                           risk_limits, created_at, created_by, description, parent_version
                    FROM snapshots
                    WHERE strategy_name = ? AND is_active = TRUE
                ''', (strategy_name,))
            else:
                cursor.execute('''
                    SELECT snapshot_id, strategy_name, version, parameters, 
                           risk_limits, created_at, created_by, description, parent_version
                    FROM snapshots
                    WHERE strategy_name = ? AND version = ?
                ''', (strategy_name, version))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            return ParameterSnapshot(
                snapshot_id=row[0],
                strategy_name=row[1],
                version=row[2],
                parameters=json.loads(row[3]),
                risk_limits=json.loads(row[4]),
                created_at=row[5],
                created_by=row[6],
                description=row[7],
                parent_version=row[8]
            )
    
    def rollback_to_version(self, strategy_name: str, version: int) -> bool:
        """
        Rollback to a previous version (creates a new snapshot based on old one).
        Returns True if successful.
        """
        snapshot = self.get_snapshot(strategy_name, version)
        if snapshot is None:
            return False
        
        # Create new snapshot based on old one
        self.create_snapshot(
            strategy_name=strategy_name,
            parameters=snapshot.parameters,
            risk_limits=snapshot.risk_limits,
            created_by="rollback",
            description=f"Rollback to version {version}",
            parent_version=version
        )
        
        return True
    
    def list_versions(self, strategy_name: str) -> List[Dict]:
        """List all versions for a strategy."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT version, created_at, created_by, description, is_active
                FROM snapshots
                WHERE strategy_name = ?
                ORDER BY version DESC
            ''', (strategy_name,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'version': row[0],
                    'created_at': row[1],
                    'created_by': row[2],
                    'description': row[3],
                    'is_active': bool(row[4])
                }
                for row in rows
            ]
    
    def add_tag(self, tag_name: str, strategy_name: str, version: int) -> bool:
        """Add a named tag to a specific version."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get snapshot_id
            cursor.execute('''
                SELECT snapshot_id FROM snapshots
                WHERE strategy_name = ? AND version = ?
            ''', (strategy_name, version))
            
            row = cursor.fetchone()
            if row is None:
                conn.close()
                return False
            
            snapshot_id = row[0]
            
            # Insert or replace tag
            cursor.execute('''
                INSERT OR REPLACE INTO tags (tag_name, snapshot_id, strategy_name)
                VALUES (?, ?, ?)
            ''', (tag_name, snapshot_id, strategy_name))
            
            conn.commit()
            conn.close()
            return True
    
    def get_tagged_version(self, tag_name: str) -> Optional[ParameterSnapshot]:
        """Get snapshot by tag name."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.snapshot_id, s.strategy_name, s.version, s.parameters,
                       s.risk_limits, s.created_at, s.created_by, s.description, s.parent_version
                FROM snapshots s
                JOIN tags t ON s.snapshot_id = t.snapshot_id
                WHERE t.tag_name = ?
            ''', (tag_name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            return ParameterSnapshot(
                snapshot_id=row[0],
                strategy_name=row[1],
                version=row[2],
                parameters=json.loads(row[3]),
                risk_limits=json.loads(row[4]),
                created_at=row[5],
                created_by=row[6],
                description=row[7],
                parent_version=row[8]
            )
    
    def delete_old_snapshots(self, strategy_name: str, keep_last_n: int) -> int:
        """Delete old snapshots, keeping only the last N. Returns count deleted."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM snapshots
                WHERE strategy_name = ? AND version NOT IN (
                    SELECT version FROM snapshots
                    WHERE strategy_name = ?
                    ORDER BY version DESC
                    LIMIT ?
                )
            ''', (strategy_name, strategy_name, keep_last_n))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            return deleted
    
    def export_all_snapshots(self, strategy_name: Optional[str] = None) -> str:
        """Export all snapshots to JSON."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if strategy_name:
                cursor.execute('''
                    SELECT strategy_name, version, parameters, risk_limits,
                           created_at, created_by, description
                    FROM snapshots
                    WHERE strategy_name = ?
                    ORDER BY version
                ''', (strategy_name,))
            else:
                cursor.execute('''
                    SELECT strategy_name, version, parameters, risk_limits,
                           created_at, created_by, description
                    FROM snapshots
                    ORDER BY strategy_name, version
                ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            data = []
            for row in rows:
                data.append({
                    'strategy_name': row[0],
                    'version': row[1],
                    'parameters': json.loads(row[2]),
                    'risk_limits': json.loads(row[3]),
                    'created_at': row[4],
                    'created_by': row[5],
                    'description': row[6]
                })
            
            return json.dumps({'snapshots': data}, separators=(',', ':'))


# Singleton instance
_vc_instance: Optional[StrategyVersionControl] = None
_instance_lock = threading.Lock()


def get_version_control(db_path: str = ":memory:") -> StrategyVersionControl:
    """Get or create the singleton VersionControl instance."""
    global _vc_instance
    if _vc_instance is None:
        with _instance_lock:
            if _vc_instance is None:
                _vc_instance = StrategyVersionControl(db_path)
    return _vc_instance


if __name__ == '__main__':
    # Example usage
    vc = get_version_control(":memory:")
    
    # Create some snapshots
    v1 = vc.create_snapshot(
        strategy_name="momentum_v1",
        parameters={'lookback': 20.0, 'threshold': 0.02},
        risk_limits={'max_dd': 0.05},
        description="Initial version"
    )
    print(f"Created version {v1}")
    
    v2 = vc.create_snapshot(
        strategy_name="momentum_v1",
        parameters={'lookback': 30.0, 'threshold': 0.025},
        risk_limits={'max_dd': 0.05},
        description="Increased lookback"
    )
    print(f"Created version {v2}")
    
    # Add a tag
    vc.add_tag("stable", "momentum_v1", v1)
    
    # List versions
    versions = vc.list_versions("momentum_v1")
    print(f"Versions: {versions}")
    
    # Get tagged version
    tagged = vc.get_tagged_version("stable")
    print(f"Tagged version params: {tagged.parameters if tagged else None}")
    
    # Rollback
    vc.rollback_to_version("momentum_v1", v1)
    
    # Export
    export = vc.export_all_snapshots()
    print(f"Export size: {len(export)} bytes")
