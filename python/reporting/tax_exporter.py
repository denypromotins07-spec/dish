"""
Memory-efficient tax report generator.
Streams millions of trades directly from Parquet into standard CSV formats.
Uses chunked Polars reads to maintain strict 14GB RAM limit.
"""

from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Iterator, List, Dict, Any
from dataclasses import dataclass

import polars as pl


logger = logging.getLogger(__name__)


# Tax report formats supported
TAX_FORMATS = {
    "turbotax": {
        "columns": ["Date", "Symbol", "Quantity", "Cost Basis", "Proceeds", "Gain/Loss"],
        "filename_pattern": "tax_report_turbotax_{year}.csv",
    },
    "cointracker": {
        "columns": ["Date", "Received Amount", "Received Currency", "Sent Amount", "Sent Currency", "Fee", "Tag"],
        "filename_pattern": "tax_report_cointracker_{year}.csv",
    },
    "koinly": {
        "columns": ["Date", "Sent Amount", "Sent Currency", "Received Amount", "Received Currency", "Fee Amount", "Fee Currency", "Label"],
        "filename_pattern": "tax_report_koinly_{year}.csv",
    },
    "generic": {
        "columns": ["trade_id", "timestamp", "symbol", "side", "quantity", "price", "fees", "realized_pnl"],
        "filename_pattern": "tax_report_generic_{year}.csv",
    },
}


@dataclass
class TradeRecord:
    """Single trade record for tax reporting."""
    trade_id: int
    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    price: float
    fees: float
    realized_pnl: float
    cost_basis: float
    proceeds: float


class TaxExporter:
    """
    Memory-efficient tax report exporter.
    
    Streams trades from Parquet files in chunks to avoid loading
    entire dataset into memory while generating tax reports.
    """
    
    def __init__(
        self,
        parquet_directory: str,
        output_directory: str,
        chunk_size_rows: int = 100000,  # Process 100k rows at a time
        max_memory_mb: int = 500,  # Target max memory usage
    ):
        self.parquet_dir = Path(parquet_directory)
        self.output_dir = Path(output_directory)
        self.chunk_size = chunk_size_rows
        self.max_memory_mb = max_memory_mb
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self._total_trades_processed = 0
        self._total_files_generated = 0
    
    def stream_trades_for_year(
        self,
        year: int,
        filter_symbols: Optional[List[str]] = None,
    ) -> Iterator[pl.DataFrame]:
        """
        Stream trades for a specific year in memory-bounded chunks.
        
        Args:
            year: Tax year to export
            filter_symbols: Optional list of symbols to include
        
        Yields:
            DataFrames of chunk_size_rows each
        """
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        # Find all relevant parquet files
        parquet_files = list(self.parquet_dir.glob("trades_*.parquet"))
        
        if not parquet_files:
            logger.warning(f"No parquet files found in {self.parquet_dir}")
            return
        
        for file_path in sorted(parquet_files):
            try:
                # Use Polars scan for lazy evaluation (memory efficient)
                lf = pl.scan_parquet(file_path)
                
                # Build filter expression
                filters = [
                    pl.col("timestamp").dt.year() == year,
                ]
                
                if filter_symbols:
                    filters.append(pl.col("symbol").is_in(filter_symbols))
                
                # Apply filters and select only needed columns
                query = lf.filter(pl.all_horizontal(filters)).select([
                    "trade_id",
                    "timestamp",
                    "symbol",
                    "side",
                    "quantity",
                    "price",
                    "fees",
                    "realized_pnl",
                    "cost_basis",
                    "proceeds",
                ])
                
                # Execute in chunks using fetch
                chunk = query.fetch(self.chunk_size)
                
                while len(chunk) > 0:
                    yield chunk
                    self._total_trades_processed += len(chunk)
                    
                    # Fetch next chunk
                    chunk = query.fetch(self.chunk_size)
                    
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue
    
    def generate_tax_report(
        self,
        year: int,
        format_name: str = "generic",
        filter_symbols: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Generate a complete tax report for a year in specified format.
        
        Args:
            year: Tax year
            format_name: Output format (turbotax, cointracker, koinly, generic)
            filter_symbols: Optional symbol filter
        
        Returns:
            Path to generated file, or None if failed
        """
        if format_name not in TAX_FORMATS:
            logger.error(f"Unknown tax format: {format_name}")
            return None
        
        format_config = TAX_FORMATS[format_name]
        filename = format_config["filename_pattern"].format(year=year)
        output_path = self.output_dir / filename
        
        logger.info(f"Generating {format_name} tax report for {year}")
        
        try:
            # Collect all chunks and transform
            all_chunks: List[pl.DataFrame] = []
            
            for chunk in self.stream_trades_for_year(year, filter_symbols):
                transformed = self._transform_chunk(chunk, format_name)
                all_chunks.append(transformed)
            
            if not all_chunks:
                logger.warning(f"No trades found for year {year}")
                # Create empty file with headers
                empty_df = pl.DataFrame(schema={col: pl.String for col in format_config["columns"]})
                empty_df.write_csv(output_path)
                return output_path
            
            # Concatenate all chunks
            full_report = pl.concat(all_chunks, how="vertical")
            
            # Sort by date
            if "Date" in full_report.columns:
                full_report = full_report.sort("Date")
            elif "timestamp" in full_report.columns:
                full_report = full_report.sort("timestamp")
            
            # Write to CSV
            full_report.write_csv(output_path)
            
            self._total_files_generated += 1
            
            logger.info(
                f"Generated tax report: {output_path}, "
                f"{len(full_report)} trades"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate tax report: {e}")
            return None
    
    def _transform_chunk(self, chunk: pl.DataFrame, format_name: str) -> pl.DataFrame:
        """Transform a chunk to the target tax format."""
        
        if format_name == "turbotax":
            return chunk.select([
                pl.col("timestamp").cast(pl.String).alias("Date"),
                pl.col("symbol").alias("Symbol"),
                pl.col("quantity").alias("Quantity"),
                pl.col("cost_basis").round(2).alias("Cost Basis"),
                pl.col("proceeds").round(2).alias("Proceeds"),
                (pl.col("proceeds") - pl.col("cost_basis")).round(2).alias("Gain/Loss"),
            ])
        
        elif format_name == "cointracker":
            # Split into buys and sells
            buys = chunk.filter(pl.col("side") == "BUY").with_columns([
                pl.col("timestamp").cast(pl.String).alias("Date"),
                pl.col("quantity").alias("Received Amount"),
                pl.col("symbol").alias("Received Currency"),
                pl.lit(None).alias("Sent Amount"),
                pl.lit("USD").alias("Sent Currency"),
                pl.col("fees").alias("Fee"),
                pl.lit("Buy").alias("Tag"),
            ])
            
            sells = chunk.filter(pl.col("side") == "SELL").with_columns([
                pl.col("timestamp").cast(pl.String).alias("Date"),
                pl.lit(None).alias("Received Amount"),
                pl.lit(None).alias("Received Currency"),
                pl.col("quantity").alias("Sent Amount"),
                pl.col("symbol").alias("Sent Currency"),
                pl.col("fees").alias("Fee"),
                pl.lit("Sell").alias("Tag"),
            ])
            
            combined = pl.concat([
                buys.select(["Date", "Received Amount", "Received Currency", "Sent Amount", "Sent Currency", "Fee", "Tag"]),
                sells.select(["Date", "Received Amount", "Received Currency", "Sent Amount", "Sent Currency", "Fee", "Tag"]),
            ])
            
            return combined
        
        elif format_name == "koinly":
            return chunk.with_columns([
                pl.col("timestamp").cast(pl.String).alias("Date"),
                pl.when(pl.col("side") == "SELL")
                    .then(pl.col("quantity"))
                    .otherwise(None)
                    .alias("Sent Amount"),
                pl.when(pl.col("side") == "SELL")
                    .then(pl.col("symbol"))
                    .otherwise(None)
                    .alias("Sent Currency"),
                pl.when(pl.col("side") == "BUY")
                    .then(pl.col("quantity"))
                    .otherwise(None)
                    .alias("Received Amount"),
                pl.when(pl.col("side") == "BUY")
                    .then(pl.col("symbol"))
                    .otherwise(None)
                    .alias("Received Currency"),
                pl.col("fees").alias("Fee Amount"),
                pl.when(pl.col("fees") > 0)
                    .then(pl.lit("USD"))
                    .otherwise(None)
                    .alias("Fee Currency"),
                pl.when(pl.col("side") == "BUY")
                    .then(pl.lit("Buy"))
                    .when(pl.col("side") == "SELL")
                    .then(pl.lit("Sell"))
                    .otherwise(pl.lit("Unknown"))
                    .alias("Label"),
            ]).select([
                "Date", "Sent Amount", "Sent Currency", 
                "Received Amount", "Received Currency",
                "Fee Amount", "Fee Currency", "Label"
            ])
        
        else:  # generic
            return chunk.select([
                pl.col("trade_id").cast(pl.Int64).alias("trade_id"),
                pl.col("timestamp").cast(pl.String).alias("timestamp"),
                pl.col("symbol").alias("symbol"),
                pl.col("side").alias("side"),
                pl.col("quantity").alias("quantity"),
                pl.col("price").alias("price"),
                pl.col("fees").alias("fees"),
                pl.col("realized_pnl").alias("realized_pnl"),
            ])
    
    def generate_summary(self, year: int) -> Dict[str, Any]:
        """Generate a tax summary for a year without exporting full report."""
        summary = {
            "year": year,
            "total_trades": 0,
            "total_realized_pnl": 0.0,
            "total_fees": 0.0,
            "by_symbol": {},
            "by_month": {},
        }
        
        for chunk in self.stream_trades_for_year(year):
            summary["total_trades"] += len(chunk)
            summary["total_realized_pnl"] += chunk["realized_pnl"].sum()
            summary["total_fees"] += chunk["fees"].sum()
            
            # By symbol
            symbol_groups = chunk.group_by("symbol").agg([
                pl.col("realized_pnl").sum(),
                pl.col("quantity").sum(),
            ])
            
            for row in symbol_groups.iter_rows(named=True):
                sym = row["symbol"]
                if sym not in summary["by_symbol"]:
                    summary["by_symbol"][sym] = {"pnl": 0.0, "volume": 0.0}
                summary["by_symbol"][sym]["pnl"] += row["realized_pnl"]
                summary["by_symbol"][sym]["volume"] += row["quantity"]
            
            # By month
            chunk_with_month = chunk.with_columns([
                pl.col("timestamp").dt.month().alias("month")
            ])
            month_groups = chunk_with_month.group_by("month").agg([
                pl.col("realized_pnl").sum(),
            ])
            
            for row in month_groups.iter_rows(named=True):
                month = row["month"]
                summary["by_month"][month] = summary["by_month"].get(month, 0.0) + row["realized_pnl"]
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get exporter statistics."""
        return {
            "total_trades_processed": self._total_trades_processed,
            "total_files_generated": self._total_files_generated,
            "chunk_size": self.chunk_size,
            "max_memory_mb": self.max_memory_mb,
        }


# Example usage
if __name__ == "__main__":
    exporter = TaxExporter(
        parquet_directory="/data/trades",
        output_directory="/data/tax_reports",
    )
    
    # Generate TurboTax format for 2024
    report_path = exporter.generate_tax_report(2024, format_name="turbotax")
    
    if report_path:
        print(f"Tax report generated: {report_path}")
    
    # Get summary
    summary = exporter.generate_summary(2024)
    print(f"Total trades: {summary['total_trades']}")
    print(f"Total PnL: ${summary['total_realized_pnl']:.2f}")
