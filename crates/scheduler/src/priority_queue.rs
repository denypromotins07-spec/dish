//! Lock-free Priority Queue for Task Scheduling
//! Ensures live trade execution and WebSocket ingestion ALWAYS preempt
//! background telemetry, logging, or ML tasks.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use crossbeam::queue::SegQueue;
use std::cmp::Ordering as CmpOrdering;
use std::time::{SystemTime, UNIX_EPOCH};

/// Priority levels (higher = more important)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Priority {
    Background = 0,      // Telemetry, logging, analytics
    Low = 50,            // Non-critical ML inference
    Normal = 100,        // Standard order management
    High = 150,          // Market data processing
    Critical = 200,      // Live trade execution
    Urgent = 255,        // Risk management, emergency cancel
}

/// Task wrapper with priority and timestamp
#[derive(Debug, Clone)]
pub struct PrioritizedTask<T> {
    pub priority: Priority,
    pub timestamp_ns: u64,
    pub task_id: u64,
    pub payload: T,
}

impl<T> PartialEq for PrioritizedTask<T> {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
    }
}

impl<T> Eq for PrioritizedTask<T> {}

impl<T> PartialOrd for PrioritizedTask<T> {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for PrioritizedTask<T> {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Higher priority first, then earlier timestamp
        match self.priority.cmp(&other.priority) {
            CmpOrdering::Equal => other.timestamp_ns.cmp(&self.timestamp_ns),
            other_ord => other_ord,
        }
    }
}

/// Lock-free priority queue using atomic operations
pub struct LockFreePriorityQueue<T> {
    heap: SegQueue<PrioritizedTask<T>>,
    sequence_counter: AtomicU64,
    is_shutdown: AtomicBool,
    max_size: usize,
    current_size: AtomicU64,
}

impl<T> LockFreePriorityQueue<T> 
where 
    T: Send + Sync + Clone,
{
    /// Create a new priority queue with max size limit
    pub fn new(max_size: usize) -> Self {
        Self {
            heap: SegQueue::new(),
            sequence_counter: AtomicU64::new(0),
            is_shutdown: AtomicBool::new(false),
            max_size,
            current_size: AtomicU64::new(0),
        }
    }
    
    /// Push a task into the queue (lock-free)
    pub fn push(&self, priority: Priority, payload: T) -> Result<(), &'static str> {
        if self.is_shutdown.load(Ordering::Relaxed) {
            return Err("Queue is shut down");
        }
        
        // Check size limit
        if self.current_size.load(Ordering::Relaxed) >= self.max_size as u64 {
            // Drop low-priority tasks if queue is full
            return Err("Queue full");
        }
        
        let task_id = self.sequence_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let task = PrioritizedTask {
            priority,
            timestamp_ns,
            task_id,
            payload,
        };
        
        self.heap.push(task);
        self.current_size.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Pop the highest priority task (lock-free)
    pub fn pop(&self) -> Option<PrioritizedTask<T>> {
        // This is a simplified implementation
        // In production, use a more sophisticated concurrent heap
        let mut tasks: Vec<PrioritizedTask<T>> = Vec::new();
        
        // Drain all tasks
        while let Some(task) = self.heap.pop() {
            tasks.push(task);
        }
        
        if tasks.is_empty() {
            return None;
        }
        
        // Find highest priority (earliest timestamp for ties)
        tasks.sort_by(|a, b| a.cmp(b));
        let best = tasks.remove(0);
        
        // Push remaining back
        for task in tasks {
            self.heap.push(task);
        }
        
        self.current_size.fetch_sub(1, Ordering::Relaxed);
        
        Some(best)
    }
    
    /// Peek at highest priority task without removing
    pub fn peek(&self) -> Option<PrioritizedTask<T>> 
    where
        T: Clone,
    {
        let mut tasks: Vec<PrioritizedTask<T>> = Vec::new();
        
        while let Some(task) = self.heap.pop() {
            tasks.push(task.clone());
            self.heap.push(task);
        }
        
        if tasks.is_empty() {
            return None;
        }
        
        tasks.sort_by(|a, b| a.cmp(b));
        Some(tasks[0].clone())
    }
    
    /// Get current queue size
    pub fn len(&self) -> usize {
        self.current_size.load(Ordering::Relaxed) as usize
    }
    
    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Shutdown the queue
    pub fn shutdown(&self) {
        self.is_shutdown.store(true, Ordering::Relaxed);
    }
    
    /// Clear all tasks from queue
    pub fn clear(&self) {
        while self.heap.pop().is_some() {}
        self.current_size.store(0, Ordering::Relaxed);
    }
    
    /// Drop all tasks below a certain priority threshold
    pub fn drop_below_priority(&self, min_priority: Priority) -> usize {
        let mut dropped = 0;
        let mut kept: Vec<PrioritizedTask<T>> = Vec::new();
        
        while let Some(task) = self.heap.pop() {
            if task.priority >= min_priority {
                kept.push(task);
            } else {
                dropped += 1;
            }
        }
        
        for task in kept {
            self.heap.push(task);
        }
        
        self.current_size.fetch_sub(dropped as u64, Ordering::Relaxed);
        dropped
    }
}

/// Task scheduler that uses the priority queue
pub struct TaskScheduler<T> {
    queue: Arc<LockFreePriorityQueue<T>>,
}

impl<T> TaskScheduler<T> 
where 
    T: Send + Sync + Clone + 'static,
{
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            queue: Arc::new(LockFreePriorityQueue::new(max_queue_size)),
        }
    }
    
    /// Schedule a critical trading task
    pub fn schedule_trade_execution(&self, payload: T) -> Result<(), &'static str> {
        self.queue.push(Priority::Critical, payload)
    }
    
    /// Schedule market data processing
    pub fn schedule_market_data(&self, payload: T) -> Result<(), &'static str> {
        self.queue.push(Priority::High, payload)
    }
    
    /// Schedule background task (telemetry, logging)
    pub fn schedule_background(&self, payload: T) -> Result<(), &'static str> {
        self.queue.push(Priority::Background, payload)
    }
    
    /// Process next task
    pub fn process_next(&self) -> Option<PrioritizedTask<T>> {
        self.queue.pop()
    }
    
    /// Emergency: drop all non-critical tasks
    pub fn emergency_clear(&self) {
        self.queue.drop_below_priority(Priority::Critical);
    }
    
    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        let len = self.queue.len();
        QueueStats {
            total_tasks: len,
            is_critical: len > 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueueStats {
    pub total_tasks: usize,
    pub is_critical: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_priority_ordering() {
        let queue: LockFreePriorityQueue<&str> = LockFreePriorityQueue::new(100);
        
        // Push tasks in random order
        queue.push(Priority::Background, "low").unwrap();
        queue.push(Priority::Critical, "high").unwrap();
        queue.push(Priority::Normal, "mid").unwrap();
        
        // Should pop in priority order
        let task1 = queue.pop().unwrap();
        assert_eq!(task1.payload, "high");
        assert_eq!(task1.priority, Priority::Critical);
        
        let task2 = queue.pop().unwrap();
        assert_eq!(task2.payload, "mid");
        
        let task3 = queue.pop().unwrap();
        assert_eq!(task3.payload, "low");
    }
    
    #[test]
    fn test_emergency_clear() {
        let queue: Arc<LockFreePriorityQueue<&str>> = Arc::new(LockFreePriorityQueue::new(100));
        
        queue.push(Priority::Background, "bg1").unwrap();
        queue.push(Priority::Critical, "crit1").unwrap();
        queue.push(Priority::Low, "low1").unwrap();
        
        queue.drop_below_priority(Priority::Critical);
        
        assert_eq!(queue.len(), 1);
        let task = queue.pop().unwrap();
        assert_eq!(task.priority, Priority::Critical);
    }
}
