"""Resource monitoring for security and performance."""

import time
from typing import Dict, Optional
from .logger_setup import logger

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - resource monitoring disabled")

class ResourceMonitor:
    """Monitor system resources to prevent DoS attacks."""
    
    def __init__(self, memory_limit_mb: int = 512, cpu_limit_percent: float = 80.0):
        """Initialize resource monitor.
        
        Args:
            memory_limit_mb: Memory limit in MB
            cpu_limit_percent: CPU usage limit percentage
        """
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.cpu_limit = cpu_limit_percent
        self.last_check = time.time()
        self.check_interval = 5.0  # Check every 5 seconds
        
    def check_resources(self) -> Dict[str, bool]:
        """Check if resource usage is within limits.
        
        Returns:
            Dictionary with resource status
        """
        if not PSUTIL_AVAILABLE:
            return {"memory_ok": True, "cpu_ok": True}
            
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return {"memory_ok": True, "cpu_ok": True}
            
        self.last_check = current_time
        
        try:
            # Get current process
            process = psutil.Process()
            
            # Check memory usage
            memory_info = process.memory_info()
            memory_ok = memory_info.rss < self.memory_limit
            
            # Check CPU usage
            cpu_percent = process.cpu_percent()
            cpu_ok = cpu_percent < self.cpu_limit
            
            if not memory_ok:
                logger.warning(f"Memory usage high: {memory_info.rss / 1024 / 1024:.1f}MB")
            if not cpu_ok:
                logger.warning(f"CPU usage high: {cpu_percent:.1f}%")
                
            return {"memory_ok": memory_ok, "cpu_ok": cpu_ok}
            
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
            return {"memory_ok": True, "cpu_ok": True}
    
    def should_throttle(self) -> bool:
        """Check if processing should be throttled.
        
        Returns:
            True if resources are constrained
        """
        status = self.check_resources()
        return not (status["memory_ok"] and status["cpu_ok"])

# Global instance
resource_monitor = ResourceMonitor()