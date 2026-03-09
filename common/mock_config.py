"""
Mock mode configuration sharing between miner and mock_validator processes.
Uses multiprocessing shared_memory for in-memory IPC communication without parent-child relationship.
"""
import asyncio
import json
import time
from multiprocessing import shared_memory
from typing import Optional
from loguru import logger


class MockConfigSharedMemory:
    """Manages configuration sharing for mock mode using shared memory."""
    
    SHARED_MEMORY_NAME = "mock_validator_config"
    SHARED_MEMORY_SIZE = 4096  # 4KB should be enough for config
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize MockConfigSharedMemory.
        
        Args:
            name: Optional name for the shared memory block. Defaults to SHARED_MEMORY_NAME.
        """
        self.name = name or self.SHARED_MEMORY_NAME
        self.shm = None
    
    def write(self, config: dict) -> bool:
        """
        Write configuration to shared memory.
        Creates shared memory block if it doesn't exist.
        
        Args:
            config: Dictionary containing configuration data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize config to JSON
            config_bytes = json.dumps(config).encode('utf-8')
            
            if len(config_bytes) > self.SHARED_MEMORY_SIZE - 4:
                logger.error(f"[MockConfig] Config too large: {len(config_bytes)} bytes (max: {self.SHARED_MEMORY_SIZE})")
                return False
            
            # Try to unlink existing shared memory first
            try:
                existing_shm = shared_memory.SharedMemory(name=self.name)
                existing_shm.close()
                existing_shm.unlink()
            except FileNotFoundError:
                pass  # Shared memory doesn't exist, which is fine
            
            # Create new shared memory
            self.shm = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=self.SHARED_MEMORY_SIZE
            )
            
            # Write data
            self.shm.buf[:len(config_bytes)] = config_bytes
            # Write length at the end (last 4 bytes)
            length_bytes = len(config_bytes).to_bytes(4, byteorder='big')
            self.shm.buf[-4:] = length_bytes
            
            logger.info(f"[MockConfig] Configuration written to shared memory '{self.name}' ({len(config_bytes)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"[MockConfig] Failed to write config to shared memory: {e}")
            return False
    
    def read(self) -> Optional[dict]:
        """
        Read configuration from shared memory.
        
        Returns:
            Configuration dictionary if successful, None otherwise
        """
        from multiprocessing import resource_tracker
        
        shm_temp = None
        try:
            # Attach to existing shared memory
            shm_temp = shared_memory.SharedMemory(name=self.name)
            
            # Unregister from resource_tracker - we're not the owner
            # This prevents resource_tracker from unlinking the shared memory on process exit
            try:
                resource_tracker.unregister(shm_temp._name, "shared_memory")
            except Exception:
                pass  # Ignore if already unregistered
            
            # Read length from last 4 bytes
            length_bytes = bytes(shm_temp.buf[-4:])
            length = int.from_bytes(length_bytes, byteorder='big')
            
            if length == 0 or length > self.SHARED_MEMORY_SIZE - 4:
                logger.warning(f"[MockConfig] Invalid length in shared memory: {length}")
                return None
            
            # Read data
            config_bytes = bytes(shm_temp.buf[:length])
            config = json.loads(config_bytes.decode('utf-8'))
            
            logger.info(f"[MockConfig] Configuration read from shared memory '{self.name}' ({length} bytes)")
            
            # Store the reference for later cleanup
            if self.shm is None:
                self.shm = shm_temp
                shm_temp = None  # Prevent closing in finally block
            
            return config
            
        except FileNotFoundError:
            logger.warning(f"[MockConfig] Shared memory '{self.name}' not found")
            return None
        except Exception as e:
            logger.error(f"[MockConfig] Failed to read config from shared memory: {e}")
            return None
        finally:
            # Close temporary connection if we didn't store it
            if shm_temp is not None:
                try:
                    shm_temp.close()
                except Exception:
                    pass

    def wait_for_config(self, shutdown_event: asyncio.Event | None, timeout: int = 30, poll_interval: float = 0.5) -> Optional[dict]:
        """
        Wait for configuration to be available in shared memory.
        
        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: Time between checks in seconds
            
        Returns:
            Configuration dictionary if successful, None otherwise
        """
        start_time = time.time()

        while time.time() - start_time < timeout and (shutdown_event is None or not shutdown_event.is_set()):
            config = self.read()
            if config is not None:
                return config
            time.sleep(poll_interval)
        
        logger.error("[MockConfig] Timeout waiting for configuration in shared memory")
        return None
    
    def cleanup(self, unlink: bool = False) -> bool:
        """
        Clean up shared memory resources.
        Should be called when done with shared memory.
        
        Args:
            unlink: If True, unlink (delete) the shared memory. 
                   Only the creator (miner) should set this to True.
                   Readers (validator) should set this to False to only close the connection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.shm is not None:
                self.shm.close()
                if unlink:
                    try:
                        self.shm.unlink()
                        logger.info(f"[MockConfig] Shared memory '{self.name}' unlinked and closed")
                    except FileNotFoundError:
                        pass  # Already unlinked
                else:
                    logger.info(f"[MockConfig] Shared memory '{self.name}' connection closed")
                self.shm = None
            return True
        except Exception as e:
            logger.error(f"[MockConfig] Failed to cleanup shared memory: {e}")
            return False
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if self.shm is not None:
            try:
                self.shm.close()
            except:
                pass
