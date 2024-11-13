import asyncio
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import redis
import json
from web3 import Web3, AsyncWeb3
from web3.exceptions import ContractLogicError, TransactionNotFound
import logging
from functools import wraps
import pandas as pd
from abc import ABC, abstractmethod
from eth_typing import Address
import backoff
from cachetools import TTLCache, cached
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exceptions
class BlockchainDataError(Exception):
    """Base exception for blockchain data retrieval errors"""
    pass

class RateLimitError(BlockchainDataError):
    """Raised when API rate limits are hit"""
    pass

class NetworkError(BlockchainDataError):
    """Raised for network-related issues"""
    pass

class DataValidationError(BlockchainDataError):
    """Raised when data validation fails"""
    pass

@dataclass
class CacheConfig:
    """Configuration for different cache layers"""
    redis_url: str
    redis_ttl: int = 3600  # 1 hour
    memory_ttl: int = 300  # 5 minutes
    memory_maxsize: int = 1000

class BaseCache(ABC):
    """Abstract base class for cache implementations"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

class RedisCache(BaseCache):
    """Redis cache implementation"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            serialized = json.dumps(value)
            self.redis.set(key, serialized, ex=ttl)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> None:
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

class MemoryCache(BaseCache):
    """In-memory cache implementation using TTLCache"""
    
    def __init__(self, maxsize: int, ttl: int):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
    
    async def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.cache[key] = value
    
    async def delete(self, key: str) -> None:
        self.cache.pop(key, None)

class BlockchainDataProvider:
    """Primary class for retrieving and caching blockchain data"""
    
    def __init__(
        self,
        web3_url: str,
        etherscan_api_key: str,
        cache_config: CacheConfig
    ):
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(web3_url))
        self.etherscan_api_key = etherscan_api_key
        self.redis_cache = RedisCache(cache_config.redis_url)
        self.memory_cache = MemoryCache(
            cache_config.memory_maxsize,
            cache_config.memory_ttl
        )
        
        # API rate limiting
        self.rate_limit = asyncio.Semaphore(5)  # 5 concurrent requests
        self.request_timestamps: List[float] = []
        self.max_requests_per_second = 5
    
    @backoff.on_exception(
        backoff.expo,
        (NetworkError, RateLimitError),
        max_tries=5
    )
    async def get_wallet_transactions(
        self,
        address: str,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None
    ) -> List[dict]:
        """Retrieve wallet transactions with caching and error handling"""
        cache_key = f"tx:{address}:{start_block}:{end_block}"
        
        # Try memory cache first
        if cached_data := await self.memory_cache.get(cache_key):
            logger.debug("Memory cache hit for transactions")
            return cached_data
        
        # Try Redis cache
        if cached_data := await self.redis_cache.get(cache_key):
            logger.debug("Redis cache hit for transactions")
            await self.memory_cache.set(cache_key, cached_data)
            return cached_data
        
        # Fetch from blockchain
        try:
            async with self.rate_limit:
                await self._check_rate_limit()
                
                async with aiohttp.ClientSession() as session:
                    params = {
                        'module': 'account',
                        'action': 'txlist',
                        'address': address,
                        'apikey': self.etherscan_api_key,
                        'sort': 'desc'
                    }
                    if start_block:
                        params['startblock'] = start_block
                    if end_block:
                        params['endblock'] = end_block
                    
                    async with session.get(
                        'https://api.etherscan.io/api',
                        params=params
                    ) as response:
                        if response.status != 200:
                            raise NetworkError(
                                f"API request failed: {response.status}"
                            )
                        
                        data = await response.json()
                        if data['status'] != '1':
                            raise BlockchainDataError(
                                f"API error: {data.get('message')}"
                            )
                        
                        transactions = data['result']
                        
                        # Validate and clean data
                        cleaned_txs = [
                            self._clean_transaction(tx) 
                            for tx in transactions
                        ]
                        
                        # Cache the results
                        await self.redis_cache.set(
                            cache_key,
                            cleaned_txs,
                            ttl=3600
                        )
                        await self.memory_cache.set(cache_key, cleaned_txs)
                        
                        return cleaned_txs
                        
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network request failed: {e}")
        except json.JSONDecodeError as e:
            raise DataValidationError(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    async def get_token_balances(
        self,
        address: str
    ) -> Dict[str, float]:
        """Retrieve token balances with caching"""
        cache_key = f"balances:{address}"
        
        # Try caches first
        if cached_data := await self.memory_cache.get(cache_key):
            return cached_data
        
        if cached_data := await self.redis_cache.get(cache_key):
            await self.memory_cache.set(cache_key, cached_data)
            return cached_data
        
        try:
            # Get ETH balance
            eth_balance = await self.w3.eth.get_balance(address)
            
            # Get ERC20 token balances (simplified example)
            # In production, you'd want to query multiple tokens
            balances = {
                'ETH': float(Web3.from_wei(eth_balance, 'ether'))
            }
            
            # Cache results
            await self.redis_cache.set(cache_key, balances, ttl=300)
            await self.memory_cache.set(cache_key, balances)
            
            return balances
            
        except Exception as e:
            logger.error(f"Error getting token balances: {e}")
            raise BlockchainDataError(f"Failed to get token balances: {e}")
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits"""
        current_time = datetime.now().timestamp()
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts <= 1
        ]
        
        if len(self.request_timestamps) >= self.max_requests_per_second:
            raise RateLimitError("Rate limit exceeded")
        
        self.request_timestamps.append(current_time)
    
    def _clean_transaction(self, tx: dict) -> dict:
        """Clean and validate transaction data"""
        required_fields = {'hash', 'from', 'to', 'value', 'gasUsed'}
        if not all(field in tx for field in required_fields):
            raise DataValidationError(f"Missing required fields in transaction")
        
        return {
            'hash': tx['hash'],
            'from_address': tx['from'],
            'to_address': tx['to'],
            'value': float(Web3.from_wei(int(tx['value']), 'ether')),
            'gas_used': int(tx['gasUsed']),
            'timestamp': datetime.fromtimestamp(int(tx['timeStamp']))
        }

class EnhancedWalletAnalyzer:
    """Enhanced version of WalletAnalyzer with blockchain data integration"""
    
    def __init__(
        self,
        data_provider: BlockchainDataProvider,
        address: str
    ):
        self.data_provider = data_provider
        self.address = address
    
    async def analyze(self) -> dict:
        """Perform complete wallet analysis"""
        try:
            # Fetch required data
            transactions = await self.data_provider.get_wallet_transactions(
                self.address
            )
            balances = await self.data_provider.get_token_balances(
                self.address
            )
            
            # Convert to pandas for analysis
            tx_df = pd.DataFrame(transactions)
            
            # Perform analysis
            analysis = {
                'executive_summary': self._generate_executive_summary(
                    tx_df,
                    balances
                ),
                'risk_assessment': self._assess_risks(tx_df),
                'behavioral_patterns': self._analyze_behavior(tx_df),
                'portfolio_metrics': self._analyze_portfolio(balances),
                'technical_metrics': self._calculate_technical_metrics(tx_df)
            }
            
            return analysis
            
        except BlockchainDataError as e:
            logger.error(f"Blockchain data error: {e}")
            raise
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    def _generate_executive_summary(
    self,
    tx_df: pd.DataFrame,
    balances: Dict[str, float]
    ) -> dict:
        if tx_df.empty:
            return {
                "summary": "No transactions found",
                "total_transactions": 0,
                "total_value": 0
            }
        
        return {
            "summary": "Active wallet",
            "total_transactions": len(tx_df),
            "total_value": balances.get('ETH', 0)
        }
    
    def _assess_risks(self, tx_df: pd.DataFrame) -> dict:
        if tx_df.empty:
            return {"overall_risk": "Unknown"}
        
        # Simple risk assessment based on transaction frequency
        tx_count = len(tx_df)
        if tx_count > 100:
            risk = "High"
        elif tx_count > 50:
            risk = "Medium"
        else:
            risk = "Low"
        
        return {
            "overall_risk": risk,
            "transaction_frequency": tx_count
        }
    
    def _analyze_behavior(self, tx_df: pd.DataFrame) -> dict:
        if tx_df.empty:
            return {
                "profile_type": "Inactive",
                "activity_level": "None",
                "main_activity": "None",
                "last_active": "Never",
                "activity_history": []
            }
        
        # Get last active time
        last_active = tx_df['timestamp'].max()
        
        # Calculate monthly activity
        tx_df['month'] = tx_df['timestamp'].dt.strftime('%b')
        monthly_counts = tx_df.groupby('month').size().reset_index()
        activity_history = [
            {"month": row['month'], "count": row[0]}
            for _, row in monthly_counts.iterrows()
        ]
        
        # Determine activity level
        tx_count = len(tx_df)
        if tx_count > 100:
            activity_level = "High"
        elif tx_count > 50:
            activity_level = "Medium"
        else:
            activity_level = "Low"
        
        return {
            "profile_type": "Active Trader" if tx_count > 50 else "Casual User",
            "activity_level": activity_level,
            "main_activity": "Trading",
            "last_active": last_active.isoformat(),
            "activity_history": activity_history
        }
    
    def _analyze_portfolio(self, balances: Dict[str, float]) -> dict:
        eth_balance = balances.get('ETH', 0)
        eth_value_usd = eth_balance * 2000  # Simplified ETH price
        
        return {
            "total_value_usd": eth_value_usd,
            "eth_percentage": 100,  # Simplified - assuming only ETH
            "usdc_percentage": 0,
            "tokens": {
                "ETH": eth_balance
            }
        }
    
    def _calculate_technical_metrics(self, tx_df: pd.DataFrame) -> dict:
        if tx_df.empty:
            return {
                "avg_gas_used": 0,
                "total_transactions": 0,
                "recent_transactions": []
            }
        
        recent_txs = []
        for _, tx in tx_df.head(3).iterrows():
            recent_txs.append({
                "type": "Transfer" if tx['value'] > 0 else "Contract Interaction",
                "protocol": "Ethereum",
                "value_usd": tx['value'] * 2000  # Simplified ETH price
            })
        
        return {
            "avg_gas_used": tx_df['gas_used'].mean(),
            "total_transactions": len(tx_df),
            "recent_transactions": recent_txs
        }

# Usage example
async def main():
    # Load configuration
    cache_config = CacheConfig(
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    )
    
    # Initialize provider
    provider = BlockchainDataProvider(
        web3_url=os.getenv('WEB3_URL'),
        etherscan_api_key=os.getenv('ETHERSCAN_API_KEY'),
        cache_config=cache_config
    )
    
    # Initialize analyzer
    analyzer = EnhancedWalletAnalyzer(
        data_provider=provider,
        address='0x...'  # Target wallet address
    )
    
    try:
        # Perform analysis
        analysis = await analyzer.analyze()
        print(json.dumps(analysis, indent=2))
    except BlockchainDataError as e:
        logger.error(f"Analysis failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())