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
    def __init__(self, redis_url: str):
        self.available = False
        try:
            self.redis = redis.from_url(redis_url)
            self.available = True
        except (redis.ConnectionError, ConnectionRefusedError):
            logger.warning("Redis unavailable - falling back to no-op cache")
    
    async def get(self, key: str) -> Optional[Any]:
        if not self.available:
            return None
        try:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        except Exception:
            # Don't log every failed operation
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self.available:
            return
        try:
            serialized = json.dumps(value, default=self._datetime_handler)
            self.redis.set(key, serialized, ex=ttl)
        except Exception:
            # Don't log every failed operation
            pass
    
    async def delete(self, key: str) -> None:
        if not self.available:
            return
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    @staticmethod
    def _datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

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
        self._session = None
        self.rate_limit = asyncio.Semaphore(5)
        self.request_timestamps: List[float] = []
        self.max_requests_per_second = 5

    async def __aenter__(self):
        await self.get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def get_eth_price(self) -> float:
        """Get current ETH price in USD from Etherscan"""
        cache_key = "eth_price_usd"
        
        if cached_price := await self.memory_cache.get(cache_key):
            return cached_price
        
        try:
            async with self.rate_limit:
                await self._check_rate_limit()
                session = await self.get_session()
                
                params = {
                    'module': 'stats',
                    'action': 'ethprice',
                    'apikey': self.etherscan_api_key
                }
                
                async with session.get(
                    'https://api.etherscan.io/api',
                    params=params
                ) as response:
                    if response.status != 200:
                        raise NetworkError(f"API request failed: {response.status}")
                    
                    data = await response.json()
                    if data['status'] != '1':
                        raise BlockchainDataError(f"API error: {data.get('message')}")
                    
                    eth_price = float(data['result']['ethusd'])
                    await self.memory_cache.set(cache_key, eth_price, ttl=300)
                    return eth_price
                    
        except Exception as e:
            logger.error(f"Error getting ETH price: {e}")
            raise BlockchainDataError(f"Failed to get ETH price: {e}")

    async def get_wallet_transactions(
        self,
        address: str,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None
    ) -> List[dict]:
        cache_key = f"tx:{address}:{start_block}:{end_block}"
        
        if cached_data := await self.memory_cache.get(cache_key):
            return cached_data
            
        if cached_data := await self.redis_cache.get(cache_key):
            await self.memory_cache.set(cache_key, cached_data)
            return cached_data
            
        try:
            async with self.rate_limit:
                await self._check_rate_limit()
                session = await self.get_session()
                
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
                        raise NetworkError(f"API request failed: {response.status}")
                    
                    data = await response.json()
                    if data['status'] != '1':
                        raise BlockchainDataError(f"API error: {data.get('message')}")
                    
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
            'timestamp': datetime.fromtimestamp(int(tx['timeStamp'])).isoformat()  # Store as ISO string
        }

class EnhancedWalletAnalyzer:
    def __init__(
        self,
        data_provider: BlockchainDataProvider,
        address: str
    ):
        self.data_provider = data_provider
        self.address = address

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.data_provider.close()

    
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
            if not tx_df.empty:
                tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'])

            # Get technical metrics
            tech_metrics = self._calculate_technical_metrics(tx_df)
            behavior = self._analyze_behavior(tx_df)

            # Perform analysis
            return {
                'profile_type': tech_metrics['user_type'],
                'risk_level': 'Low',
                'activity_level': behavior['activity_level'],
                'main_activity': behavior['main_activity'],
                'last_active': behavior['last_active'],
                'first_active': behavior['first_active'],
                'total_value_usd': (await self._analyze_portfolio(balances))['total_value_usd'],
                'portfolio_metrics': await self._analyze_portfolio(balances),
                'activity_history': behavior['activity_history'],
                'technical_metrics': tech_metrics, 
                'profitability_metrics': self._assess_profitability(tx_df),
                'behavioral_patterns': {
                    'transaction_frequency': tech_metrics['transaction_frequency']
                }
            }
                
        except BlockchainDataError as e:
            logger.error(f"Blockchain data error: {e}")
            raise
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            raise BlockchainDataError(f"Failed to analyze wallet: {str(e)}")
    
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
            return {"overall_risk": 0}
        
        # Simple risk assessment based on transaction frequency
        tx_count = len(tx_df)
        
        return {
            "transaction_frequency": tx_count
        }
    
    def _analyze_behavior(self, tx_df: pd.DataFrame) -> dict:
        if tx_df.empty:
            return {
                "profile_type": "Inactive",
                "activity_level": "None",
                "main_activity": "None",
                "last_active": "Never",
                "first_active": "Never",
                "activity_history": []
            }
        
        # Convert timestamp strings to datetime objects if needed
        if isinstance(tx_df['timestamp'].iloc[0], str):
            tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'])
        
        # Calculate monthly activity
        tx_df['month'] = tx_df['timestamp'].dt.strftime('%b')
        monthly_counts = tx_df.groupby('month').size().reset_index()
        activity_history = [
            {
                "month": row['month'],
                "count": int(row[0])  # Change back to 'count'
            }
            for _, row in monthly_counts.iterrows()
        ]
        
        # Get last active time and calculate time difference
        last_tx_time = tx_df['timestamp'].max()
        first_tx_time = tx_df['timestamp'].min()
        now = datetime.now()
        
        # Format time differences for both last and first active
        def format_time_diff(time_diff):
            if time_diff.total_seconds() < 24 * 3600:  # Less than 24 hours
                hours = int(time_diff.total_seconds() / 3600)
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif time_diff.days < 30:  # Less than 30 days
                days = time_diff.days
                return f"{days} day{'s' if days != 1 else ''} ago"
            elif time_diff.days < 365:  # Less than a year
                months = int(time_diff.days / 30)
                return f"{months} month{'s' if months != 1 else ''} ago"
            else:
                years = int(time_diff.days / 365)
                return f"{years} year{'s' if years != 1 else ''} ago"

        last_active = format_time_diff(now - last_tx_time)
        first_active = format_time_diff(now - first_tx_time)
        
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
            "last_active": last_active,
            "first_active": first_active,
            "activity_history": activity_history
        }
    
    async def _analyze_portfolio(self, balances: Dict[str, float]) -> dict:
        eth_balance = balances.get('ETH', 0)
        eth_price = await self.data_provider.get_eth_price()
        eth_value_usd = round(eth_balance * eth_price, 2)
        
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
                "transaction_frequency": "None",
                "user_type": "Inactive - No transactions found"
            }
        
        # Calculate metrics
        avg_gas = float(tx_df['gas_used'].mean())
        gas_std = float(tx_df['gas_used'].std())
        tx_count = len(tx_df)
        
        # Determine user type based on combined patterns
        user_type = self._determine_user_type(
            avg_gas=avg_gas,
            gas_std=gas_std,
            tx_count=tx_count
        )

        return {
            "avg_gas_used": avg_gas,
            "total_transactions": tx_count,
            "transaction_frequency": self._calculate_frequency(tx_count),
            "user_type": user_type
        }

    def _determine_user_type(self, avg_gas: float, gas_std: float, tx_count: int) -> str:
        # Handle very low transaction counts separately to avoid misclassification
        if tx_count < 5:
            return "New User - Too few transactions for classification"
        
        # Define thresholds
        LOW_GAS = 50000
        MED_GAS = 150000
        HIGH_GAS = 300000
        
        # Calculate consistency score (lower means more consistent)
        gas_consistency = gas_std / avg_gas if avg_gas > 0 else 0
        
        # Classification logic
        if tx_count > 100:  # Very active users
            if avg_gas > HIGH_GAS and gas_consistency > 1.5:
                return "Bot"
            elif avg_gas > MED_GAS:
                return "Trader"
            else:
                return "Trader"
                
        elif tx_count > 30:  # Moderately active users
            if avg_gas < LOW_GAS and gas_consistency < 0.5:
                return "Hodler"
            elif LOW_GAS <= avg_gas <= MED_GAS:
                return "Trader"
            else:
                return "Trader"
                
        else:  # Low activity users
            if avg_gas < LOW_GAS:
                return "Investor"
            elif avg_gas > HIGH_GAS:
                return "Opportunist"
            else:
                return "Casual"

    def _calculate_frequency(self, tx_count: int) -> str:
        if tx_count > 100:
            return "Very High"
        elif tx_count > 50:
            return "High"
        elif tx_count > 20:
            return "Medium"
        elif tx_count > 5:
            return "Low"
        return "Very Low"
    
    def _assess_profitability(self, tx_df: pd.DataFrame) -> dict:
        if tx_df.empty:
            return {
                "status": "Unknown",
                "total_profit_loss": 0,
                "profit_loss_percentage": 0,
                "successful_trades": 0,
                "total_trades": 0
            }
        
        try:
            # Calculate inflows and outflows
            tx_df['inflow'] = tx_df.apply(
                lambda x: float(x['value']) if x['to_address'].lower() == self.address.lower() else 0,
                axis=1
            )
            tx_df['outflow'] = tx_df.apply(
                lambda x: float(x['value']) if x['from_address'].lower() == self.address.lower() else 0,
                axis=1
            )
            
            total_inflow = tx_df['inflow'].sum()
            total_outflow = tx_df['outflow'].sum()
            net_position = total_inflow - total_outflow
            
            # Calculate profit/loss percentage
            if total_outflow > 0:
                profit_loss_percentage = (net_position / total_outflow) * 100
            else:
                profit_loss_percentage = 0
            
            # Count successful trades (where inflow > outflow)
            successful_trades = len(tx_df[tx_df['inflow'] > tx_df['outflow']])
            total_trades = len(tx_df)
            
            # Determine profitability status
            if profit_loss_percentage > 20:
                status = "Highly Profitable"
            elif profit_loss_percentage > 5:
                status = "Profitable"
            elif profit_loss_percentage > -5:
                status = "Break Even"
            elif profit_loss_percentage > -20:
                status = "Loss Making"
            else:
                status = "High Loss"
                
            return {
                "status": status,
                "total_profit_loss": round(net_position, 4),
                "profit_loss_percentage": round(profit_loss_percentage, 2),
                "successful_trades": successful_trades,
                "total_trades": total_trades
            }
            
        except Exception as e:
            logger.error(f"Error calculating profitability: {e}")
            return {
                "status": "Error",
                "total_profit_loss": 0,
                "profit_loss_percentage": 0,
                "successful_trades": 0,
                "total_trades": 0
            }

async def analyze_wallet(address: str) -> dict:
    cache_config = CacheConfig(
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    )
    
    provider = BlockchainDataProvider(
        web3_url=os.getenv('WEB3_URL'),
        etherscan_api_key=os.getenv('ETHERSCAN_API_KEY'),
        cache_config=cache_config
    )
    
    async with provider as p:
        analyzer = EnhancedWalletAnalyzer(p, address)
        return await analyzer.analyze()

async def main():
    try:
        address = '0x...'  # Target wallet address
        analysis = await analyze_wallet(address)
        print(json.dumps(analysis, indent=2))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())