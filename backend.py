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
from dotenv import load_dotenv
load_dotenv()

# Validate required environment variables
web3_url = os.getenv('WEB3_URL')
etherscan_api_key = os.getenv('BASESCAN_API_KEY')
if not web3_url or not etherscan_api_key:
    raise ValueError("Missing required environment variables: WEB3_URL and BASESCAN_API_KEY must be set")

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
        coingecko_api_key: str,
        cache_config: CacheConfig
    ):
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(web3_url))
        self.etherscan_api_key = etherscan_api_key
        self.coingecko_api_key = coingecko_api_key 
        self.redis_cache = RedisCache(cache_config.redis_url)
        self.memory_cache = MemoryCache(
            cache_config.memory_maxsize,
            cache_config.memory_ttl
        )
        self._session = None

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
            session = await self.get_session()
            
            params = {
                'module': 'stats',
                'action': 'ethprice',
                'apikey': self.etherscan_api_key
            }
            
            async with session.get(
                'https://api.basescan.org/api',
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
        
    async def get_gas_price(self) -> int:
        """Get current gas price in wei"""
        cache_key = "gas_price"
        
        if cached_price := await self.memory_cache.get(cache_key):
            return cached_price
            
        try:
            session = await self.get_session()
            
            # Use Base's JSON-RPC endpoint instead
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_gasPrice",
                "params": [],
                "id": 1
            }
            
            async with session.post(
                'https://mainnet.base.org',  # Base RPC endpoint
                json=payload
            ) as response:
                if response.status != 200:
                    raise NetworkError(f"API request failed: {response.status}")
                
                data = await response.json()
                if 'error' in data:
                    raise BlockchainDataError(f"RPC error: {data['error']}")
                
                gas_price = int(data['result'], 16)  # Convert hex string to int
                
                # Cache the result
                await self.memory_cache.set(cache_key, gas_price, ttl=60)  # Cache for 1 minute
                
                return gas_price
                    
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            raise BlockchainDataError(f"Failed to get gas price: {e}")

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
                'https://api.basescan.org/api',
                params=params
            ) as response:
                if response.status != 200:
                    raise NetworkError(f"API request failed: {response.status}")
                
                data = await response.json()
                if data['status'] != '1':
                    if data['message'] == 'No transactions found':
                        return []
                    raise BlockchainDataError(f"API error: {data.get('message')}")
                
                transactions = data['result']
                
                # Clean and format transactions
                cleaned_txs = []
                for tx in transactions:
                    cleaned_tx = {
                        'hash': tx['hash'],
                        'from_address': tx['from'],
                        'to_address': tx['to'],
                        'value': float(Web3.from_wei(int(tx['value']), 'ether')),
                        'gas_used': int(tx['gasUsed']),
                        'timestamp': datetime.fromtimestamp(int(tx['timeStamp'])).isoformat()
                    }
                    cleaned_txs.append(cleaned_tx)
                
                # Cache the results
                await self.redis_cache.set(cache_key, cleaned_txs, ttl=3600)
                await self.memory_cache.set(cache_key, cleaned_txs)
                
                return cleaned_txs
                        
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network request failed: {e}")
        except json.JSONDecodeError as e:
            raise DataValidationError(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    async def get_token_balances(self, address: str) -> Dict[str, Dict]:
        """Retrieve all token balances using Blockscout API"""
        cache_key = f"balances:{address}"
        
        logger.info(f"Getting token balances for address: {address}")
        
        try:
            # Get ETH balance
            eth_balance = await self.w3.eth.get_balance(address)
            logger.info(f"ETH balance: {eth_balance}")
            
            balances = {}
            balances['ETH'] = {
                'amount': float(Web3.from_wei(eth_balance, 'ether')),
                'symbol': 'ETH',
                'contract_address': None  # Native ETH has no contract
            }
            
            # Get ERC20 token balances using Blockscout API
            session = await self.get_session()
            
            api_url = f"https://base.blockscout.com/api/v2/addresses/{address}/tokens"
            
            async with session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Raw API response: {json.dumps(data, indent=2)}")  # Debug log the raw response
                    
                    if 'items' in data:
                        for token in data['items']:
                            try:
                                # Check if token data exists and has required fields
                                if not isinstance(token, dict):
                                    continue
                                    
                                token_data = token.get('token', {})
                                if not token_data:
                                    continue
                                    
                                # Extract required fields with defaults
                                symbol = token_data.get('symbol')
                                address = token_data.get('address')
                                decimals = token_data.get('decimals')
                                value = token.get('value')
                                
                                # Skip if missing required data
                                if not all([symbol, address, decimals, value]):
                                    logger.debug(f"Skipping token due to missing data: {token_data}")
                                    continue
                                
                                # Create token key
                                token_key = symbol
                                if token_key in balances:
                                    token_key = f"{symbol}-{address[:6]}"
                                
                                # Convert balance
                                try:
                                    decimals = int(decimals)
                                    balance = float(value) / (10 ** decimals)
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Error converting balance for {symbol}: {e}")
                                    continue
                                
                                # Only add tokens with non-zero balance
                                if balance > 0:
                                    balances[token_key] = {
                                        'amount': balance,
                                        'symbol': symbol,
                                        'contract_address': address,
                                        'decimals': decimals
                                    }
                                    logger.info(f"Added token {symbol} with balance {balance}")
                                    
                            except Exception as e:
                                symbol = token.get('token', {}).get('symbol', 'unknown')
                                logger.warning(f"Error processing token {symbol}: {str(e)}")
                                logger.debug(f"Problematic token data: {json.dumps(token, indent=2)}")
                                continue
                else:
                    logger.warning(f"Blockscout API returned status {response.status}")
                    response_text = await response.text()
                    logger.warning(f"Response: {response_text}")
            
            # Cache results if we have any
            if balances:
                await self.redis_cache.set(cache_key, balances, ttl=300)
                await self.memory_cache.set(cache_key, balances)
            
            logger.info(f"Final balances: {json.dumps(balances, indent=2)}")
            return balances
                
        except Exception as e:
            logger.error(f"Error getting token balances: {e}")
            raise BlockchainDataError(f"Failed to get token balances: {str(e)}")
        
    async def get_token_prices(self, tokens: Dict[str, Dict]) -> Dict[str, float]:
        """Get token prices in USD using Uniswap V3 on Base"""
        cache_key = f"token_prices:{','.join(sorted(tokens.keys()))}"
        logger.info(f"Getting prices for tokens: {list(tokens.keys())}")
        
        if cached_data := await self.memory_cache.get(cache_key):
            return cached_data
                
        try:
            prices = {}
            
            # Handle ETH separately
            if 'ETH' in tokens:
                eth_price = await self.get_eth_price()
                logger.info(f"ETH price: ${eth_price}")
                prices['ETH'] = eth_price

            # Uniswap V3 Factory ABI
            factory_abi = [
                {
                    "inputs": [
                        {"internalType": "address", "name": "tokenA", "type": "address"},
                        {"internalType": "address", "name": "tokenB", "type": "address"},
                        {"internalType": "uint24", "name": "fee", "type": "uint24"}
                    ],
                    "name": "getPool",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]

            # Uniswap V3 Pool ABI
            pool_abi = [
                {
                    "inputs": [],
                    "name": "slot0",
                    "outputs": [
                        {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
                        {"internalType": "int24", "name": "tick", "type": "int24"},
                        {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
                        {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
                        {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
                        {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
                        {"internalType": "bool", "name": "unlocked", "type": "bool"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "token0",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "token1",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]

            # Uniswap V3 Factory address on Base
            factory_address = "0x33128a8fC17869897dcE68Ed026d694621f6FDfD"
            factory_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(factory_address),
                abi=factory_abi
            )

            # WETH and USDC addresses on Base
            weth_address = "0x4200000000000000000000000000000000000006"
            usdc_address = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

            # Common pool fees in Uniswap V3
            fee_tiers = [100, 500, 3000, 10000]  # 0.01%, 0.05%, 0.3%, 1%

            for token_key, token_data in tokens.items():
                if token_key not in prices and token_data.get('contract_address'):
                    try:
                        token_address = Web3.to_checksum_address(token_data['contract_address'])
                        pool_found = False

                        # Try different fee tiers with WETH
                        for fee in fee_tiers:
                            try:
                                logger.info(f"Checking {token_key} with fee tier {fee}")
                                pool_address = await factory_contract.functions.getPool(
                                    token_address,
                                    weth_address,
                                    fee
                                ).call()

                                logger.info(f"Found pool address: {pool_address}")

                                if pool_address and pool_address != "0x0000000000000000000000000000000000000000":
                                    pool_contract = self.w3.eth.contract(
                                        address=Web3.to_checksum_address(pool_address),
                                        abi=pool_abi
                                    )

                                    # Get token order
                                    logger.info(f"Getting token order for pool {pool_address}")
                                    token0_address = await pool_contract.functions.token0().call()
                                    token1_address = await pool_contract.functions.token1().call()
                                    is_token0 = token_address.lower() == token0_address.lower()

                                    logger.info(f"Token0: {token0_address}")
                                    logger.info(f"Token1: {token1_address}")
                                    logger.info(f"Is token0: {is_token0}")

                                    # Get current price from slot0
                                    logger.info("Getting slot0 data")
                                    slot0 = await pool_contract.functions.slot0().call()
                                    sqrt_price_x96 = slot0[0]
                                    logger.info(slot0)
                                    
                                    # Calculate price from sqrtPriceX96
                                    try:
                                        token0_decimals = token_data['decimals'] if is_token0 else 18  # WETH decimals
                                        token1_decimals = 18 if is_token0 else token_data['decimals']  # WETH decimals

                                        if is_token0:
                                            price = ((sqrt_price_x96 / (2**96))**2) / (10**token1_decimals / 10**token0_decimals)
                                        else:
                                            price = ((sqrt_price_x96 / (2**96))**2) * (10**token0_decimals / 10**token1_decimals)

                                        # Convert to USD
                                        token_price = price * eth_price

                                        # Add debug logging
                                        logger.info(f"sqrtPriceX96: {sqrt_price_x96}")
                                        logger.info(f"raw_price: {price}")
                                        logger.info(f"eth_price: {eth_price}")
                                        logger.info(f"Final USD price for {token_key}: ${token_price}")

                                        # Sanity check - if price is unreasonable, set to 0
                                        if token_price > 1000:  # Assuming no token should be worth more than $1000
                                            logger.warning(f"Price seems unreasonable for {token_key}, setting to 0")
                                            token_price = 0
                                        
                                        prices[token_key] = token_price
                                        pool_found = True
                                        break
                                    except Exception as e:
                                        logger.error(f"Error calculating price: {e}")
                                        continue

                            except Exception as e:
                                logger.debug(f"Failed to get price from {fee} fee tier: {e}")
                                continue

                        if not pool_found:
                            # Try USDC pairs if WETH pairs failed
                            for fee in fee_tiers:
                                try:
                                    pool_address = await factory_contract.functions.getPool(
                                        token_address,
                                        usdc_address,
                                        fee
                                    ).call()

                                    if pool_address and pool_address != "0x0000000000000000000000000000000000000000":
                                        pool_contract = self.w3.eth.contract(
                                            address=Web3.to_checksum_address(pool_address),
                                            abi=pool_abi
                                        )

                                        token0_address = await pool_contract.functions.token0().call()
                                        is_token0 = token_address.lower() == token0_address.lower()

                                        slot0 = await pool_contract.functions.slot0().call()
                                        sqrt_price_x96 = slot0[0]

                                        token0_decimals = token_data['decimals'] if is_token0 else 6  # USDC decimals
                                        token1_decimals = 6 if is_token0 else token_data['decimals']  # USDC decimals

                                        if is_token0:
                                            price = ((sqrt_price_x96 / (2**96))**2) / (10**token1_decimals / 10**token0_decimals)
                                        else:
                                            price = ((sqrt_price_x96 / (2**96))**2) * (10**token0_decimals / 10**token1_decimals)

                                        token_price = price  # Already in USD since paired with USDC

                                        logger.info(f"Got price for {token_key} from USDC pair: ${token_price}")
                                        prices[token_key] = token_price
                                        pool_found = True
                                        break

                                except Exception as e:
                                    logger.debug(f"Failed to get price from USDC pair: {e}")
                                    continue

                        if not pool_found:
                            logger.warning(f"No Uniswap V3 pools found for {token_key}")
                            prices[token_key] = 0

                    except Exception as e:
                        logger.warning(f"Failed to get price for {token_key}: {e}")
                        prices[token_key] = 0

            # Cache the results
            if prices:
                await self.memory_cache.set(cache_key, prices, ttl=300)
            logger.info(f"Final prices: {json.dumps(prices, indent=2)}")
            return prices
                    
        except Exception as e:
            logger.warning(f"Error fetching token prices: {e}")
            return prices

    async def get_token_info(self, token_symbol: str) -> Optional[Dict]:
        """Get token information from cache or Etherscan"""
        cache_key = f"token_info:{token_symbol}"
        
        if cached_data := await self.memory_cache.get(cache_key):
            return cached_data
        
        return None  # If not found in cache

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
            transactions = await self.data_provider.get_wallet_transactions(self.address)
            balances = await self.data_provider.get_token_balances(self.address)
            portfolio_analysis = await self._analyze_portfolio(balances)
            
            # Convert to pandas for analysis
            tx_df = pd.DataFrame(transactions)
            if not tx_df.empty:
                tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'])

            # Get technical metrics
            tech_metrics = await self._calculate_technical_metrics(tx_df)  # Added await here
            behavior = self._analyze_behavior(tx_df)

            # Perform analysis
            return {
                        'profile_type': tech_metrics['user_type'],
                        'risk_level': 'Low',
                        'activity_level': behavior['activity_level'],
                        'main_activity': behavior['main_activity'],
                        'last_active': behavior['last_active'],
                        'first_active': behavior['first_active'],
                        'total_value_usd': portfolio_analysis['total_value_usd'],
                        'portfolio': {
                            'tokens': balances,
                            'prices': portfolio_analysis['token_prices'],
                            'usd_values': portfolio_analysis['usd_values'],
                            'percentages': portfolio_analysis['percentages']
                        },
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
        
        # Calculate daily activity
        tx_df['day'] = tx_df['timestamp'].dt.strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
        daily_counts = tx_df.groupby('day').size().reset_index()
        activity_history = [
            {
                "day": row['day'],
                "count": int(row[0])
            }
            for _, row in daily_counts.iterrows()
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
    
    async def _analyze_portfolio(self, balances: Dict[str, Dict]) -> dict:
        """Analyze portfolio composition with USD values"""
        token_prices = await self.data_provider.get_token_prices(balances)
        
        portfolio_usd = {}
        total_usd = 0
        
        for token_key, token_data in balances.items():
            price = token_prices.get(token_key, 0)
            amount = token_data['amount']
            
            usd_value = amount * price
            portfolio_usd[token_key] = usd_value
            total_usd += usd_value
        
        percentages = {
            token: (value / total_usd * 100 if total_usd > 0 else 0)
            for token, value in portfolio_usd.items()
        }
        
        return {
            "total_value_usd": total_usd,
            "tokens": {k: v['amount'] for k, v in balances.items()},
            "token_prices": token_prices,
            "usd_values": portfolio_usd,
            "percentages": percentages
        }
    

    async def _calculate_technical_metrics(self, tx_df: pd.DataFrame) -> dict:
        if tx_df.empty:
            return {
                "avg_gas_used": 0,
                "avg_gas_price": 0,
                "avg_gas_paid_usd": 0,
                "total_transactions": 0,
                "transaction_frequency": "None",
                "user_type": "Inactive - No transactions found"
            }
        
        # Calculate metrics
        avg_gas = float(tx_df['gas_used'].mean())
        gas_std = float(tx_df['gas_used'].std())
        tx_count = len(tx_df)

        # Get current gas price and ETH price
        gas_price = await self.data_provider.get_gas_price()
        eth_price = await self.data_provider.get_eth_price()
        
        # Calculate average gas paid in USD
        # Convert wei to ETH: divide by 1e18
        # Then multiply by ETH price to get USD value
        avg_gas_paid_usd = (avg_gas * gas_price * eth_price) / 1e18
        
        # Determine user type based on combined patterns
        user_type = self._determine_user_type(
            avg_gas=avg_gas,
            gas_std=gas_std,
            tx_count=tx_count
        )

        return {
            "avg_gas_used": avg_gas,
            "avg_gas_price": gas_price,
            "eth_price": eth_price,
            "avg_gas_paid_usd": avg_gas_paid_usd,
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
        etherscan_api_key=os.getenv('BASESCAN_API_KEY'),
        coingecko_api_key=os.getenv('COINGECKO_API_KEY'),
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