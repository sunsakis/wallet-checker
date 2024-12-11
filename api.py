from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Only load dotenv in development
if os.getenv('ENVIRONMENT') != 'production':
    try:
        from dotenv import load_dotenv
        load_dotenv('.env.local')
        logger.info("Loaded environment from .env files")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping local env files")

from backend import (
    BlockchainDataProvider,
    EnhancedWalletAnalyzer,
    CacheConfig,
    BlockchainDataError
)

# Pydantic models for API responses
class Transaction(BaseModel):
    type: str
    protocol: str
    value_usd: float

class Portfolio(BaseModel):
    eth_percentage: float
    usdc_percentage: float

class ActivityPoint(BaseModel):
    month: str
    transactions: int

class Profitability(BaseModel): 
    status: str
    total_profit_loss: float
    profit_loss_percentage: float
    successful_trades: int
    total_trades: int

class TechnicalMetrics(BaseModel): 
    avg_gas_used: float
    total_transactions: int
    transaction_frequency: str

class WalletProfile(BaseModel):
    address: str
    profile_type: str
    risk_level: str
    activity_level: str
    main_activity: str
    last_active: str
    first_active: str
    total_value_usd: float
    portfolio: Portfolio
    recent_transactions: List[Transaction]
    activity_history: List[ActivityPoint]
    profitability: Profitability
    technical_metrics: TechnicalMetrics

app = FastAPI(title="Check Wallet", version="0.1")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.walletchecker.xyz",
        "https://xmsg-eta.vercel.app/",
        "http://localhost:5173",  # Vite's default dev port
        "http://localhost:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"status": "online", "service": "Crypto Wallet Analysis API"}

@app.get("/analyze/{address}")
async def analyze_wallet(address: str):
    try:
        # Get all environment variables once
        web3_url = os.getenv('WEB3_URL')
        etherscan_api_key = os.getenv('BASESCAN_API_KEY')
        coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        redis_url = os.getenv('REDIS_URL')

        # Validate required variables
        if not web3_url:
            raise HTTPException(
                status_code=500,
                detail="WEB3_URL environment variable is not set"
            )
        
        if not etherscan_api_key:
            raise HTTPException(
                status_code=500,
                detail="BASESCAN_API_KEY environment variable is not set"
            )

        cache_config = CacheConfig(
            redis_url=redis_url or 'redis://localhost:6379/0'
        )
        
        provider = BlockchainDataProvider(
            web3_url=web3_url,
            etherscan_api_key=etherscan_api_key,
            coingecko_api_key=coingecko_api_key,
            cache_config=cache_config
        )
        
        async with provider as p:
            analyzer = EnhancedWalletAnalyzer(data_provider=p, address=address)
            raw_analysis = await analyzer.analyze()
            
            # Structure the response to match what the frontend expects
            analysis = {
                'profile_type': raw_analysis['profile_type'],
                'risk_level': raw_analysis['risk_level'],
                'activity_level': raw_analysis['activity_level'],
                'main_activity': raw_analysis['main_activity'],
                'last_active': raw_analysis['last_active'],
                'first_active': raw_analysis['first_active'],
                'total_value_usd': raw_analysis['total_value_usd'],
                'portfolio': {
                    'tokens': raw_analysis['portfolio']['tokens'],
                    'prices': raw_analysis['portfolio']['prices'],
                    'usd_values': raw_analysis['portfolio']['usd_values'],
                    'percentages': raw_analysis['portfolio']['percentages']
                },
                'activity_history': raw_analysis['activity_history'],
                'technical_metrics': raw_analysis['technical_metrics'],
                'profitability_metrics': raw_analysis['profitability_metrics'],
                'behavioral_patterns': raw_analysis['behavioral_patterns']
            }
            
            return analysis
            
    except BlockchainDataError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)  # Add exc_info for better debugging
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)