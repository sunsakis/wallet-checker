from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional
from datetime import datetime
import os

try:
    from dotenv import load_dotenv
    load_dotenv('.env.local')
except ImportError:
    pass

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

class Profitability(BaseModel):  # Add new model
    status: str
    total_profit_loss: float
    profit_loss_percentage: float
    successful_trades: int
    total_trades: int

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

app = FastAPI(title="Wallet Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"status": "online", "service": "Wallet Analysis API"}

@app.get("/analyze/{address}", response_model=WalletProfile)
async def analyze_wallet(address: str):
    try:
        # Initialize provider with environment variables
        cache_config = CacheConfig(
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        )
        
        provider = BlockchainDataProvider(
            web3_url=os.getenv('WEB3_URL'),
            etherscan_api_key=os.getenv('ETHERSCAN_API_KEY'),
            cache_config=cache_config
        )
        
        # Use async context manager to properly handle resources
        async with provider as p:
            analyzer = EnhancedWalletAnalyzer(data_provider=p, address=address)
            analysis = await analyzer.analyze()

            profitability_data = analysis['profitability_metrics']
            
            return WalletProfile(
                address=address,
                profile_type=analysis['behavioral_patterns'].get('profile_type', 'Unknown'),
                risk_level=analysis['risk_assessment'].get('overall_risk', 'Medium'),
                activity_level=analysis['behavioral_patterns'].get('activity_level', 'Medium'),
                main_activity=analysis['behavioral_patterns'].get('main_activity', 'Trading'),
                last_active=analysis['behavioral_patterns'].get('last_active', 'Unknown'),
                first_active=analysis['behavioral_patterns'].get('first_active', 'Unknown'),
                total_value_usd=analysis['portfolio_metrics'].get('total_value_usd', 0.0),
                portfolio=Portfolio(
                    eth_percentage=analysis['portfolio_metrics'].get('eth_percentage', 0),
                    usdc_percentage=analysis['portfolio_metrics'].get('usdc_percentage', 0)
                ),
                recent_transactions=[
                    Transaction(
                        type=tx['type'],
                        protocol=tx['protocol'],
                        value_usd=tx['value_usd']
                    )
                    for tx in analysis.get('recent_transactions', [])[:3]
                ],
                activity_history=[
                    ActivityPoint(
                        month=point['month'],
                        transactions=point['count']
                    )
                    for point in analysis['behavioral_patterns'].get('activity_history', [])
                ],
                profitability=Profitability(
                    status=profitability_data['status'],
                    total_profit_loss=profitability_data['total_profit_loss'],
                    profit_loss_percentage=profitability_data['profit_loss_percentage'],
                    successful_trades=profitability_data['successful_trades'],
                    total_trades=profitability_data['total_trades']
                )
            )
            
    except BlockchainDataError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)