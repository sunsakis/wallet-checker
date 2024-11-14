import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Wallet, AlertTriangle, TrendingUp, Clock, DollarSign} from 'lucide-react';

const WalletAnalysisDashboard = () => {
  const [address, setAddress] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [analysisData, setAnalysisData] = useState(null);

  const validateAddress = (addr) => {
    return /^0x[a-fA-F0-9]{40}$/.test(addr);
  };

  const handleAnalyze = async () => {
    if (!validateAddress(address)) {
      setError('Please enter a valid Ethereum address');
      return;
    }
  
    setLoading(true);
    setError('');
  
    try {
      const response = await fetch(`http://localhost:8000/analyze/${address}`);
      if (!response.ok) {
        throw new Error('Failed to fetch wallet data');
      }
      
      const data = await response.json();
      setAnalysisData({
        profile: {
          status: data.profile_type,
          riskLevel: data.risk_level,
          activityLevel: data.activity_level,
          mainActivity: data.main_activity,
          lastActive: data.last_active,
          firstActive: data.first_active,
          totalValue: `${data.total_value_usd.toLocaleString()}`,
          profitability: {
            status: data.profitability?.status || 'Unknown',
            total_profit_loss: data.profitability?.total_profit_loss || 0,
            profit_loss_percentage: data.profitability?.profit_loss_percentage || 0,
            successful_trades: data.profitability?.successful_trades || 0,
            total_trades: data.profitability?.total_trades || 0
          }
        },
        portfolio: {
          eth: data.portfolio.eth_percentage,
          usdc: data.portfolio.usdc_percentage,
          tokens: data.portfolio_metrics?.tokens || {}
        },
        recentTransactions: data.recent_transactions.map(tx => ({
          type: tx.type,
          protocol: tx.protocol,
          value: `$${tx.value_usd.toLocaleString()}`
        })),
        activityData: data.activity_history,
        technicalMetrics: {
          avgGasUsed: data.technical_metrics?.avg_gas_used,
          totalTransactions: data.technical_metrics?.total_transactions,
          txFrequency: data.behavioral_patterns?.transaction_frequency
        },
        executiveSummary: data.executive_summary
      });
    } catch (err) {
      setError('Failed to fetch wallet analysis: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const WalletProfile = ({ profile }) => (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Wallet className="h-6 w-6" />
          Wallet Profile
        </CardTitle>
        <CardDescription>Overview and key metrics</CardDescription>
      </CardHeader>
      <CardContent>
      {/* <div>
        <p className="text-sm font-medium">Profitability</p>
        <p className="text-2xl font-bold flex items-center gap-2">
          <TrendingUp className={`h-5 w-5 ${
            profile.profitability?.status === 'Highly Profitable' ? 'text-green-500' : 
            profile.profitability?.status === 'Profitable' ? 'text-green-400' :
            profile.profitability?.status === 'Break Even' ? 'text-yellow-500' :
            profile.profitability?.status === 'Loss Making' ? 'text-red-400' : 'text-red-500'
          }`} />
          {profile.profitability?.status || 'Unknown'}
        </p>
        <div className="mt-2 space-y-2">
          <div className="flex justify-between text-sm">
            <span>Total P/L:</span>
            <span className={profile.profitability?.total_profit_loss >= 0 ? 'text-green-500' : 'text-red-500'}>
              {profile.profitability?.total_profit_loss >= 0 ? '+' : ''}
              {profile.profitability?.total_profit_loss?.toFixed(4) || 0} ETH
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span>P/L %:</span>
            <span className={profile.profitability?.profit_loss_percentage >= 0 ? 'text-green-500' : 'text-red-500'}>
              {profile.profitability?.profit_loss_percentage >= 0 ? '+' : ''}
              {profile.profitability?.profit_loss_percentage?.toFixed(2) || 0}%
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Success Rate:</span>
            <span>
              {profile.profitability?.successful_trades && profile.profitability?.total_trades
                ? ((profile.profitability.successful_trades / profile.profitability.total_trades) * 100).toFixed(1)
                : '0'}%
            </span>
          </div>
        </div>
        </div> */}
        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="text-sm font-medium">Status</p>
            <p className="text-2xl font-bold">{profile.status}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Risk Level</p>
            <p className="text-2xl font-bold flex items-center gap-2">
              <AlertTriangle className={`h-5 w-5 ${
                profile.riskLevel === 'High' ? 'text-red-500' : 
                profile.riskLevel === 'Medium' ? 'text-yellow-500' : 'text-green-500'
              }`} />
              {profile.riskLevel}
            </p>
          </div>
          <div>
            <p className="text-sm font-medium">Activity Level</p>
            <p className="text-2xl font-bold flex items-center gap-2">
              <Activity className="h-5 w-5" />
              {profile.activityLevel}
            </p>
          </div>
          <div>
            <p className="text-sm font-medium">First Transaction</p>
            <p className="text-xl font-bold flex items-center gap-2">
              <Activity className="h-5 w-5" />
              {profile.firstActive}
            </p>
          </div>
          <div>
            <p className="text-sm font-medium">Last Active</p>
            <p className="text-xl font-bold flex items-center gap-2">
              <Clock className="h-5 w-5" />
              {profile.lastActive}
            </p>
          </div>
          <div>
            <p className="text-sm font-medium">Total Value</p>
            <p className="text-2xl font-bold flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              {profile.totalValue}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const TechnicalMetrics = ({ metrics }) => (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-6 w-6" />
          Technical Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <p className="text-sm font-medium">Average Gas Used</p>
            <p className="text-2xl font-bold">{metrics.avgGasUsed?.toFixed(0) || 'N/A'}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Total Transactions</p>
            <p className="text-2xl font-bold">{metrics.totalTransactions || 0}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Transaction Frequency</p>
            <p className="text-2xl font-bold">{metrics.txFrequency || 'Low'}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const PortfolioAllocation = ({ portfolio }) => (
    <Card>
      <CardHeader>
        <CardTitle>Portfolio Allocation</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {Object.entries(portfolio.tokens).map(([token, amount]) => (
            <div key={token}>
              <div className="flex justify-between mb-1">
                <span>{token}</span>
                <span>{amount.toFixed(4)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${portfolio[token.toLowerCase()] || 0}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );

  const RecentTransactions = ({ transactions }) => (
    <Card>
      <CardHeader>
        <CardTitle>Recent Transactions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {transactions.map((tx, index) => (
            <div key={index} className="flex justify-between items-center border-b pb-2">
              <div>
                <p className="font-medium">{tx.type}</p>
                <p className="text-sm text-gray-500">{tx.protocol}</p>
              </div>
              <p className="font-medium">{tx.value}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );

  const ActivityChart = ({ data }) => (
    <Card>
      <CardHeader>
        <CardTitle>Activity Overview</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="transactions" stroke="#2563eb" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4">Wallet Analysis Dashboard</h1>
          <div className="flex gap-4">
            <Input
              placeholder="Enter Ethereum wallet address (0x...)"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              className="flex-1"
            />
            <Button 
              onClick={handleAnalyze} 
              disabled={loading}
            >
              {loading ? (
                <span>Analyzing...</span>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />
                  Analyze
                </>
              )}
            </Button>
          </div>
          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        {analysisData && (
          <>
            <div className="grid grid-cols-3 gap-6 mb-6">
              <WalletProfile profile={analysisData.profile} />
              <TechnicalMetrics metrics={analysisData.technicalMetrics} />
            </div>
            
            <div className="grid grid-cols-1 gap-6">
              <ActivityChart data={analysisData.activityData} />
              <PortfolioAllocation portfolio={analysisData.portfolio} />
              <RecentTransactions transactions={analysisData.recentTransactions} />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default WalletAnalysisDashboard;