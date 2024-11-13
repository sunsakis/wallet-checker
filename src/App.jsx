import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

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
      // In production, replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate API call
      
      // Mock data for demonstration
      setAnalysisData({
        profile: {
          status: 'Active Trader',
          riskLevel: 'Low',
          activityLevel: 'Medium',
          mainActivity: 'DeFi Trading',
          lastActive: '2 hours ago',
          totalValue: '$25,000'
        },
        portfolio: {
          eth: 60,
          usdc: 40
        },
        recentTransactions: [
          { type: 'Added Liquidity', protocol: 'Uniswap', value: '$5,000' },
          { type: 'Borrowed', protocol: 'Aave', value: '$3,000' },
          { type: 'Swap', protocol: 'Uniswap', value: '$2,000' }
        ],
        activityData: [
          { month: 'Jan', transactions: 45 },
          { month: 'Feb', transactions: 52 },
          { month: 'Mar', transactions: 38 },
          { month: 'Apr', transactions: 41 },
          { month: 'May', transactions: 35 }
        ]
      });
    } catch (err) {
      setError('Failed to fetch wallet analysis. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const WalletProfile = ({ profile }) => (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle>📊 Wallet Profile</CardTitle>
        <CardDescription>Overview and key metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm font-medium">Status</p>
            <p className="text-2xl font-bold">{profile.status}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Risk Level</p>
            <p className="text-2xl font-bold">{profile.riskLevel}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Activity Level</p>
            <p className="text-2xl font-bold">{profile.activityLevel}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Total Value</p>
            <p className="text-2xl font-bold">{profile.totalValue}</p>
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
          <div>
            <div className="flex justify-between mb-1">
              <span>ETH</span>
              <span>{portfolio.eth}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full"
                style={{ width: `${portfolio.eth}%` }}
              ></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span>USDC</span>
              <span>{portfolio.usdc}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full"
                style={{ width: `${portfolio.usdc}%` }}
              ></div>
            </div>
          </div>
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
          <div className="grid grid-cols-3 gap-6">
            <WalletProfile profile={analysisData.profile} />
            <PortfolioAllocation portfolio={analysisData.portfolio} />
            <RecentTransactions transactions={analysisData.recentTransactions} />
            <ActivityChart data={analysisData.activityData} />
          </div>
        )}
      </div>
    </div>
  );
};

export default WalletAnalysisDashboard;