import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Activity, Wallet, Flame, TrendingUp, Clock, DollarSign, ArrowLeftRight} from 'lucide-react';

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
      const response = await fetch(`${import.meta.env.VITE_API_URL}/analyze/${address}`,
        {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          },
          mode: 'cors', // Explicitly set CORS mode
        }
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch wallet data');
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
          portfolio_metrics: data.portfolio
        },
        portfolio: data.portfolio,
        technicalMetrics: {
          avgGasUsed: data.technical_metrics.avg_gas_used,
          avg_gas_paid_usd: data.technical_metrics.avg_gas_paid_usd,
          totalTransactions: data.technical_metrics.total_transactions,
          txFrequency: data.technical_metrics.transaction_frequency
        },
        recentTransactions: data.recent_transactions || [],
        activityData: data.activity_history || [],
        executiveSummary: data.executive_summary
      });
    } catch (err) {
      setError('Failed to fetch wallet analysis: ' + err.message);
    } finally {
      setLoading(false);
    }
};

const WalletProfile = ({ profile, metrics }) => {
  // Function to determine main holding from portfolio metrics
  const getMainHolding = () => {
    const tokens = profile.portfolio_metrics?.tokens;
    if (!tokens || Object.keys(tokens).length === 0) {
        return "No holdings";
    }

    // Check if ETH exists in tokens
    if ('ETH' in tokens) {
        return `${tokens['ETH'].amount.toFixed(4)} ETH`;
    }

    // If no ETH, fallback to previous logic
    const mainToken = Object.entries(tokens)
        .reduce((max, [token, data]) => {
            return data.amount > max[1].amount ? [token, data] : max;
        }, ['', { amount: 0 }]);

    return `${mainToken[1].amount.toFixed(4)} ${mainToken[0]}`;
};

    return (
      <Card className="col-span-full lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wallet className="h-6 w-6" />
            Wallet Profile
          </CardTitle>
          <CardDescription>Overview and key metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
            <p className="text-sm font-medium">Balance</p>
            <p className="text-2xl font-bold">
                {getMainHolding()}
            </p>
            </div>
            <div>
              <p className="text-sm font-medium">Transactions</p>
              <p className="text-2xl font-bold flex items-center gap-2">
                  <ArrowLeftRight className="h-5 w-5" />
                  {metrics.totalTransactions >= 9000 
                      ? ">9k"
                      : metrics.totalTransactions.toLocaleString()
                  }
              </p>
            </div>
            <div>
              <p className="text-sm font-medium">Average Gas Paid</p>
              <p className="text-xl font-bold flex items-center gap-2">
                  <Flame className="h-5 w-5" />
                  ${metrics.avg_gas_paid_usd.toFixed(2)}
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
                {Number(profile.totalValue).toFixed(2)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const Portfolio = ({ portfolio }) => {
    // Early return if no portfolio data
    if (!portfolio?.tokens || Object.keys(portfolio.tokens).length === 0) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Wallet className="h-6 w-6" />
                        Token Portfolio
                    </CardTitle>
                    <CardDescription>No tokens found in this wallet</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    // Helper function to get token price
    const getTokenPrice = (token) => {
        // If it's WETH, use ETH price
        if (token === 'WETH' && portfolio.prices['ETH']) {
            return portfolio.prices['ETH'];
        }
        return portfolio.prices[token] || 0;
    };

    // Transform the data
    const pieData = Object.entries(portfolio.tokens)
        .map(([token, tokenData]) => {
            const price = getTokenPrice(token);
            const valueUSD = tokenData.amount * price;
            return {
                name: token,
                amount: tokenData.amount,
                price: price,
                value: valueUSD,
                displayValue: valueUSD
            };
        })
        // Filter out dust amounts (worth less than $0.01)
        .filter(token => token.value > 0.01)
        .sort((a, b) => b.value - a.value);

    // Calculate total USD value
    const totalUSD = pieData.reduce((sum, item) => sum + item.value, 0);

    // Colors for different tokens
    const COLORS = ['#627EEA', '#2775CA', '#26A17B', '#F5AC37', '#EC4899'];

    return (
        <Card className="col-span-full lg:col-span-1">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Wallet className="h-6 w-6" />
                    Token Portfolio
                </CardTitle>
                <CardDescription className="flex flex-col sm:flex-row justify-between">
                    <span className="font-medium">
                        Total: ${totalUSD.toLocaleString(undefined, {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2
                        })}
                    </span>
                </CardDescription>
            </CardHeader>
            <CardContent>
                {pieData.length > 0 ? (
                    <>
                        <div className="h-[250px] sm:h-[300px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieData}
                                        dataKey="value"
                                        nameKey="name"
                                        cx="50%"
                                        cy="50%"
                                        outerRadius={80}
                                        fill="#8884d8"
                                        labelLine={false}
                                        label={({ name, percent }) =>
                                            `${name} ${(percent * 100).toFixed(1)}%`
                                        }
                                    >
                                        {pieData.map((entry, index) => (
                                            <Cell
                                                key={`cell-${index}`}
                                                fill={COLORS[index % COLORS.length]}
                                            />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        formatter={(value, name, props) => [
                                            `$${value.toLocaleString(undefined, {
                                                minimumFractionDigits: 2,
                                                maximumFractionDigits: 2
                                            })}`,
                                            name
                                        ]}
                                    />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Token list below chart */}
                        <div className="mt-6 space-y-4">
                            {pieData.map((token, index) => (
                                <div key={token.name} className="flex justify-between items-center">
                                    <span className="flex items-center gap-2">
                                        <div
                                            className="w-3 h-3 rounded-full"
                                            style={{ backgroundColor: COLORS[index % COLORS.length] }}
                                        />
                                        <span className="font-medium">{token.name}</span>
                                    </span>
                                    <div className="text-right">
                                        <div className="font-medium">
                                            ${token.value.toLocaleString(undefined, {
                                                minimumFractionDigits: 2,
                                                maximumFractionDigits: 2
                                            })}
                                        </div>
                                        <div className="text-sm text-gray-500">
                                            {token.amount.toLocaleString(undefined, {
                                                maximimumSignificantDigits: 6
                                            })} × ${token.price.toLocaleString(undefined, {
                                                minimumFractionDigits: 2,
                                                maximumFractionDigits: 2
                                            })}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </>
                ) : (
                    <div className="text-center text-gray-500 py-8">
                        No significant token holdings found
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

  // const RecentTransactions = ({ transactions }) => (
  //   <Card>
  //     <CardHeader>
  //       <CardTitle>Recent Transactions</CardTitle>
  //     </CardHeader>
  //     <CardContent>
  //       <div className="space-y-4">
  //         {transactions.map((tx, index) => (
  //           <div key={index} className="flex justify-between items-center border-b pb-2">
  //             <div>
  //               <p className="font-medium">{tx.type}</p>
  //               <p className="text-sm text-gray-500">{tx.protocol}</p>
  //             </div>
  //             <p className="font-medium">{tx.value}</p>
  //           </div>
  //         ))}
  //       </div>
  //     </CardContent>
  //   </Card>
  // );

  const ActivityChart = ({ data }) => (
    <Card>
        <CardHeader>
            <CardTitle>Activity Overview</CardTitle>
        </CardHeader>
        <CardContent>
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <XAxis dataKey="day" />
                        <YAxis />
                        <Tooltip />
                        <Line 
                            type="monotone" 
                            dataKey="count"
                            stroke="#2563eb" 
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen w-full bg-gray-200">
      <div className="container mx-auto p-4 sm:p-8">
        <div className="mb-6 sm:mb-8">
          <h1 className="text-2xl sm:text-3xl font-bold mb-4">Scan Base Wallet</h1>
          <div className="flex flex-col sm:flex-row gap-4">
            <Input
              placeholder="Enter Ethereum address (0x...)"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              className="flex-1"
            />
            <Button 
              onClick={handleAnalyze} 
              disabled={loading}
              className="w-full sm:w-auto"
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
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6 mb-6">
              <WalletProfile profile={analysisData.profile} metrics={analysisData.technicalMetrics}/>
              <Portfolio portfolio={analysisData.portfolio} />
            </div>
  
            <div className="grid grid-cols-1 gap-4 sm:gap-6 mb-6">
              <ActivityChart data={analysisData.activityData} />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default WalletAnalysisDashboard;