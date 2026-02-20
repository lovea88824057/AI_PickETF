# app.py - 优化版：2位小数 + 历史决策记录
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

# ============ 数据模块 ============
class ETFData:
    def __init__(self):
        self.etf_list = {
            '510300': '沪深300ETF',
            '510500': '中证500ETF', 
            '159915': '创业板ETF',
            '588000': '科创50ETF',
            '512880': '证券ETF',
            '515030': '新能源车ETF',
            '512480': '半导体ETF',
            '512690': '酒ETF',
            '512170': '医疗ETF',
            '518880': '黄金ETF'
        }
        self.cash_code = 'CASH'
        self.cash_name = '💰 空仓观望'
    
    def get_etf_data(self, symbol, start_date, end_date):
        """获取ETF历史数据"""
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq"
            )
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
            df = df[df['收盘'] > 0].dropna(subset=['收盘', '开盘'])
            return df
        except Exception as e:
            print(f"获取{symbol}失败: {e}")
            return None
    
    def calculate_features(self, df):
        """计算技术指标"""
        if len(df) < 30:
            return None
            
        df = df.copy()
        df['return_5'] = df['收盘'].pct_change(5)
        df['return_10'] = df['收盘'].pct_change(10)
        df['return_20'] = df['收盘'].pct_change(20)
        df['ma20'] = df['收盘'].rolling(20).mean()
        df['ma20_bias'] = (df['收盘'] - df['ma20']) / df['ma20']
        df['volatility'] = df['收盘'].pct_change().rolling(20).std()
        
        return df

# ============ AI模型 ============
class SmartModel:
    def __init__(self):
        self.weights = {
            'return_5': 0.25,
            'return_10': 0.20,
            'return_20': 0.25,
            'ma20_bias': 0.20,
            'volatility': -0.10
        }
        self.cash_threshold = 45
        self.market_bear_threshold = -0.05
        self.max_volatility = 0.03
        
    def predict(self, df, market_df=None):
        """预测ETF得分，返回2位小数"""
        if len(df) == 0:
            return 50.0, {}
            
        latest = df.iloc[-1]
        score = 50.0
        signals = {}
        
        for feature, weight in self.weights.items():
            if feature in latest and pd.notna(latest[feature]):
                if feature == 'volatility':
                    vol_score = max(0, 1 - latest[feature] / self.max_volatility) * 50
                    score += (vol_score - 25) * abs(weight)
                    signals['volatility'] = round(latest[feature] * 100, 2)
                else:
                    score += latest[feature] * weight * 100
                    signals[feature] = round(latest[feature] * 100, 2)
        
        # 确保2位小数
        score = round(min(max(score, 0), 100), 2)
        
        market_bear = False
        if market_df is not None and len(market_df) > 5:
            market_return_5 = market_df['收盘'].pct_change(5).iloc[-1]
            if market_return_5 < self.market_bear_threshold:
                market_bear = True
            signals['market_return_5'] = round(market_return_5 * 100, 2)
        
        signals['market_bear'] = market_bear
        signals['raw_score'] = score
        
        return score, signals
    
    def should_hold_cash(self, all_scores, market_df=None):
        """判断是否空仓"""
        if not all_scores:
            return True, "无有效数据"
        
        max_score = max(all_scores.values())
        avg_score = sum(all_scores.values()) / len(all_scores)
        
        if max_score < self.cash_threshold:
            return True, f"最高分{max_score:.2f}低于阈值{self.cash_threshold}"
        
        if market_df is not None and len(market_df) > 5:
            market_return_5 = market_df['收盘'].pct_change(5).iloc[-1]
            if market_return_5 < self.market_bear_threshold:
                return True, f"大盘5日跌{market_return_5*100:.2f}%"
        
        if avg_score < 40 and max_score - avg_score < 10:
            return True, f"市场低迷(均分{avg_score:.2f})"
        
        return False, "信号良好"

# ============ 回测引擎（记录历史决策） ============
class BacktestEngine:
    def __init__(self, strategy):
        self.strategy = strategy
        self.trade_log = []
        self.nav_history = []
        self.decision_history = []  # 新增：历史决策记录
        
    def run_backtest(self, start_date, end_date, initial_capital=100000):
        """运行回测，记录每日决策"""
        print(f"开始回测: {start_date.date()} 至 {end_date.date()}")
        
        data_start = start_date - timedelta(days=60)
        all_data = {}
        
        market_df = self.strategy.data.get_etf_data('510300', data_start, end_date)
        
        for symbol in self.strategy.data.etf_list.keys():
            df = self.strategy.data.get_etf_data(symbol, data_start, end_date)
            if df is not None and len(df) > 40:
                df = self.strategy.data.calculate_features(df)
                if df is not None:
                    all_data[symbol] = df
        
        if len(all_data) < 5:
            print("数据不足")
            return None
        
        common_dates = None
        for df in all_data.values():
            dates = set(df['日期'])
            common_dates = dates if common_dates is None else common_dates & dates
        
        trade_dates = sorted([d for d in common_dates if d >= start_date])
        print(f"交易日数量: {len(trade_dates)}")
        
        capital = initial_capital
        current_holding = None
        holding_shares = 0
        cash_position = initial_capital
        self.trade_log = []
        self.nav_history = []
        self.decision_history = []  # 清空历史决策
        
        commission_rate = 0.0001
        min_commission = 5
        cash_annual_return = 0.02
        daily_cash_return = cash_annual_return / 252
        
        for i in range(len(trade_dates)):
            today = trade_dates[i]
            
            today_data = {}
            for symbol, df in all_data.items():
                row = df[df['日期'] == today]
                if len(row) > 0:
                    today_data[symbol] = {
                        'open': row['开盘'].values[0],
                        'close': row['收盘'].values[0]
                    }
            
            if not today_data:
                continue
            
            # 记录今日决策
            daily_decision = {
                'date': today.strftime('%Y-%m-%d'),
                'prev_holding': current_holding or 'CASH',
                'scores': {},
                'decision': '',
                'reason': '',
                'action': 'HOLD'  # HOLD, BUY, SELL, SWITCH
            }
            
            if i > 0:
                yesterday = trade_dates[i-1]
                yesterday_data = {}
                market_yest = None
                
                for symbol, df in all_data.items():
                    yest_df = df[df['日期'] == yesterday]
                    if len(yest_df) > 0:
                        yesterday_data[symbol] = yest_df
                
                if market_df is not None:
                    market_yest = market_df[market_df['日期'] == yesterday]
                
                if yesterday_data:
                    all_scores = {}
                    all_signals = {}
                    
                    for symbol, df in yesterday_data.items():
                        if len(df) > 0:
                            score, signals = self.strategy.model.predict(df, market_yest)
                            all_scores[symbol] = score
                            all_signals[symbol] = signals
                    
                    # 记录所有分数（2位小数）
                    daily_decision['scores'] = {k: round(v, 2) for k, v in all_scores.items()}
                    
                    should_cash, cash_reason = self.strategy.model.should_hold_cash(all_scores, market_yest)
                    
                    if should_cash:
                        target = 'CASH'
                        target_score = 0.0
                        daily_decision['decision'] = 'CASH'
                        daily_decision['reason'] = cash_reason
                    else:
                        target = max(all_scores, key=all_scores.get)
                        target_score = all_scores[target]
                        daily_decision['decision'] = target
                        daily_decision['reason'] = f"得分最高: {target_score:.2f}分"
                    
                    # 确定操作类型
                    if current_holding != target:
                        if current_holding is None or current_holding == 'CASH':
                            daily_decision['action'] = 'BUY'
                        elif target == 'CASH':
                            daily_decision['action'] = 'SELL'
                        else:
                            daily_decision['action'] = 'SWITCH'
                    
                    # 执行交易
                    if current_holding != target:
                        if current_holding and current_holding != 'CASH' and current_holding in today_data:
                            sell_price = today_data[current_holding]['open']
                            sell_value = holding_shares * sell_price
                            commission = max(sell_value * commission_rate, min_commission)
                            cash_position = sell_value - commission
                            
                            self.trade_log.append({
                                'date': today.strftime('%Y-%m-%d'),
                                'action': 'SELL',
                                'symbol': current_holding,
                                'price': round(sell_price, 3),
                                'shares': round(holding_shares, 2),
                                'value': round(sell_value, 2)
                            })
                            
                            current_holding = None
                            holding_shares = 0
                        
                        if target != 'CASH':
                            if target in today_data:
                                buy_price = today_data[target]['open']
                                commission = max(cash_position * commission_rate, min_commission)
                                actual_cash = cash_position - commission
                                holding_shares = actual_cash / buy_price
                                cash_position = 0
                                
                                self.trade_log.append({
                                    'date': today.strftime('%Y-%m-%d'),
                                    'action': 'BUY',
                                    'symbol': target,
                                    'price': round(buy_price, 3),
                                    'shares': round(holding_shares, 2),
                                    'score': round(target_score, 2)
                                })
                                
                                current_holding = target
                        else:
                            self.trade_log.append({
                                'date': today.strftime('%Y-%m-%d'),
                                'action': 'CASH',
                                'reason': cash_reason,
                                'cash_value': round(cash_position, 2)
                            })
                            current_holding = 'CASH'
                    else:
                        daily_decision['action'] = 'HOLD'
                        daily_decision['reason'] += " (继续持有)"
            
            # 记录决策
            self.decision_history.append(daily_decision)
            
            # 计算净值
            if current_holding and current_holding != 'CASH' and current_holding in today_data:
                close_price = today_data[current_holding]['close']
                nav = holding_shares * close_price
            else:
                if current_holding == 'CASH':
                    cash_position = cash_position * (1 + daily_cash_return)
                nav = cash_position
            
            self.nav_history.append({
                'date': today.strftime('%Y-%m-%d'),
                'nav': round(nav, 2),
                'holding': current_holding or 'CASH',
                'return_pct': round((nav / initial_capital - 1) * 100, 2)
            })
        
        return self.calculate_metrics(initial_capital)
    
    def calculate_metrics(self, initial_capital):
        """计算回测指标"""
        if not self.nav_history or len(self.nav_history) < 2:
            return None
        
        nav_df = pd.DataFrame(self.nav_history)
        
        total_return = (nav_df['nav'].iloc[-1] / initial_capital - 1) * 100
        days = len(nav_df)
        annual_return = ((1 + total_return/100) ** (252 / days) - 1) * 100 if days > 0 else 0
        
        cummax = nav_df['nav'].cummax()
        drawdown = (nav_df['nav'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        returns = nav_df['nav'].pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() * 252 - 0.03) / (returns.std() * np.sqrt(252))
        else:
            sharpe = 0
        
        cash_days = len(nav_df[nav_df['holding'] == 'CASH'])
        cash_ratio = (cash_days / len(nav_df)) * 100
        
        buy_trades = [t for t in self.trade_log if t['action'] == 'BUY']
        
        print(f"回测完成: 收益{total_return:.2f}%, 空仓{cash_ratio:.1f}%, 决策记录{len(self.decision_history)}条")
        
        return {
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'trade_count': len(buy_trades),
            'cash_ratio': round(cash_ratio, 2),
            'start_date': nav_df['date'].iloc[0],
            'end_date': nav_df['date'].iloc[-1],
            'final_nav': round(nav_df['nav'].iloc[-1], 2),
            'initial_capital': initial_capital,
            'nav_history': self.nav_history,
            'trade_log': self.trade_log,
            'decision_history': self.decision_history  # 新增
        }
    
    def get_chart_data(self, period='month'):
        """获取图表数据"""
        if not self.nav_history:
            return []
        
        df = pd.DataFrame(self.nav_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        end_date = df['date'].max()
        period_days = {'week': 7, 'month': 30, 'half': 180, 'year': 365}
        
        if period in period_days:
            start_date = end_date - timedelta(days=period_days[period])
            filtered = df[df['date'] >= start_date].copy()
        else:
            filtered = df.copy()
        
        if len(filtered) > 60:
            step = len(filtered) // 60
            filtered = filtered.iloc[::step]
        
        return [{
            'date': row['date'].strftime('%m-%d') if period in ['week', 'month'] else row['date'].strftime('%Y-%m'),
            'value': row['nav'],
            'return_pct': row['return_pct'],
            'holding': row['holding'],
            'is_cash': row['holding'] == 'CASH'
        } for _, row in filtered.iterrows()]
    
    def get_decisions(self, limit=50):
        """获取最近决策记录"""
        if not self.decision_history:
            return []
        return self.decision_history[-limit:][::-1]  # 倒序，最新的在前

# ============ 策略实例 ============
class Strategy:
    def __init__(self):
        self.data = ETFData()
        self.model = SmartModel()
        self.backtest = BacktestEngine(self)
    
    def get_recommendation(self):
        """获取今日推荐"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        all_scores = {}
        all_details = {}
        
        market_df = self.data.get_etf_data('510300', start_date, end_date)
        
        for symbol in self.data.etf_list.keys():
            df = self.data.get_etf_data(symbol, start_date, end_date)
            if df is None or len(df) < 30:
                continue
            
            df = self.data.calculate_features(df)
            if df is None:
                continue
            
            score, signals = self.model.predict(df, market_df)
            
            latest = df.iloc[-1]
            prev_close = df.iloc[-2]['收盘'] if len(df) > 1 else latest['收盘']
            change_pct = (latest['收盘'] - prev_close) / prev_close * 100
            
            all_scores[symbol] = round(score, 2)  # 2位小数
            all_details[symbol] = {
                'name': self.data.etf_list.get(symbol, symbol),
                'score': round(score, 2),
                'price': round(latest['收盘'], 3),
                'change_pct': round(change_pct, 2),
                'signals': {k: round(v, 2) if isinstance(v, float) else v for k, v in signals.items()}
            }
        
        if not all_scores:
            return None
        
        should_cash, cash_reason = self.model.should_hold_cash(all_scores, market_df)
        
        if should_cash:
            recommendation = {
                'code': 'CASH',
                'name': self.data.cash_name,
                'score': 0.0,
                'reason': cash_reason
            }
        else:
            best_etf = max(all_scores, key=all_scores.get)
            recommendation = {
                'code': best_etf,
                'name': all_details[best_etf]['name'],
                'score': all_scores[best_etf],
                'price': all_details[best_etf]['price'],
                'change_pct': all_details[best_etf]['change_pct']
            }
        
        # 排名列表（全部2位小数）
        ranking = [{'code': 'CASH', 'name': self.data.cash_name, 'score': 0.0, 'is_cash': True}]
        for code, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            ranking.append({
                'code': code,
                'name': all_details[code]['name'],
                'score': score,
                'is_cash': False
            })
        
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'recommendation': recommendation['code'],
            'recommend_name': recommendation['name'],
            'confidence': recommendation.get('score', 0.0),
            'cash_reason': cash_reason if should_cash else None,
            'should_cash': should_cash,
            'all_scores': ranking,
            'details': all_details,
            'market_status': '熊市' if should_cash else '正常'
        }

strategy = Strategy()

# ============ 网页界面（新增决策记录页面） ============
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF AI选股器 - 决策记录版</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5; 
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #333; font-size: 28px; margin-bottom: 10px; }
        .header p { color: #666; font-size: 14px; }
        
        .nav-tabs {
            display: flex;
            background: white;
            border-radius: 12px;
            padding: 4px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .nav-tab {
            flex: 1;
            padding: 12px;
            border: none;
            background: transparent;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            color: #666;
            transition: all 0.3s;
            text-align: center;
        }
        .nav-tab.active {
            background: #667eea;
            color: white;
            font-weight: bold;
        }
        
        .page { display: none; }
        .page.active { display: block; }
        
        .warning {
            background: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        }
        
        .recommend-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .recommend-card.cash {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        
        .tag {
            background: rgba(255,255,255,0.2);
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            margin-bottom: 16px;
        }
        .tag.alert { background: rgba(255,0,0,0.3); font-weight: bold; }
        
        .etf-code { font-size: 48px; font-weight: bold; margin: 10px 0; }
        .etf-name { font-size: 20px; opacity: 0.9; margin-bottom: 20px; }
        .cash-reason { 
            background: rgba(255,255,255,0.15);
            padding: 12px;
            border-radius: 8px;
            font-size: 14px;
            margin-top: 10px;
        }
        
        .metrics { display: flex; gap: 20px; margin-top: 20px; }
        .metric { text-align: center; flex: 1; }
        .metric-value { font-size: 24px; font-weight: bold; display: block; }
        .metric-label { font-size: 12px; opacity: 0.8; margin-top: 4px; }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .chart-title { font-size: 18px; font-weight: bold; color: #333; }
        
        .period-tabs {
            display: flex;
            background: #f0f0f0;
            border-radius: 8px;
            padding: 4px;
            gap: 4px;
        }
        .period-tab {
            padding: 8px 16px;
            border: none;
            background: transparent;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            color: #666;
        }
        .period-tab.active {
            background: white;
            color: #667eea;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-top: 20px;
        }
        .stat-item {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
        }
        .stat-item.highlight { background: #e3f2fd; border: 2px solid #2196f3; }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            display: block;
        }
        .stat-value.positive { color: #f56c6c; }
        .stat-value.negative { color: #67c23a; }
        .stat-value.cash { color: #11998e; }
        .stat-label { font-size: 12px; color: #999; margin-top: 4px; }
        
        /* 决策记录样式 */
        .decision-list {
            max-height: 600px;
            overflow-y: auto;
        }
        .decision-item {
            border-left: 4px solid #667eea;
            background: #f8f9fa;
            padding: 16px;
            margin-bottom: 12px;
            border-radius: 0 12px 12px 0;
        }
        .decision-item.cash { border-left-color: #11998e; }
        .decision-item.switch { border-left-color: #ff9800; }
        
        .decision-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .decision-date { font-weight: bold; color: #333; font-size: 14px; }
        .decision-action {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .action-buy { background: #f56c6c; color: white; }
        .action-sell { background: #909399; color: white; }
        .action-switch { background: #ff9800; color: white; }
        .action-hold { background: #67c23a; color: white; }
        .action-cash { background: #11998e; color: white; }
        
        .decision-body {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .decision-main {
            flex: 1;
        }
        .decision-from-to {
            font-size: 16px;
            color: #333;
            margin-bottom: 4px;
        }
        .decision-arrow {
            color: #667eea;
            font-weight: bold;
            margin: 0 8px;
        }
        .decision-reason {
            font-size: 13px;
            color: #666;
        }
        .decision-scores {
            text-align: right;
            font-size: 12px;
            color: #999;
        }
        .score-detail {
            display: inline-block;
            margin-left: 8px;
        }
        
        .ranking-title { 
            font-size: 18px; 
            font-weight: bold; 
            margin-bottom: 16px;
            color: #333;
        }
        .rank-item {
            display: flex;
            align-items: center;
            padding: 14px 0;
            border-bottom: 1px solid #eee;
        }
        .rank-item:last-child { border-bottom: none; }
        .rank-item.cash {
            background: #f0f9f4;
            border-radius: 8px;
            margin: 4px 0;
            padding: 14px;
        }
        .rank-num {
            width: 32px;
            height: 32px;
            background: #f0f0f0;
            border-radius: 50%;
            text-align: center;
            line-height: 32px;
            font-weight: bold;
            color: #666;
            font-size: 14px;
        }
        .rank-num.cash { background: #11998e !important; color: white !important; }
        .rank-1 { background: #ffd700 !important; color: #333 !important; }
        .rank-2 { background: #c0c0c0 !important; color: #333 !important; }
        .rank-3 { background: #cd7f32 !important; color: white !important; }
        .rank-info { flex: 1; margin-left: 12px; }
        .rank-name { font-size: 16px; font-weight: 500; color: #333; display: block; }
        .rank-name.cash { color: #11998e; font-weight: bold; }
        .rank-code { font-size: 12px; color: #999; margin-top: 2px; }
        .rank-score { font-size: 20px; font-weight: bold; color: #667eea; font-family: 'Courier New', monospace; }
        .rank-score.cash { color: #11998e; }
        
        .legend {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
        .legend-dot.cash { background: #11998e; }
        .legend-dot.etf { background: #667eea; }
        
        .loading { text-align: center; padding: 60px; color: #999; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 ETF AI选股器</h1>
            <p>智能空仓版 | 历史决策可追溯</p>
        </div>
        
        <!-- 导航标签 -->
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchPage('dashboard')">📊 实时推荐</button>
            <button class="nav-tab" onclick="switchPage('backtest')">📈 回测收益</button>
            <button class="nav-tab" onclick="switchPage('decisions')">📋 决策记录</button>
        </div>
        
        <!-- 页面1: 实时推荐 -->
        <div id="page-dashboard" class="page active">
            <div class="warning">
                <strong>💡 策略特点：</strong>AI评分精确到2位小数，当所有ETF评分低于45.00分或市场大跌时自动空仓。
            </div>
            <div id="recommend-section"></div>
            
            <div class="card">
                <div class="ranking-title">📊 今日ETF评分排行（满分100.00）</div>
                <div id="ranking-list"></div>
            </div>
        </div>
        
        <!-- 页面2: 回测收益 -->
        <div id="page-backtest" class="page">
            <div class="card">
                <div class="chart-header">
                    <span class="chart-title">策略回测收益走势</span>
                    <div class="period-tabs">
                        <button class="period-tab" onclick="switchPeriod('week')">近1周</button>
                        <button class="period-tab active" onclick="switchPeriod('month')">近1月</button>
                        <button class="period-tab" onclick="switchPeriod('half')">近半年</button>
                        <button class="period-tab" onclick="switchPeriod('year')">近1年</button>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="returnChart"></canvas>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-dot cash"></div>
                        <span>绿色点 = 空仓期</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot etf"></div>
                        <span>蓝色线 = 持仓期</span>
                    </div>
                </div>
                <div class="stats-grid" id="stats-grid"></div>
            </div>
        </div>
        
        <!-- 页面3: 决策记录 -->
        <div id="page-decisions" class="page">
            <div class="card">
                <div class="chart-header">
                    <span class="chart-title">📋 历史决策记录（最近50条）</span>
                    <span style="font-size: 12px; color: #999;">显示评分细节和决策理由</span>
                </div>
                <div class="decision-list" id="decision-list">
                    <div class="loading">加载中...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentPeriod = 'month';
        let returnChart = null;
        let backtestData = null;
        
        // 切换页面
        function switchPage(page) {
            document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(`page-${page}`).classList.add('active');
            
            if (page === 'backtest' && !backtestData) {
                loadBacktest('month');
            } else if (page === 'decisions') {
                loadDecisions();
            }
        }
        
        async function loadRecommendation() {
            try {
                const res = await fetch('/api/recommend');
                const data = await res.json();
                if (!data) return;
                
                const rec = data.recommendation;
                const isCash = data.should_cash;
                
                let html = `
                    <div class="card recommend-card ${isCash ? 'cash' : ''}">
                        <span class="tag ${isCash ? 'alert' : ''}">
                            ${isCash ? '⚠️ 建议空仓' : '🏆 今日推荐买入'}
                        </span>
                        <div class="etf-code">${rec}</div>
                        <div class="etf-name">${data.recommend_name}</div>
                `;
                
                if (isCash) {
                    html += `
                        <div class="cash-reason">
                            <strong>空仓原因：</strong>${data.cash_reason}<br>
                            <small>空仓期间享受货币基金收益(约2%年化)</small>
                        </div>
                    `;
                } else {
                    const detail = data.details[rec];
                    html += `
                        <div class="metrics">
                            <div class="metric">
                                <span class="metric-value">¥${detail.price}</span>
                                <span class="metric-label">当前价格</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">${data.confidence.toFixed(2)}</span>
                                <span class="metric-label">AI评分</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">${detail.change_pct}%</span>
                                <span class="metric-label">今日涨跌</span>
                            </div>
                        </div>
                    `;
                }
                
                html += '</div>';
                document.getElementById('recommend-section').innerHTML = html;
                
                // 排名列表（2位小数）
                let rankHtml = '';
                data.all_scores.forEach((item, idx) => {
                    const isCashItem = item.is_cash;
                    const rankClass = isCashItem ? 'cash' : (idx <= 3 ? `rank-${idx}` : '');
                    const scoreClass = isCashItem ? 'cash' : '';
                    const nameClass = isCashItem ? 'cash' : '';
                    
                    rankHtml += `
                        <div class="rank-item ${isCashItem ? 'cash' : ''}">
                            <div class="rank-num ${rankClass} ${isCashItem ? 'cash' : ''}">
                                ${isCashItem ? '💰' : idx}
                            </div>
                            <div class="rank-info">
                                <span class="rank-name ${nameClass}">${item.name}</span>
                                ${!isCashItem ? `<span class="rank-code">${item.code}</span>` : ''}
                            </div>
                            <div class="rank-score ${scoreClass}">
                                ${isCashItem ? '避险' : item.score.toFixed(2)}
                            </div>
                        </div>
                    `;
                });
                document.getElementById('ranking-list').innerHTML = rankHtml;
                
            } catch (e) {
                console.error('加载推荐失败:', e);
            }
        }
        
        async function loadBacktest(period = 'month') {
            try {
                const res = await fetch(`/api/backtest?period=${period}&days=365`);
                const data = await res.json();
                
                if (!data || data.error) {
                    document.getElementById('stats-grid').innerHTML = 
                        `<div class="loading">${data?.error || '回测失败'}</div>`;
                    return;
                }
                
                backtestData = data;
                const metrics = data.metrics;
                
                const totalClass = metrics.total_return >= 0 ? 'positive' : 'negative';
                const annualClass = metrics.annual_return >= 0 ? 'positive' : 'negative';
                
                document.getElementById('stats-grid').innerHTML = `
                    <div class="stat-item">
                        <span class="stat-value ${totalClass}">${metrics.total_return >= 0 ? '+' : ''}${metrics.total_return}%</span>
                        <span class="stat-label">累计收益</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value ${annualClass}">${metrics.annual_return >= 0 ? '+' : ''}${metrics.annual_return}%</span>
                        <span class="stat-label">年化收益</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value negative">${metrics.max_drawdown}%</span>
                        <span class="stat-label">最大回撤</span>
                    </div>
                    <div class="stat-item highlight">
                        <span class="stat-value cash">${metrics.cash_ratio}%</span>
                        <span class="stat-label">空仓时间占比</span>
                    </div>
                `;
                
                drawChart(data.chart_data, metrics.total_return >= 0);
                
            } catch (e) {
                console.error('加载回测失败:', e);
            }
        }
        
        async function loadDecisions() {
            try {
                if (!backtestData) {
                    // 先加载回测数据
                    const res = await fetch(`/api/backtest?period=year&days=365`);
                    backtestData = await res.json();
                }
                
                const decisions = backtestData.metrics.decision_history;
                const listEl = document.getElementById('decision-list');
                
                if (!decisions || decisions.length === 0) {
                    listEl.innerHTML = '<div class="loading">暂无决策记录</div>';
                    return;
                }
                
                let html = '';
                decisions.forEach((d, idx) => {
                    const actionClass = {
                        'BUY': 'action-buy',
                        'SELL': 'action-sell',
                        'SWITCH': 'action-switch',
                        'HOLD': 'action-hold',
                        'CASH': 'action-cash'
                    }[d.action] || 'action-hold';
                    
                    const actionText = {
                        'BUY': '买入',
                        'SELL': '卖出',
                        'SWITCH': '换仓',
                        'HOLD': '持有',
                        'CASH': '空仓'
                    }[d.action] || d.action;
                    
                    const itemClass = d.decision === 'CASH' ? 'cash' : (d.action === 'SWITCH' ? 'switch' : '');
                    
                    // 构建分数详情字符串
                    let scoresHtml = '';
                    if (d.scores && Object.keys(d.scores).length > 0) {
                        const sortedScores = Object.entries(d.scores)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3);
                        scoresHtml = sortedScores.map(([code, score]) => 
                            `<span class="score-detail">${code}:${score.toFixed(2)}</span>`
                        ).join('');
                    }
                    
                    html += `
                        <div class="decision-item ${itemClass}">
                            <div class="decision-header">
                                <span class="decision-date">${d.date}</span>
                                <span class="decision-action ${actionClass}">${actionText}</span>
                            </div>
                            <div class="decision-body">
                                <div class="decision-main">
                                    <div class="decision-from-to">
                                        ${d.prev_holding === 'CASH' ? '💰 空仓' : d.prev_holding}
                                        <span class="decision-arrow">→</span>
                                        ${d.decision === 'CASH' ? '💰 空仓' : d.decision}
                                    </div>
                                    <div class="decision-reason">${d.reason}</div>
                                </div>
                                <div class="decision-scores">
                                    ${scoresHtml}
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                listEl.innerHTML = html;
                
            } catch (e) {
                console.error('加载决策记录失败:', e);
                document.getElementById('decision-list').innerHTML = 
                    '<div class="loading">加载失败</div>';
            }
        }
        
        function drawChart(chartData, isPositive) {
            const ctx = document.getElementById('returnChart').getContext('2d');
            
            if (returnChart) {
                returnChart.destroy();
            }
            
            const labels = chartData.map(d => d.date);
            const values = chartData.map(d => d.value);
            const cashPoints = chartData.map((d, i) => d.is_cash ? values[i] : null);
            
            const gradient = ctx.createLinearGradient(0, 0, 0, 300);
            if (isPositive) {
                gradient.addColorStop(0, 'rgba(245, 108, 108, 0.3)');
                gradient.addColorStop(1, 'rgba(245, 108, 108, 0)');
            } else {
                gradient.addColorStop(0, 'rgba(103, 194, 58, 0.3)');
                gradient.addColorStop(1, 'rgba(103, 194, 58, 0)');
            }
            
            returnChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '策略净值',
                        data: values,
                        borderColor: isPositive ? '#f56c6c' : '#67c23a',
                        backgroundColor: gradient,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }, {
                        label: '空仓期',
                        data: cashPoints,
                        backgroundColor: '#11998e',
                        borderColor: '#11998e',
                        pointRadius: 4,
                        pointStyle: 'circle',
                        showLine: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const idx = context.dataIndex;
                                    const item = chartData[idx];
                                    if (context.datasetIndex === 1 && item.is_cash) {
                                        return ['状态: 空仓避险', `净值: ¥${item.value.toFixed(2)}`];
                                    }
                                    return [
                                        `净值: ¥${context.parsed.y.toFixed(2)}`,
                                        `收益率: ${item.return_pct >= 0 ? '+' : ''}${item.return_pct}%`,
                                        `持仓: ${item.holding === 'CASH' ? '空仓' : item.holding}`
                                    ];
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { maxTicksLimit: 6, font: { size: 10 } }
                        },
                        y: {
                            grid: { color: '#f0f0f0' },
                            ticks: {
                                callback: function(value) {
                                    return '¥' + value.toFixed(0);
                                },
                                font: { size: 10 }
                            }
                        }
                    }
                }
            });
        }
        
        function switchPeriod(period) {
            currentPeriod = period;
            document.querySelectorAll('.period-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            event.target.classList.add('active');
            loadBacktest(period);
        }
        
        // 初始化
        loadRecommendation();
    </script>
</body>
</html>
"""

# ============ API路由 ============

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/recommend', methods=['GET'])
def recommend():
    result = strategy.get_recommendation()
    return jsonify(result)

@app.route('/api/backtest', methods=['GET'])
def backtest():
    period = request.args.get('period', 'month')
    days = int(request.args.get('days', 365))
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    metrics = strategy.backtest.run_backtest(start_date, end_date)
    
    if not metrics:
        return jsonify({"error": "回测失败"})
    
    chart_data = strategy.backtest.get_chart_data(period)
    
    return jsonify({
        "metrics": metrics,
        "chart_data": chart_data,
        "period": period
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)