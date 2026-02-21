# app.py - 多策略版：扩展框架支持多种投资风格
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

# ============ 策略定义 ============
STRATEGIES = {
    'momentum': {
        'name': '追涨杀跌',
        'english': 'Momentum Trading',
        'description': '追踪市场热点，快速响应场景变化',
        'profession': '隔壁老翁',
        'detail': '喜欢追涨杀跌，追踪热点赛道，快速切换持仓，高风险高收益，市场情绪主导交易决策。',
        'style': '激进型',
        'color': '#667eea',
        'icon': '⚡'
    },
    'value': {
        'name': '稳健派跌',
        'english': 'Conservative Dividend Strategy',
        'description': '坚守20日均线，专注高股息白马股，宏观避险第一',
        'profession': '白马猎手',
        'detail': '专注银行、电力等高分红白马股，以20日均线为防线，破线即卖，规避宏观政策风险，追求稳定收益。',
        'style': '稳健型',
        'color': '#11998e',
        'icon': '🏛️'
    },
    'balanced': {
        'name': '量化均衡',
        'english': 'Balanced Strategy',
        'description': '风险与收益平衡配置，追求稳定增长',
        'profession': 'Quant工程师',
        'detail': '用代码优化交易逻辑，用数据说话，追求量化回测表现。通过技术指标和统计模型精确控制风险，打造稳定的投资系统。',
        'style': '量化型',
        'color': '#f59e0b',
        'icon': '⚖️'
    },
    'growth': {
        'name': '信仰成长',
        'english': 'Growth Investing',
        'description': '投资高增长企业，布局未来赛道',
        'profession': '赛道探险家',
        'detail': '甄别优质成长赛道，布局产业升级方向，追求长期产业浪潮。',
        'style': '成长型',
        'color': '#ec4899',
        'icon': '🚀'
    }
}

# ============ 数据模块 ============
class ETFData:
    def __init__(self):
        self.etf_list = {
            # 宽基ETF（最核心）
            '510300': '沪深300ETF',      # 沪深300 - 大盘必需
            '510500': '中证500ETF',      # 中证500 - 中盘必需
            # 行业主题ETF（代表性）
            '159915': '创业板ETF',       # 创业板 - 成长代表
            '588000': '科创50ETF',       # 科创50 - 科技代表
            # 高股息白马ETF（稳健派重点）
            '512880': '证券ETF',         # 金融安全
            '512800': '银行ETF',         # 银行安全
            '512630': '电力ETF',         # 电力安全
            '512200': '消费50ETF',       # 消费龙头
            # 其他行业ETF（多元化）
            '515030': '新能源车ETF',     # 成长新兴
            '512480': '半导体ETF',       # 科技细分
            '512690': '酒ETF',           # 消费细分
            '512170': '医疗ETF',         # 防御细分
            '512810': '食品ETF',         # 消费防御
            # 防御类ETF（保护性）
            '518880': '黄金ETF'          # 避险资产
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
            
            # 检查返回的数据是否为空
            if df is None or len(df) == 0:
                print(f"获取{symbol}失败: 返回空数据")
                return None
            
            # 检查必需列是否存在
            required_cols = ['日期', '收盘', '开盘']
            for col in required_cols:
                if col not in df.columns:
                    print(f"获取{symbol}失败: 缺少列'{col}'，返回列为{list(df.columns)}")
                    return None
            
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
            df = df[df['收盘'] > 0].dropna(subset=['收盘', '开盘'])
            
            if len(df) < 30:
                print(f"获取{symbol}成功但数据过少: {len(df)}条")
                return None
                
            return df
        except KeyError as e:
            print(f"获取{symbol}失败: 字段错误 {e}")
            return None
        except Exception as e:
            print(f"获取{symbol}失败: {type(e).__name__} - {e}")
            return None
    
    def calculate_features(self, df):
        """计算技术指标（支持多种量化指标）"""
        if len(df) < 30:
            return None
            
        df = df.copy()
        
        # ===== 基础收益率指标 =====
        df['return_5'] = df['收盘'].pct_change(5)
        df['return_10'] = df['收盘'].pct_change(10)
        df['return_20'] = df['收盘'].pct_change(20)
        
        # ===== 移动平均线系统 =====
        df['ma5'] = df['收盘'].rolling(5).mean()
        df['ma20'] = df['收盘'].rolling(20).mean()
        df['ma60'] = df['收盘'].rolling(60).mean()
        df['ma20_bias'] = (df['收盘'] - df['ma20']) / df['ma20']
        
        # ===== 波动率指标 =====
        df['returns'] = df['收盘'].pct_change()
        df['volatility'] = df['收盘'].pct_change().rolling(20).std()
        
        # ===== RSI指标（相对强弱指标）=====
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        # ===== MACD指标（动量指标）=====
        exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
        exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ===== 布林带指标（波动率带状）=====
        df['bb_std'] = df['收盘'].rolling(20).std()
        df['bb_middle'] = df['收盘'].rolling(20).mean()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['收盘'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ===== 趋势确认信号 =====
        df['trend_signal'] = ((df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])).astype(int)
        
        return df

# ============ AI模型（支持多策略框架） ============
class SmartModel:
    def __init__(self, strategy_type='momentum'):
        self.strategy_type = strategy_type
        
        # 追涨杀跌策略（Momentum Trading）
        if strategy_type == 'momentum':
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
        
        # 稳健派跌策略（Conservative Dividend Strategy）- 坚守20日均线
        elif strategy_type == 'value':
            self.weights = {
                'return_5': 0.05,        # 极低：不追逐短期波动
                'return_10': 0.10,       # 低：中期缓慢上升
                'return_20': 0.15,       # 低：关注长期趋势
                'ma20_bias': 0.60,       # 极高权重：20日均线是核心防线
                'volatility': -0.10      # 轻度惩罚波动率
            }
            self.cash_threshold = 55   # 更严格的空仓线（保守避险）
            self.market_bear_threshold = -0.06  # 宏观政策风险敏感（-6%触发）
            self.max_volatility = 0.020  # 严控波动率（相对严格）
        
        # 稳健均衡策略（Balanced Quant Strategy）- 量化多指标融合
        elif strategy_type == 'balanced':
            # 权重配置：多指标加权组合
            self.weights = {
                'trend_score': 0.30,      # 趋势确认权重（30%）
                'rsi_score': 0.20,        # RSI相对强弱权重（20%）
                'macd_score': 0.25,       # MACD动量权重（25%）
                'bb_score': 0.15,         # 布林带位置权重（15%）
                'volatility': -0.10       # 波动率风险惩罚（-10%）
            }
            # 量化参数
            self.cash_threshold = 42      # 空仓决策线（更宽松，增加机会）
            self.market_bear_threshold = -0.05  # 宏观风险阈值
            self.max_volatility = 0.035   # 最大容忍波动率
            self.dynamic_position_sizing = True  # 启用动态仓位管理
        
        # 成长信仰策略（Growth）- 预留框架，待实现
        elif strategy_type == 'growth':
            self.weights = {
                'return_5': 0.30,
                'return_10': 0.25,
                'return_20': 0.20,
                'ma20_bias': 0.15,
                'volatility': -0.05
            }
            self.cash_threshold = 40
            self.market_bear_threshold = -0.03
            self.max_volatility = 0.035
        
    def predict(self, df, market_df=None):
        """预测ETF得分，返回2位小数"""
        if len(df) == 0:
            return 50.0, {}
            
        latest = df.iloc[-1]
        score = 50.0
        signals = {}
        
        # ===== 对于稳健派跌策略：破20日均线卖出硬性规则 =====
        if self.strategy_type == 'value' and 'ma20_bias' in latest:
            ma20_bias = latest.get('ma20_bias', 0)
            signals['ma20_below_line'] = int(ma20_bias < 0)  # 是否跌破20日均线（0或1）
            
            # 如果跌破20日均线，直接降低评分到警戒线
            if ma20_bias < 0:
                # 记录跌破程度
                signals['break_distance'] = round(ma20_bias * 100, 2)
                # 根据跌破程度进行惩罚：跌破幅度越大，惩罚越重
                break_depth = abs(ma20_bias)
                if break_depth > 0.05:  # 跌破超过5%
                    score = 30.0  # 降到卖出信号
                elif break_depth > 0.02:  # 跌破超过2%
                    score = 40.0  # 降到警戒线
                else:  # 刚刚跌破
                    score = 45.0  # 降到中位
        
        # ===== 对于量化均衡策略：多指标融合信号 =====
        elif self.strategy_type == 'balanced':
            # 信号1：趋势确认（权重30%）
            # 判断价格是否在均线上方：ma5 > ma20 > ma60
            if len(df) >= 60:
                trend_signal = 50
                if latest['ma5'] > latest['ma20'] > latest['ma60']:
                    trend_signal = 85  # 完全看涨
                elif latest['ma5'] < latest['ma20'] < latest['ma60']:
                    trend_signal = 15  # 完全看跌
                elif latest['ma5'] > latest['ma20']:
                    trend_signal = 70  # 中期看涨
                elif latest['ma5'] < latest['ma20']:
                    trend_signal = 30  # 中期看跌
                signals['trend_score'] = round(trend_signal, 2)
                score += (trend_signal - 50) * 0.30
            
            # 信号2：RSI超卖/超买（权重20%）
            # RSI < 30 超卖（买入）, RSI > 70 超买（卖出）
            if 'rsi14' in latest and pd.notna(latest['rsi14']):
                rsi = latest['rsi14']
                rsi_signal = 50
                if rsi < 30:
                    rsi_signal = 80  # 强烈买入信号
                elif rsi < 40:
                    rsi_signal = 65
                elif rsi > 70:
                    rsi_signal = 20  # 强烈卖出信号
                elif rsi > 60:
                    rsi_signal = 35
                elif rsi > 50:
                    rsi_signal = 60
                else:
                    rsi_signal = 40
                signals['rsi_score'] = round(rsi_signal, 2)
                signals['rsi'] = round(rsi, 2)
                score += (rsi_signal - 50) * 0.20
            
            # 信号3：MACD动量（权重25%）
            # MACD金叉为买入信号，死叉为卖出信号
            if 'macd' in latest and 'macd_signal' in latest and len(df) > 1:
                macd_signal = 50
                prev_macd = df.iloc[-2]['macd']
                prev_signal = df.iloc[-2]['macd_signal']
                
                # 检测金叉死叉
                if prev_macd < prev_signal and latest['macd'] > latest['macd_signal']:
                    macd_signal = 80  # 金叉（强买）
                elif prev_macd > prev_signal and latest['macd'] < latest['macd_signal']:
                    macd_signal = 20  # 死叉（强卖）
                elif latest['macd'] > latest['macd_signal'] and latest['macd'] > 0:
                    macd_signal = 70  # 上升趋势
                elif latest['macd'] < latest['macd_signal'] and latest['macd'] < 0:
                    macd_signal = 30  # 下降趋势
                
                signals['macd_score'] = round(macd_signal, 2)
                signals['macd'] = round(latest['macd'], 6)
                score += (macd_signal - 50) * 0.25
            
            # 信号4：布林带位置（权重15%）
            # bb_position: 0=下轨(超卖), 0.5=中线, 1=上轨(超买)
            if 'bb_position' in latest and pd.notna(latest['bb_position']):
                bb_signal = 50
                bb_pos = latest['bb_position']
                if bb_pos < 0.2:
                    bb_signal = 75  # 接近下轨，买入机会
                elif bb_pos < 0.4:
                    bb_signal = 65
                elif bb_pos > 0.8:
                    bb_signal = 25  # 接近上轨，卖出风险
                elif bb_pos > 0.6:
                    bb_signal = 35
                else:
                    bb_signal = 50  # 中线附近，中性
                signals['bb_score'] = round(bb_signal, 2)
                signals['bb_position'] = round(bb_pos, 2)
                score += (bb_signal - 50) * 0.15
            
            # 波动率惩罚（权重-10%）
            if 'volatility' in latest and pd.notna(latest['volatility']):
                volatility = latest['volatility']
                vol_penalty = 0
                if volatility > self.max_volatility * 1.5:
                    vol_penalty = -20  # 高波动率严厉惩罚
                elif volatility > self.max_volatility:
                    vol_penalty = -10  # 中等惩罚
                else:
                    vol_penalty = 0  # 无惩罚
                signals['volatility'] = round(volatility * 100, 2)
                score += vol_penalty
        
        # ===== 正常的加权评分（用于momentum、value等策略）=====
        elif score == 50.0:  # 只有在没有触发止损时才进行正常评分
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
        
        # 市场熊市检测
        market_bear = 0
        if market_df is not None and len(market_df) > 5:
            market_return_5 = market_df['收盘'].pct_change(5).iloc[-1]
            if market_return_5 < self.market_bear_threshold:
                market_bear = 1
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
    def __init__(self, strategy_type='momentum'):
        self.strategy_type = strategy_type
        self.data = ETFData()
        self.model = SmartModel(strategy_type=strategy_type)
        self.backtest = BacktestEngine(self)
    
    def get_recommendation(self):
        """获取今日推荐"""
        # 检查策略是否已完整实现
        unimplemented_strategies = {
            'growth': '🚀 信仰成长策略 - 开发中'
        }
        
        if self.strategy_type in unimplemented_strategies:
            return {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'status': 'unimplemented',
                'message': unimplemented_strategies[self.strategy_type],
                'recommendation': 'N/A',
                'recommend_name': unimplemented_strategies[self.strategy_type],
                'confidence': 0.0,
                'cash_reason': '该策略正在开发中，敬请期待',
                'should_cash': 1,
                'all_scores': [],
                'details': {},
                'market_status': '待完成'
            }
        
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
        ranking = [{'code': 'CASH', 'name': self.data.cash_name, 'score': 0.0, 'is_cash': 1}]
        for code, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            ranking.append({
                'code': code,
                'name': all_details[code]['name'],
                'score': score,
                'is_cash': 0
            })
        
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'recommendation': recommendation['code'],
            'recommend_name': recommendation['name'],
            'confidence': recommendation.get('score', 0.0),
            'cash_reason': cash_reason if should_cash else None,
            'should_cash': 1 if should_cash else 0,
            'all_scores': ranking,
            'details': all_details,
            'market_status': '熊市' if should_cash else '正常'
        }

# 默认创建"追涨杀跌"策略实例（使用延迟加载，只在首次请求时初始化）
strategy = None

def get_current_strategy(strategy_type='momentum'):
    """获取策略实例，使用单例模式避免重复初始化"""
    global strategy
    if strategy is None or strategy.strategy_type != strategy_type:
        strategy = Strategy(strategy_type=strategy_type)
    return strategy

# ============ 网页界面（多策略卡片版） ============
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF AI投资助手 - 多策略平台</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .container { max-width: 1200px; margin: 0 auto; }
        
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { 
            color: white; 
            font-size: 36px; 
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .header p { 
            color: rgba(255,255,255,0.8); 
            font-size: 16px; 
            text-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        
        /* ========== 策略选择页面 ========== */
        .strategy-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }
        
        .strategy-card {
            background: white;
            border-radius: 20px;
            padding: 24px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
            border: 3px solid transparent;
            position: relative;
            overflow: hidden;
        }
        
        .strategy-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--color);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }
        
        .strategy-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 16px 32px rgba(0,0,0,0.2);
            border-color: var(--color);
        }
        
        .strategy-card:hover::before {
            transform: scaleX(1);
        }
        
        .strategy-card.active {
            border-color: var(--color);
            background: linear-gradient(135deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
            box-shadow: 0 16px 32px rgba(102,126,234,0.3);
        }
        
        .strategy-icon {
            font-size: 32px;
            margin-bottom: 12px;
        }
        
        .strategy-title {
            font-size: 20px;
            font-weight: bold;
            color: var(--color);
            margin-bottom: 4px;
        }
        
        .strategy-subtitle {
            font-size: 12px;
            color: #999;
            margin-bottom: 12px;
            font-weight: 500;
        }
        
        .strategy-desc {
            font-size: 14px;
            color: #666;
            line-height: 1.6;
            margin-bottom: 16px;
            min-height: 40px;
        }
        
        .strategy-profession {
            background: var(--color);
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 12px;
            display: inline-block;
        }
        
        .strategy-detail {
            font-size: 12px;
            color: #888;
            line-height: 1.5;
            border-top: 1px solid #f0f0f0;
            padding-top: 12px;
        }
        
        .strategy-badge {
            position: absolute;
            top: 12px;
            right: 12px;
            background: var(--color);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: bold;
        }
        
        /* ========== 内容页面 ========== */
        .content-page {
            display: none;
        }
        
        .content-page.active {
            display: block;
        }
        
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
            background: var(--color);
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
            background: linear-gradient(135deg, var(--color) 0%, rgba(0,0,0,0.1));
            color: white;
        }
        
        .tag {
            background: rgba(255,255,255,0.2);
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            margin-bottom: 16px;
        }
        
        .etf-code { font-size: 48px; font-weight: bold; margin: 10px 0; }
        .etf-name { font-size: 20px; opacity: 0.9; margin-bottom: 20px; }
        
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
            color: var(--color);
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
        .stat-item.highlight { background: #e3f2fd; border: 2px solid var(--color); }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            display: block;
        }
        .stat-value.positive { color: var(--color); }
        .stat-value.negative { color: #67c23a; }
        .stat-label { font-size: 12px; color: #999; margin-top: 4px; }
        
        /* 决策记录样式 */
        .decision-list {
            max-height: 600px;
            overflow-y: auto;
        }
        .decision-item {
            border-left: 4px solid var(--color);
            background: #f8f9fa;
            padding: 16px;
            margin-bottom: 12px;
            border-radius: 0 12px 12px 0;
        }
        
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
            background: var(--color);
            color: white;
        }
        
        .decision-body {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .decision-main { flex: 1; }
        .decision-from-to { font-size: 16px; color: #333; margin-bottom: 4px; }
        .decision-arrow { color: var(--color); font-weight: bold; margin: 0 8px; }
        .decision-reason { font-size: 13px; color: #666; }
        
        /* 排名列表 */
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
        .rank-1 { background: #ffd700 !important; color: #333 !important; }
        .rank-2 { background: #c0c0c0 !important; color: #333 !important; }
        .rank-3 { background: #cd7f32 !important; color: white !important; }
        .rank-info { flex: 1; margin-left: 12px; }
        .rank-name { font-size: 16px; font-weight: 500; color: #333; display: block; }
        .rank-code { font-size: 12px; color: #999; margin-top: 2px; }
        .rank-score { font-size: 20px; font-weight: bold; color: var(--color); font-family: 'Courier New', monospace; }
        
        .back-button {
            display: inline-block;
            padding: 10px 20px;
            background: white;
            color: var(--color);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            margin-bottom: 20px;
            transition: all 0.3s;
        }
        
        .back-button:hover {
            background: var(--color);
            color: white;
        }
        
        .loading { text-align: center; padding: 60px; color: #999; }
        
        @media (max-width: 768px) {
            .strategy-grid {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 24px;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 ETF AI投资助手</h1>
            <p>智能多策略平台 | 选择适合你的投资风格</p>
        </div>
        
        <!-- 策略选择页面 -->
        <div id="strategy-selection-page">
            <div class="warning">
                💡 <strong>平台说明：</strong>该平台支持多种投资风格。选择一个策略卡片，查看实时推荐、回测表现和决策记录。
            </div>
            
            <div class="strategy-grid" id="strategy-grid"></div>
        </div>
        
        <!-- 内容页面（选择策略后显示） -->
        <div id="content-page" class="content-page">
            <button class="back-button" onclick="backToSelection()">← 返回策略选择</button>
            
            <div class="nav-tabs">
                <button class="nav-tab active" onclick="switchPage('dashboard')">📊 实时推荐</button>
                <button class="nav-tab" onclick="switchPage('backtest')">📈 回测收益</button>
                <button class="nav-tab" onclick="switchPage('decisions')">📋 决策记录</button>
            </div>
            
            <!-- 页面1: 实时推荐 -->
            <div id="page-dashboard" class="page active">
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
                    <div class="stats-grid" id="stats-grid"></div>
                </div>
            </div>
            
            <!-- 页面3: 决策记录 -->
            <div id="page-decisions" class="page">
                <div class="card">
                    <div class="chart-header">
                        <span class="chart-title">📋 历史决策记录（最近50条）</span>
                    </div>
                    <div class="decision-list" id="decision-list">
                        <div class="loading">加载中...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 策略配置（与后端STRATEGIES一致）
        const strategies = {
            'momentum': {
                'name': '追涨杀跌',
                'english': 'Momentum Trading',
                'description': '追踪市场热点，快速响应场景变化',
                'profession': '隔壁老翁',
                'detail': '喜欢追涨杀跌，追踪热点赛道，快速切换持仓，高风险高收益，市场情绪主导交易决策。',
                'style': '激进型',
                'color': '#667eea',
                'icon': '⚡'
            },
            'value': {
                'name': '稳健派跌',
                'english': 'Conservative Dividend Strategy',
                'description': '坚守20日均线，专注高股息白马股，宏观避险第一',
                'profession': '白马猎手',
                'detail': '专注银行、电力等高分红白马股，以20日均线为防线，破线即卖，规避宏观政策风险，追求稳定收益。',
                'style': '稳健型',
                'color': '#11998e',
                'icon': '🏛️'
            },
            'balanced': {
                'name': '量化均衡',
                'english': 'Balanced Strategy',
                'description': '风险与收益平衡配置，追求稳定增长',
                'profession': 'Quant工程师',
                'detail': '用代码优化交易逻辑，用数据说话，追求量化回测表现。通过技术指标和统计模型精确控制风险，打造稳定的投资系统。',
                'style': '量化型',
                'color': '#f59e0b',
                'icon': '⚖️'
            },
            'growth': {
                'name': '信仰成长',
                'english': 'Growth Investing',
                'description': '投资高增长企业，布局未来赛道',
                'profession': '赛道探险家',
                'detail': '甄别优质成长赛道，布局产业升级方向，追求长期产业浪潮。',
                'style': '成长型',
                'color': '#ec4899',
                'icon': '🚀'
            }
        };
        
        let currentStrategy = 'momentum';  // 默认策略
        let currentPeriod = 'month';
        let returnChart = null;
        let backtestData = null;
        
        // 策略选择（接受元素引用和策略ID，避免依赖全局 event）
        function selectStrategy(elem, strategyId) {
            currentStrategy = strategyId;

            // 更新UI样式
            document.querySelectorAll('.strategy-card').forEach(card => {
                card.classList.remove('active');
            });
            // 使用传入的元素定位并添加 active
            elem.closest('.strategy-card').classList.add('active');

            // 显示内容页面，隐藏选择页面
            document.getElementById('strategy-selection-page').style.display = 'none';
            document.getElementById('content-page').classList.add('active');

            // 设置颜色变量
            const color = strategies[strategyId].color;
            document.documentElement.style.setProperty('--color', color);

            // 加载推荐
            loadRecommendation();
        }
        
        // 返回策略选择页面
        function backToSelection() {
            document.getElementById('strategy-selection-page').style.display = 'block';
            document.getElementById('content-page').classList.remove('active');
            document.querySelectorAll('.strategy-card').forEach(card => {
                card.classList.remove('active');
            });
        }
        
        // 初始化策略卡片
        function initStrategies() {
            const grid = document.getElementById('strategy-grid');
            let html = '';
            
            for (const [key, strategy] of Object.entries(strategies)) {
                html += `
                    <div class="strategy-card" onclick="selectStrategy(this, '${key}')" style="--color: ${strategy.color}">
                        <div class="strategy-icon">${strategy.icon}</div>
                        <div class="strategy-title">${strategy.name}</div>
                        <div class="strategy-subtitle">${strategy.english}</div>
                        <div class="strategy-desc">${strategy.description}</div>
                        <div class="strategy-profession">${strategy.profession}</div>
                        <div class="strategy-detail">${strategy.detail}</div>
                        <div class="strategy-badge" style="background: ${strategy.color};">${strategy.style}</div>
                    </div>
                `;
            }
            
            grid.innerHTML = html;
        }
        
        // 切换内容页面
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
                const res = await fetch(`/api/recommend?strategy=${currentStrategy}`);
                const data = await res.json();
                if (!data) return;
                
                // 检查策略是否已实现
                if (data.status === 'unimplemented') {
                    const html = `
                        <div class="card recommend-card" style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
                            <div style="font-size: 48px; margin-bottom: 20px;">🔧</div>
                            <div style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #333;">
                                ${data.recommend_name}
                            </div>
                            <div style="font-size: 16px; color: #666; margin-bottom: 30px;">
                                该策略正在开发中，敬请期待！
                            </div>
                            <div style="font-size: 14px; color: #999; padding: 20px; background: rgba(255,255,255,0.8); border-radius: 8px;">
                                我们正在精心打磨这个策略，争取为您提供更优质的投资建议。<br>
                                请先使用其他已完成的策略吧！
                            </div>
                        </div>
                    `;
                    document.getElementById('recommend-section').innerHTML = html;
                    document.getElementById('ranking-list').innerHTML = '';
                    return;
                }
                
                const rec = data.recommendation;
                const isCash = data.should_cash;
                
                let html = `
                    <div class="card recommend-card">
                        <span class="tag">
                            ${isCash ? '⚠️ 建议空仓' : '🏆 今日推荐买入'}
                        </span>
                        <div class="etf-code">${rec}</div>
                        <div class="etf-name">${data.recommend_name}</div>
                `;
                
                if (!isCash) {
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
                
                // 排名列表
                let rankHtml = '';
                data.all_scores.forEach((item, idx) => {
                    const isCashItem = item.is_cash;
                    const rankClass = isCashItem ? '' : (idx <= 3 ? `rank-${idx}` : '');
                    
                    rankHtml += `
                        <div class="rank-item">
                            <div class="rank-num ${rankClass}">
                                ${isCashItem ? '💰' : idx}
                            </div>
                            <div class="rank-info">
                                <span class="rank-name">${item.name}</span>
                                ${!isCashItem ? `<span class="rank-code">${item.code}</span>` : ''}
                            </div>
                            <div class="rank-score">
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
                // 检查是否是未完成策略
                const unimplementedStrategies = ['growth'];
                if (unimplementedStrategies.includes(currentStrategy)) {
                    document.getElementById('stats-grid').innerHTML = 
                        `<div class="loading" style="padding: 40px; text-align: center;">
                            <div style="font-size: 48px; margin-bottom: 20px;">🔧</div>
                            <div>该策略正在开发中，回测功能敬请期待！</div>
                        </div>`;
                    return;
                }
                
                // 根据 period 确定天数
                const periodDays = {
                    'week': 7,
                    'month': 30,
                    'half': 180,
                    'year': 365
                };
                const days = periodDays[period] || 365;
                
                const res = await fetch(`/api/backtest?strategy=${currentStrategy}&period=${period}&days=${days}`);
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
                        <span class="stat-value">${metrics.cash_ratio}%</span>
                        <span class="stat-label">空仓占比</span>
                    </div>
                `;
                
                drawChart(data.chart_data, metrics.total_return >= 0);
                
            } catch (e) {
                console.error('加载回测失败:', e);
            }
        }
        
        async function loadDecisions() {
            try {
                // 检查是否是未完成策略
                const unimplementedStrategies = ['growth'];
                if (unimplementedStrategies.includes(currentStrategy)) {
                    const listEl = document.getElementById('decision-list');
                    listEl.innerHTML = 
                        `<div class="loading" style="padding: 40px; text-align: center;">
                            <div style="font-size: 48px; margin-bottom: 20px;">🔧</div>
                            <div>该策略正在开发中，决策历史敬请期待！</div>
                        </div>`;
                    return;
                }
                
                if (!backtestData) {
                    const res = await fetch(`/api/backtest?strategy=${currentStrategy}&period=year&days=365`);
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
                    const actionText = {
                        'BUY': '买入',
                        'SELL': '卖出',
                        'SWITCH': '换仓',
                        'HOLD': '持有',
                        'CASH': '空仓'
                    }[d.action] || d.action;
                    
                    html += `
                        <div class="decision-item">
                            <div class="decision-header">
                                <span class="decision-date">${d.date}</span>
                                <span class="decision-action">${actionText}</span>
                            </div>
                            <div class="decision-body">
                                <div class="decision-main">
                                    <div class="decision-from-to">
                                        ${d.prev_holding === 'CASH' ? '💰' : d.prev_holding}
                                        <span class="decision-arrow">→</span>
                                        ${d.decision === 'CASH' ? '💰' : d.decision}
                                    </div>
                                    <div class="decision-reason">${d.reason}</div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                listEl.innerHTML = html;
                
            } catch (e) {
                console.error('加载决策记录失败:', e);
            }
        }
        
        function drawChart(chartData, isPositive) {
            const ctx = document.getElementById('returnChart').getContext('2d');
            
            if (returnChart) {
                returnChart.destroy();
            }
            
            const color = getComputedStyle(document.documentElement).getPropertyValue('--color').trim();
            const labels = chartData.map(d => d.date);
            const values = chartData.map(d => d.value);
            
            const gradient = ctx.createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, color + '40');
            gradient.addColorStop(1, color + '00');
            
            returnChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '策略净值',
                        data: values,
                        borderColor: color,
                        backgroundColor: gradient,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
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
                                    return [
                                        `净值: ¥${context.parsed.y.toFixed(2)}`,
                                        `收益率: ${item.return_pct >= 0 ? '+' : ''}${item.return_pct}%`
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
        window.addEventListener('load', function() {
            initStrategies();
            document.documentElement.style.setProperty('--color', strategies.momentum.color);
        });
    </script>
</body>
</html>
"""

# ============ API路由 ============"""

# ============ API路由 ============

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

def get_strategy(strategy_type='momentum'):
    """根据策略类型返回对应的策略实例（使用缓存）"""
    return get_current_strategy(strategy_type)

@app.route('/api/recommend', methods=['GET'])
def recommend():
    strategy_id = request.args.get('strategy', 'momentum')
    # 为每个请求创建对应策略的实例
    current_strategy = get_strategy(strategy_id)
    result = current_strategy.get_recommendation()
    if result:
        # 添加策略元信息
        result['strategy'] = strategy_id
        result['strategy_name'] = STRATEGIES[strategy_id]['name']
    return jsonify(result)

@app.route('/api/backtest', methods=['GET'])
def backtest():
    strategy_id = request.args.get('strategy', 'momentum')
    period = request.args.get('period', 'month')
    days = int(request.args.get('days', 365))
    
    # 为每个请求创建对应策略的实例
    current_strategy = get_strategy(strategy_id)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    metrics = current_strategy.backtest.run_backtest(start_date, end_date)
    
    if not metrics:
        return jsonify({"error": "回测失败"})
    
    chart_data = current_strategy.backtest.get_chart_data(period)
    
    return jsonify({
        "metrics": metrics,
        "chart_data": chart_data,
        "period": period,
        "strategy": strategy_id,
        "strategy_name": STRATEGIES[strategy_id]['name']
    })

if __name__ == '__main__':
    import os
    import sys
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # 在Render等云平台上，使用gunicorn启动
    if os.environ.get('RENDER'):
        # Render环境：使用gunicorn
        os.system(f'gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:{port} app:app')
    else:
        # 本地开发：使用Flask自带服务器
        app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)