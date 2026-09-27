"""
处理Business News数据（2017-2021）
提取并聚合国家-行业级别的情感分数
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
sys.path.append(str(Path(__file__).parent.parent))

from configs.expanded_config import COUNTRIES, INDUSTRIES

class BusinessNewsProcessor:
    """Business News情感处理器"""
    
    def __init__(self):
        self.data_dir = Path('data/raw/business_news')
        self.output_dir = Path('data/processed/news_sentiment')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 从配置获取目标国家和行业
        self.target_countries = COUNTRIES
        self.target_industries = INDUSTRIES
        
        # 构建关键词映射
        self.country_keywords = self._build_country_keywords()
        self.industry_keywords = self._build_industry_keywords()
        
        print(f"目标国家: {len(self.target_countries)}")
        print(f"目标行业: {len(self.target_industries)}")
    
    def _build_country_keywords(self):
        """构建国家关键词"""
        keywords = {}
        for country_name, info in self.target_countries.items():
            # 基础关键词
            base_keywords = [
                country_name.lower().replace('_', ' '),
                info['iso3'].lower()
            ]
            
            # 添加常见别名
            aliases = {
                'USA': ['america', 'us ', 'united states'],
                'UK': ['britain', 'united kingdom', 'british'],
                'South_Korea': ['korea', 'korean', 'south korea'],
                'UAE': ['emirates', 'dubai'],
                'Czech': ['czech republic', 'czechia'],
                'Netherlands': ['dutch', 'holland'],
                'Switzerland': ['swiss']
            }
            
            if country_name in aliases:
                base_keywords.extend(aliases[country_name])
            
            keywords[country_name] = base_keywords
        
        return keywords
    
    def _build_industry_keywords(self):
        """构建行业关键词"""
        keywords = {}
        for industry_name, info in self.target_industries.items():
            keywords[industry_name] = [kw.lower() for kw in info['keywords']]
        return keywords
    
    def match_countries(self, text: str):
        """从文本中匹配国家"""
        text_lower = text.lower()
        matched = []
        
        for country_name, keywords in self.country_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched.append(country_name)
                    break
        
        return matched
    
    def match_industries(self, text: str):
        """从文本中匹配行业"""
        text_lower = text.lower()
        matched = []
        
        for industry_name, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched.append(industry_name)
                    break
        
        return matched
    
    def process_news_data(self):
        """处理新闻数据"""
        print("\n" + "="*80)
        print("处理Business News数据...")
        print("="*80)
        
        # 读取数据
        news_file = self.data_dir / 'business_news_sentiment.csv'
        if not news_file.exists():
            print(f"✗ 文件不存在: {news_file}")
            return
        
        print(f"\n读取文件: {news_file.name}")
        df = pd.read_csv(news_file)
        print(f"  总记录数: {len(df):,}")
        print(f"  列名: {df.columns.tolist()}")
        
        # 解析日期
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # 过滤2017-2021
        df = df[(df['Date'] >= '2017-01-01') & (df['Date'] <= '2021-12-31')].copy()
        print(f"  2017-2021数据: {len(df):,} 条")
        
        # 提取年份
        df['year'] = df['Date'].dt.year
        
        # 转换情感标签为数值
        # POSITIVE=1, NEGATIVE=-1
        sentiment_map = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
        df['sentiment_score'] = df['sentiment'].map(sentiment_map)
        df = df.dropna(subset=['sentiment_score'])
        
        print(f"\n情感分布:")
        print(df['sentiment'].value_counts())
        
        # 匹配国家和行业
        print(f"\n匹配国家和行业...")
        df['countries'] = df['Title'].apply(self.match_countries)
        df['industries'] = df['Title'].apply(self.match_industries)
        
        # 只保留有匹配的记录
        df_matched = df[
            (df['countries'].str.len() > 0) & 
            (df['industries'].str.len() > 0)
        ].copy()
        
        print(f"  匹配成功: {len(df_matched):,}/{len(df):,} ({len(df_matched)/len(df)*100:.1f}%)")
        
        # 展开为多条记录（一条新闻可能匹配多个国家-行业对）
        records = []
        for _, row in df_matched.iterrows():
            for country in row['countries']:
                for industry in row['industries']:
                    records.append({
                        'year': row['year'],
                        'country': country,
                        'industry': industry,
                        'sentiment_score': row['sentiment_score'],
                        'confidence': row.get('confidence', 1.0)
                    })
        
        df_expanded = pd.DataFrame(records)
        print(f"  展开后记录数: {len(df_expanded):,}")
        
        # 按年份-国家-行业聚合
        print(f"\n按年份-国家-行业聚合...")
        agg_data = df_expanded.groupby(['year', 'country', 'industry']).agg({
            'sentiment_score': ['mean', 'count', 'std'],
            'confidence': 'mean'
        }).reset_index()
        
        # 展平列名
        agg_data.columns = ['year', 'country', 'industry', 
                           'sentiment_mean', 'news_count', 'sentiment_std', 
                           'confidence_mean']
        
        print(f"  聚合后: {len(agg_data)} 个国家-行业-年份组合")
        
        # 保存结果
        output_file = self.output_dir / 'news_sentiment_aggregated.csv'
        agg_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ 已保存: {output_file}")
        
        # 统计
        print(f"\n" + "="*80)
        print("统计信息:")
        print(f"  年份: {sorted(agg_data['year'].unique())}")
        print(f"  国家数: {agg_data['country'].nunique()}")
        print(f"  行业数: {agg_data['industry'].nunique()}")
        print(f"  平均每组新闻数: {agg_data['news_count'].mean():.1f}")
        print(f"  平均情感分数: {agg_data['sentiment_mean'].mean():.3f}")
        
        print(f"\n按年份统计:")
        year_stats = agg_data.groupby('year').agg({
            'sentiment_mean': 'mean',
            'news_count': 'sum'
        })
        for year, row in year_stats.iterrows():
            print(f"  {year}: 情感={row['sentiment_mean']:+.3f}, 新闻数={int(row['news_count'])}")
        
        print(f"\n按国家统计（前10）:")
        country_stats = agg_data.groupby('country')['news_count'].sum().sort_values(ascending=False)
        for country, count in country_stats.head(10).items():
            print(f"  {country:20s}: {int(count)} 条新闻")
        
        print(f"\n按行业统计:")
        industry_stats = agg_data.groupby('industry')['news_count'].sum().sort_values(ascending=False)
        for industry, count in industry_stats.items():
            print(f"  {industry:20s}: {int(count)} 条新闻")
        
        # 创建完整的国家-行业-年份矩阵（填充缺失值）
        print(f"\n创建完整矩阵...")
        self._create_complete_matrix(agg_data)
        
        print("\n✓ Business News处理完成！")
    
    def _create_complete_matrix(self, agg_data):
        """创建完整的国家-行业-年份矩阵"""
        # 获取所有组合
        years = [2017, 2018, 2019, 2020, 2021]
        countries = list(self.target_countries.keys())
        industries = list(self.target_industries.keys())
        
        # 创建完整索引
        from itertools import product
        full_index = pd.DataFrame(
            list(product(years, countries, industries)),
            columns=['year', 'country', 'industry']
        )
        
        # 合并数据
        complete_data = full_index.merge(
            agg_data,
            on=['year', 'country', 'industry'],
            how='left'
        )
        
        # 填充缺失值
        complete_data['sentiment_mean'].fillna(0, inplace=True)
        complete_data['news_count'].fillna(0, inplace=True)
        complete_data['sentiment_std'].fillna(0, inplace=True)
        complete_data['confidence_mean'].fillna(0, inplace=True)
        
        # 保存
        output_file = self.output_dir / 'news_sentiment_complete.csv'
        complete_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  ✓ 完整矩阵已保存: {output_file}")
        print(f"    总记录数: {len(complete_data):,} ({len(years)}年 × {len(countries)}国 × {len(industries)}行业)")
        print(f"    有新闻数据: {(complete_data['news_count'] > 0).sum()} ({(complete_data['news_count'] > 0).sum()/len(complete_data)*100:.1f}%)")


def main():
    """主函数"""
    processor = BusinessNewsProcessor()
    
    print("\n是否开始处理？")
    choice = input("继续？[y/N]: ").strip().lower()
    
    if choice == 'y':
        processor.process_news_data()
    else:
        print("已取消")


if __name__ == "__main__":
    main()

