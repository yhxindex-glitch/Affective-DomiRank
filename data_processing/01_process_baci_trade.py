"""
处理BACI贸易数据（2017-2021）
提取40个国家×12个行业的双边贸易流
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.expanded_config import COUNTRIES, INDUSTRIES

class BACIProcessor:
    """BACI数据处理器"""
    
    def __init__(self):
        self.data_dir = Path('data/raw/archive')
        self.output_dir = Path('data/processed/baci')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 从配置获取目标国家和行业（必须先设置）
        self.target_countries = COUNTRIES
        self.target_industries = INDUSTRIES
        
        print(f"目标国家: {len(self.target_countries)}")
        print(f"目标行业: {len(self.target_industries)}")
        
        # 加载国家和产品代码映射
        self.country_map = self._load_country_codes()
        self.product_map = self._load_product_codes()
    
    def _load_country_codes(self):
        """加载BACI国家代码映射"""
        country_file = self.data_dir / 'country_codes.csv'
        if not country_file.exists():
            print(f"⚠️  国家代码文件不存在: {country_file}")
            return {}
        
        df = pd.read_csv(country_file)
        print(f"\n✓ 加载国家代码: {len(df)} 个国家")
        print(f"  列名: {df.columns.tolist()}")
        
        # 建立ISO3到BACI代码的映射
        country_map = {}
        
        # 尝试识别列名
        code_col = None
        name_col = None
        iso3_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'country_code' in col_lower or col == 'i':
                code_col = col
            elif 'country_name' in col_lower or 'name' in col_lower:
                name_col = col
            elif 'iso' in col_lower or 'iso3' in col_lower:
                iso3_col = col
        
        if code_col and (name_col or iso3_col):
            for _, row in df.iterrows():
                baci_code = row[code_col]
                
                # 通过ISO3匹配
                if iso3_col and pd.notna(row.get(iso3_col)):
                    iso3 = str(row[iso3_col]).strip()
                    for country_name, country_info in self.target_countries.items():
                        if country_info['iso3'] == iso3:
                            country_map[country_info['iso3']] = int(baci_code)
                            break
                
                # 通过名称匹配
                if name_col and pd.notna(row.get(name_col)):
                    name = str(row[name_col]).strip().lower()
                    for country_name, country_info in self.target_countries.items():
                        target_name = country_name.lower().replace('_', ' ')
                        if target_name in name or name in target_name:
                            if country_info['iso3'] not in country_map:
                                country_map[country_info['iso3']] = int(baci_code)
        
        print(f"  映射成功: {len(country_map)}/{len(self.target_countries)} 个目标国家")
        
        # 显示映射结果
        for country_name, country_info in self.target_countries.items():
            iso3 = country_info['iso3']
            if iso3 in country_map:
                print(f"    ✓ {country_name:20s} ({iso3}) → {country_map[iso3]}")
            else:
                print(f"    ✗ {country_name:20s} ({iso3}) → 未找到")
        
        return country_map
    
    def _load_product_codes(self):
        """加载BACI产品代码映射"""
        product_file = self.data_dir / 'product_codes.csv'
        if not product_file.exists():
            print(f"\n⚠️  产品代码文件不存在: {product_file}")
            return {}
        
        df = pd.read_csv(product_file)
        print(f"\n✓ 加载产品代码: {len(df)} 个产品")
        
        # 建立HS2到HS6产品列表的映射
        product_map = {}
        
        # 识别产品代码列
        code_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'code' in col_lower or 'hs' in col_lower or 'product' in col_lower:
                code_col = col
                break
        
        if code_col:
            for industry_name, industry_info in self.target_industries.items():
                hs2_codes = industry_info['hs2']
                hs6_products = []
                
                for hs2 in hs2_codes:
                    # BACI使用HS6代码（6位数字）
                    # 找到所有以该HS2开头的HS6代码
                    matches = df[df[code_col].astype(str).str.zfill(6).str.startswith(hs2)]
                    hs6_products.extend(matches[code_col].tolist())
                
                product_map[industry_name] = hs6_products
                print(f"    {industry_name:20s} (HS{'/'.join(hs2_codes):10s}): {len(hs6_products)} 个HS6产品")
        
        return product_map
    
    def process_year(self, year: int) -> pd.DataFrame:
        """处理单个年份的BACI数据"""
        print(f"\n{'='*60}")
        print(f"处理 {year} 年数据...")
        print(f"{'='*60}")
        
        year_file = self.data_dir / f'Year_{year}' / f'Year_{year}.csv'
        if not year_file.exists():
            print(f"✗ 文件不存在: {year_file}")
            return pd.DataFrame()
        
        print(f"读取文件: {year_file.name}")
        df = pd.read_csv(year_file)
        print(f"  原始记录数: {len(df):,}")
        print(f"  列名: {df.columns.tolist()}")
        
        # 识别列名
        year_col = 't'
        exporter_col = 'i'
        importer_col = 'j'
        product_col = 'k'
        value_col = 'v'
        quantity_col = 'q'
        
        # 过滤目标国家
        target_baci_codes = list(self.country_map.values())
        df_filtered = df[
            df[exporter_col].isin(target_baci_codes) & 
            df[importer_col].isin(target_baci_codes)
        ].copy()
        
        print(f"  过滤目标国家后: {len(df_filtered):,} 条")
        
        # 构建反向映射（BACI代码→ISO3）
        baci_to_iso3 = {v: k for k, v in self.country_map.items()}
        
        # 构建国家-行业级别的贸易数据
        records = []
        
        for industry_name, hs6_products in self.product_map.items():
            if not hs6_products:
                continue
            
            # 过滤该行业的产品
            df_industry = df_filtered[df_filtered[product_col].isin(hs6_products)]
            
            if df_industry.empty:
                continue
            
            # 按出口国-进口国聚合
            agg_data = df_industry.groupby([exporter_col, importer_col]).agg({
                value_col: 'sum',
                quantity_col: 'sum'
            }).reset_index()
            
            # 添加国家和行业信息
            for _, row in agg_data.iterrows():
                exporter_iso3 = baci_to_iso3.get(row[exporter_col])
                importer_iso3 = baci_to_iso3.get(row[importer_col])
                
                if exporter_iso3 and importer_iso3:
                    # 查找国家名称
                    exporter_name = None
                    importer_name = None
                    for name, info in self.target_countries.items():
                        if info['iso3'] == exporter_iso3:
                            exporter_name = name
                        if info['iso3'] == importer_iso3:
                            importer_name = name
                    
                    records.append({
                        'year': year,
                        'exporter': exporter_name,
                        'exporter_iso3': exporter_iso3,
                        'importer': importer_name,
                        'importer_iso3': importer_iso3,
                        'industry': industry_name,
                        'trade_value': row[value_col],  # 千美元
                        'trade_quantity': row[quantity_col]  # 吨
                    })
            
            print(f"    {industry_name:20s}: {len(agg_data):,} 条贸易流")
        
        result_df = pd.DataFrame(records)
        print(f"\n  最终记录数: {len(result_df):,}")
        
        if not result_df.empty:
            print(f"  国家对数: {result_df[['exporter', 'importer']].drop_duplicates().shape[0]}")
            print(f"  行业数: {result_df['industry'].nunique()}")
            print(f"  总贸易额: ${result_df['trade_value'].sum()/1000000:.1f}M")
        
        return result_df
    
    def process_all_years(self, start_year=2017, end_year=2021):
        """处理所有年份的数据"""
        print("\n" + "="*80)
        print("BACI数据处理: 2017-2021")
        print("="*80)
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            df = self.process_year(year)
            
            if not df.empty:
                all_data.append(df)
                
                # 保存年度数据
                output_file = self.output_dir / f'baci_trade_{year}.csv'
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"  ✓ 已保存: {output_file}")
        
        if all_data:
            # 合并所有年份
            combined_df = pd.concat(all_data, ignore_index=True)
            output_file = self.output_dir / 'baci_trade_2017_2021.csv'
            combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✓ 合并数据已保存: {output_file}")
            
            # 统计
            print(f"\n" + "="*80)
            print("总体统计:")
            print(f"  总记录数: {len(combined_df):,}")
            print(f"  年份: {combined_df['year'].unique()}")
            print(f"  国家数: {combined_df['exporter'].nunique()}")
            print(f"  行业数: {combined_df['industry'].nunique()}")
            print(f"  总贸易额: ${combined_df['trade_value'].sum()/1000000:.1f}M")
            
            print(f"\n按年份统计:")
            year_stats = combined_df.groupby('year').agg({
                'trade_value': 'sum',
                'exporter': 'nunique'
            })
            for year, row in year_stats.iterrows():
                print(f"  {year}: ${row['trade_value']/1000000:.1f}M, {row['exporter']} 个国家")
            
            print(f"\n按行业统计（前10）:")
            industry_stats = combined_df.groupby('industry')['trade_value'].sum().sort_values(ascending=False)
            for industry, value in industry_stats.head(10).items():
                print(f"  {industry:20s}: ${value/1000000:.1f}M")
        
        print("\n✓ BACI数据处理完成！")


def main():
    """主函数"""
    processor = BACIProcessor()
    
    print("\n是否开始处理？这可能需要几分钟...")
    choice = input("继续？[y/N]: ").strip().lower()
    
    if choice == 'y':
        processor.process_all_years(2017, 2021)
    else:
        print("已取消")


if __name__ == "__main__":
    main()

