"""
扩展后的国家和行业配置
用于BACI 2017-2021年度数据 + Business News情感分析
"""

# ===== 60个主要贸易国 =====
COUNTRIES = {
    # 原有14国
    'Germany': {'iso3': 'DEU', 'region': 'Europe', 'tier': 1},
    'USA': {'iso3': 'USA', 'region': 'Americas', 'tier': 1},
    'Japan': {'iso3': 'JPN', 'region': 'Asia', 'tier': 1},
    'Italy': {'iso3': 'ITA', 'region': 'Europe', 'tier': 1},
    'UK': {'iso3': 'GBR', 'region': 'Europe', 'tier': 1},
    'China': {'iso3': 'CHN', 'region': 'Asia', 'tier': 1},
    'France': {'iso3': 'FRA', 'region': 'Europe', 'tier': 1},
    'Netherlands': {'iso3': 'NLD', 'region': 'Europe', 'tier': 1},
    'Mexico': {'iso3': 'MEX', 'region': 'Americas', 'tier': 1},
    'Canada': {'iso3': 'CAN', 'region': 'Americas', 'tier': 1},
    'Austria': {'iso3': 'AUT', 'region': 'Europe', 'tier': 1},
    'Spain': {'iso3': 'ESP', 'region': 'Europe', 'tier': 1},
    'Sweden': {'iso3': 'SWE', 'region': 'Europe', 'tier': 1},
    'South_Korea': {'iso3': 'KOR', 'region': 'Asia', 'tier': 1},
    
    # 扩展26国（主要贸易伙伴）
    'India': {'iso3': 'IND', 'region': 'Asia', 'tier': 2},
    'Brazil': {'iso3': 'BRA', 'region': 'Americas', 'tier': 2},
    'Russia': {'iso3': 'RUS', 'region': 'Europe', 'tier': 2},
    'Turkey': {'iso3': 'TUR', 'region': 'Europe', 'tier': 2},
    'Poland': {'iso3': 'POL', 'region': 'Europe', 'tier': 2},
    'Indonesia': {'iso3': 'IDN', 'region': 'Asia', 'tier': 2},
    'Thailand': {'iso3': 'THA', 'region': 'Asia', 'tier': 2},
    'Vietnam': {'iso3': 'VNM', 'region': 'Asia', 'tier': 2},
    'Malaysia': {'iso3': 'MYS', 'region': 'Asia', 'tier': 2},
    'Singapore': {'iso3': 'SGP', 'region': 'Asia', 'tier': 2},
    'Belgium': {'iso3': 'BEL', 'region': 'Europe', 'tier': 2},
    'Switzerland': {'iso3': 'CHE', 'region': 'Europe', 'tier': 2},
    'Norway': {'iso3': 'NOR', 'region': 'Europe', 'tier': 2},
    'Denmark': {'iso3': 'DNK', 'region': 'Europe', 'tier': 2},
    'Finland': {'iso3': 'FIN', 'region': 'Europe', 'tier': 2},
    'Czech': {'iso3': 'CZE', 'region': 'Europe', 'tier': 2},
    'Ireland': {'iso3': 'IRL', 'region': 'Europe', 'tier': 2},
    'Portugal': {'iso3': 'PRT', 'region': 'Europe', 'tier': 2},
    'Greece': {'iso3': 'GRC', 'region': 'Europe', 'tier': 2},
    'Hungary': {'iso3': 'HUN', 'region': 'Europe', 'tier': 2},
    'Australia': {'iso3': 'AUS', 'region': 'Oceania', 'tier': 2},
    'New_Zealand': {'iso3': 'NZL', 'region': 'Oceania', 'tier': 2},
    'Israel': {'iso3': 'ISR', 'region': 'Asia', 'tier': 2},
    'Saudi_Arabia': {'iso3': 'SAU', 'region': 'Asia', 'tier': 2},
    'UAE': {'iso3': 'ARE', 'region': 'Asia', 'tier': 2},
    'South_Africa': {'iso3': 'ZAF', 'region': 'Africa', 'tier': 2},
    'Argentina': {'iso3': 'ARG', 'region': 'Americas', 'tier': 2},
    'Chile': {'iso3': 'CHL', 'region': 'Americas', 'tier': 2},
    'Colombia': {'iso3': 'COL', 'region': 'Americas', 'tier': 2},
    'Philippines': {'iso3': 'PHL', 'region': 'Asia', 'tier': 2},
    
    # 新增20国（扩展到60国）
    'Hong_Kong': {'iso3': 'HKG', 'region': 'Asia', 'tier': 3},
    'Taiwan': {'iso3': 'TWN', 'region': 'Asia', 'tier': 3},
    'Pakistan': {'iso3': 'PAK', 'region': 'Asia', 'tier': 3},
    'Bangladesh': {'iso3': 'BGD', 'region': 'Asia', 'tier': 3},
    'Egypt': {'iso3': 'EGY', 'region': 'Africa', 'tier': 3},
    'Nigeria': {'iso3': 'NGA', 'region': 'Africa', 'tier': 3},
    'Kenya': {'iso3': 'KEN', 'region': 'Africa', 'tier': 3},
    'Morocco': {'iso3': 'MAR', 'region': 'Africa', 'tier': 3},
    'Peru': {'iso3': 'PER', 'region': 'Americas', 'tier': 3},
    'Ecuador': {'iso3': 'ECU', 'region': 'Americas', 'tier': 3},
    'Ukraine': {'iso3': 'UKR', 'region': 'Europe', 'tier': 3},
    'Romania': {'iso3': 'ROU', 'region': 'Europe', 'tier': 3},
    'Bulgaria': {'iso3': 'BGR', 'region': 'Europe', 'tier': 3},
    'Croatia': {'iso3': 'HRV', 'region': 'Europe', 'tier': 3},
    'Slovakia': {'iso3': 'SVK', 'region': 'Europe', 'tier': 3},
    'Slovenia': {'iso3': 'SVN', 'region': 'Europe', 'tier': 3},
    'Luxembourg': {'iso3': 'LUX', 'region': 'Europe', 'tier': 3},
    'Lithuania': {'iso3': 'LTU', 'region': 'Europe', 'tier': 3},
    'Latvia': {'iso3': 'LVA', 'region': 'Europe', 'tier': 3},
    'Estonia': {'iso3': 'EST', 'region': 'Europe', 'tier': 3},
}

# ===== 20个关键行业 =====
INDUSTRIES = {
    # 原有5个行业
    'Automotive': {
        'hs2': ['87'],
        'description': '车辆及其零件',
        'keywords': ['car', 'auto', 'vehicle', 'tesla', 'toyota', 'ford', 'volkswagen', 'bmw']
    },
    'Steel': {
        'hs2': ['72'],
        'description': '钢铁',
        'keywords': ['steel', 'iron', 'metal', 'thyssenkrupp', 'arcelormittal']
    },
    'Plastics': {
        'hs2': ['39'],
        'description': '塑料及其制品',
        'keywords': ['plastic', 'polymer', 'basf', 'dow', 'dupont']
    },
    'Electronics': {
        'hs2': ['85'],
        'description': '电机、电气设备',
        'keywords': ['electronics', 'chip', 'semiconductor', 'intel', 'samsung', 'tsmc']
    },
    'Machinery': {
        'hs2': ['84'],
        'description': '机械器具',
        'keywords': ['machinery', 'equipment', 'caterpillar', 'siemens', 'bosch']
    },
    
    # 扩展7个行业
    'Pharmaceuticals': {
        'hs2': ['30'],
        'description': '药品',
        'keywords': ['pharma', 'drug', 'medicine', 'pfizer', 'novartis', 'roche']
    },
    'Chemicals': {
        'hs2': ['28', '29'],
        'description': '化工产品',
        'keywords': ['chemical', 'chemistry', 'basf', 'dow']
    },
    'Textiles': {
        'hs2': ['61', '62', '63'],
        'description': '纺织品',
        'keywords': ['textile', 'fabric', 'clothing', 'garment']
    },
    'Energy': {
        'hs2': ['27'],
        'description': '矿物燃料、能源',
        'keywords': ['energy', 'oil', 'gas', 'coal', 'petroleum']
    },
    'Food': {
        'hs2': ['01', '02', '03', '04', '07', '08', '10', '11', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
        'description': '食品和农产品',
        'keywords': ['food', 'agriculture', 'farming', 'crop']
    },
    'Wood': {
        'hs2': ['44', '45', '46'],
        'description': '木材及制品',
        'keywords': ['wood', 'timber', 'lumber', 'furniture']
    },
    'Paper': {
        'hs2': ['47', '48', '49'],
        'description': '纸张及制品',
        'keywords': ['paper', 'pulp', 'printing']
    },
    
    # 新增8个行业（扩展到20行业）
    'Rubber': {
        'hs2': ['40'],
        'description': '橡胶及其制品',
        'keywords': ['rubber', 'tire', 'michelin', 'goodyear']
    },
    'Footwear': {
        'hs2': ['64'],
        'description': '鞋类',
        'keywords': ['shoe', 'footwear', 'boot', 'nike', 'adidas']
    },
    'Furniture': {
        'hs2': ['94'],
        'description': '家具',
        'keywords': ['furniture', 'chair', 'table', 'ikea']
    },
    'Optical_Instruments': {
        'hs2': ['90'],
        'description': '光学、医疗仪器',
        'keywords': ['optical', 'medical equipment', 'instrument', 'lens']
    },
    'Copper': {
        'hs2': ['74'],
        'description': '铜及其制品',
        'keywords': ['copper', 'bronze']
    },
    'Aluminum': {
        'hs2': ['76'],
        'description': '铝及其制品',
        'keywords': ['aluminum', 'aluminium']
    },
    'Transport_Equipment': {
        'hs2': ['86', '88', '89'],
        'description': '运输设备（铁路、船舶、航空）',
        'keywords': ['railway', 'aircraft', 'ship', 'boeing', 'airbus']
    },
    'Minerals': {
        'hs2': ['25', '26'],
        'description': '矿产品',
        'keywords': ['mineral', 'ore', 'mining']
    },
}

# ===== 计算统计 =====
NUM_COUNTRIES = len(COUNTRIES)
NUM_INDUSTRIES = len(INDUSTRIES)
NUM_NODES = NUM_COUNTRIES * NUM_INDUSTRIES
NUM_YEARS = 5  # 2017-2021
EXPECTED_SAMPLES = NUM_NODES * NUM_YEARS

print(f"""
配置统计：
====================
国家数: {NUM_COUNTRIES}
行业数: {NUM_INDUSTRIES}
总节点数: {NUM_NODES} ({NUM_COUNTRIES} × {NUM_INDUSTRIES})
年份数: {NUM_YEARS}
预期训练样本: {EXPECTED_SAMPLES} ({NUM_NODES} × {NUM_YEARS})
""")

# ===== BACI国家代码映射（需要查询country_codes.csv）=====
# 将在处理BACI数据时根据country_codes.csv自动映射
BACI_COUNTRY_MAPPING = {}

# ===== 导出配置 =====
__all__ = ['COUNTRIES', 'INDUSTRIES', 'NUM_COUNTRIES', 'NUM_INDUSTRIES', 
           'NUM_NODES', 'NUM_YEARS', 'EXPECTED_SAMPLES']

