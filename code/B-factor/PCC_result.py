import glob
import pandas as pd
import os
import numpy as np

# 134+7  couple = np.arange(1,31,1) 1~30
pdbID=[	'1YJO', '2OL9', '2OLX', '3FVA', '3HYD', '3NVG', '3Q2X',
	'1F8R', '1GCO', '1H6V', '1IDP', '1KMM', '1QKI', '1WLY', '2A50', '2AH1', '2BCM', '2COV', '2D5W', '2DPL', '2E10', '2ETX', '2FN9', '2I49', '2IMF', '2J32', '2J9W',
    '2O6X', '2POF', '2PSF', '2Q52', '2RE2', '2VE8', '2VPA', '2VYO', '2W1V', '2W2A', '2X1Q', '2X9Z', '2XHF', '2Y7L', '2YLB', '2YNY', '2ZCM', '2ZU1', '3AMC', '3B5O',
    '3BA1', '3DRF', '3DWV', '3FTD', '3G1S', '3HHP', '3K6Y', '3L41', '3LG3', '3LJI', '3M3P', '3MGN', '3MRE', '3N11', '3NE0', '3NPV', '3OQY', '3PID', '3PKV', '3PTL',
	'3PVE', '3PZ9', '3QDS', '3R6D', '3SEB', '3SR3', '3SUK', '3SZH', '3T0H', '3TDN', '3TUA', '3U6G', '3UR8', '3US6', '3V1A', '3V75', '3VN0', '3VOR', '3VUB', '3VVV',
	'3VZ9', '3W4Q', '3ZBD', '3ZIT', '3ZRX', '3ZSL', '3ZZP', '3ZZY', '4A02', '4ACJ', '4AE7', '4AM1', '4ANN', '4AVR', '4AXY', '4B6G', '4B9G', '4DD5', '4DKN', '4DND',
	'4DPZ', '4DQ7', '4DT4', '4EK3', '4ERY', '4ES1', '4EUG', '4F01', '4F3J', '4FR9', '4G14', '4G2E', '4G5X', '4G6C', '4G7X', '4GA2', '4GMQ', '4GS3', '4H4J', '4H89',
	'4HDE', '4HJP', '4HWM', '4IL7', '4J11', '4J5O', '4J5Q', '4J78', '4JG2', '4JVU', '4JYP', '4KEF', '5CYT', '6RXN']

# 223 couple = 0~30/0~20；
# pdbID=['1ABA', '1AHO', '1AIE', '1AKG', '1ATG', '1BGF', '1BX7', '1BYI', '1CCR', '1CYO', '1DF4', '1E5K', '1ES5', '1ETL', '1ETM', '1ETN', '1EW4', '1FF4', '1FK5', '1GK7',
#        '1GVD', '1GXU', '1HJE', '1I71', '1IFR', '1K8U', '1KNG', '1KR4', '1KYC', '1LR7', '1MF7', '1N7E', '1NKD', '1NKO', '1NLS', '1NNX',
# # couple = np.arange(0,21,1)
#        '1NOA', '1NOT', '1O06', '1O08', '1OB4', '1OB7', '1OPD', '1P9I', '1PEF', '1PEN', '1PMY', '1PZ4', '1Q9B', '1QAU', '1QTO', '1R29', '1R7J', '1RJU', '1RRO', '1SAU', '1TGR', '1TZV', '1U06', '1U7I',
#        '1U9C', '1UHA', '1UKU', '1ULR', '1UOY', '1USE', '1USM', '1UTG', '1V05', '1V70', '1VRZ', '1W2L', '1WBE', '1WHI', '1WPA', '1X3O', '1XY1', '1XY2', '1Y6X', '1YZM',
#        '1Z21', '1ZCE', '1ZVA', '2AGK', '2B0A', '2BF9', '2BRF', '2C71', '2CE0', '2CG7', '2CWS', '2DKO', '2DSX', '2E3H', '2EAQ', '2EHP', '2EHS', '2ERW', '2FB6', '2FG1',
#        '2FQ3', '2G69', '2G7O', '2G7S', '2GKG', '2GOM', '2GXG', '2GZQ', '2HQK', '2HYK', '2I24', '2IBL', '2IGD', '2IP6', '2IVY', '2JKU', '2JLI', '2JLJ', '2MCM', '2NLS',
#        '2NR7', '2NUH', '2OA2', '2OCT', '2OHW', '2OKT', '2PKT', '2PLT', '2PMR', '2PPN', '2PTH', '2Q4N', '2QJL', '2R16', '2R6Q', '2RB8', '2RFR', '2V9V', '2VH7', '2VIM',
#        '2VQ4', '2VY8', '2W6A', '2WJ5', '2WUJ', '2WW7', '2WWE', '2X25', '2X3M', '2X5Y', '2Y0T', '2Y72', '2Y9F', '3A0M', '3A7L', '3AUB', '3BED', '3BQX', '3BZQ', '3BZZ',
#        '3E5T', '3E7R', '3EUR', '3F2Z', '3F7E', '3FCN', '3FE7', '3FKE', '3FMY', '3FOD', '3FSO', '3GBW', '3GHJ', '3HFO', '3HNY', '3HP4', '3HWU', '3HZ8', '3I2V', '3I2Z',
#        '3I4O', '3I7M', '3IHS', '3IVV', '3KBE', '3KGK', '3KZD', '3LAA', '3LAX', '3M8J', '3M9J', '3M9Q', '3MAB', '3MD4', '3MD5', '3MEA', '3NGG', '3NZL', '3O0P', '3O5P',
#        '3OBQ', '3P6J', '3PD7', '3PES', '3PIW', '3PSM', '3PZZ', '3Q6L', '3QPA', '3R87', '3RQ9', '3RY0', '3RZY', '3S0A', '3SD2', '3SED', '3SO6', '3T3K', '3T47', '3TOW',
#        '3TYS', '3U97', '3UCI']

# 定义目标文件夹路径和文件名模板
for i in range(len(pdbID)):
    print(pdbID[i])
    csv_path = ('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/result/1_364_3_1/1_%s_3_1/csv_%s_3_1' % ( pdbID[i], pdbID[i]))
    n_num = len(open(r'/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID[i], 'r').readlines())
    print(n_num)
    # file_pattern = f'/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/PCC_223/result/PCC_{pdbID[i]}_*.csv'  #通配符*将匹配所有满足模式的文件名
    file_pattern = f'/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/PCC/result/PCC_{pdbID[i]}_*.csv'  # 通配符*将匹配所有满足模式的文件名

    file_list = glob.glob(file_pattern)

    # 初始化最大值
    max_Pearson = -1.0
    max_couple, max_cutoff, max_RMSE_value, max_R_squared = 0.0, 0.0, 0.0, 0.0

    # 遍历目标文件夹下所有pdbID对应的csv文件
    for file in file_list:
        # 读取csv文件
        df = pd.read_csv(file, header=None)  # 没有标题行

        # 获取R_squared、R_squared_adj、couple、cutoff值
        couple = df.iloc[0, 2]
        cutoff = df.iloc[0, 3]
        Pearson = df.iloc[0, 4]
        RMSE_value = df.iloc[0, 5]
        R_squared = df.iloc[0, 6]
        # 更新 Pearson 最大值
        if Pearson > max_Pearson:
            max_couple = couple
            max_cutoff = cutoff
            max_Pearson = Pearson
            max_RMSE_value = RMSE_value
            max_R_squared = R_squared

    # 创建一个字典，将 pdbID 等存储为键值对
    d_P = {'pdbID': [pdbID[i]], 'n_num': [n_num],
           'couple': [max_couple], 'cutoff':[max_cutoff],
           'max_Pearson': [np.around(max_Pearson, 3)],
           'RMSE': [np.around(max_RMSE_value, 3)],
           'R_squared': [np.around(max_R_squared, 3)]}
    # 将字典转换为 DataFrame
    df_result = pd.DataFrame(d_P)
    # 将 DataFrame 写入 csv 文件，每一行末尾不加换行符
    # with open('result_223_PCC.csv', 'a') as f:
    with open('result_134+7_PCC.csv', 'a') as f:
        df_result.to_csv(f, header=f.tell() == 0, index=False)