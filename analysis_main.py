"""
@File         :   analysis_main.py
@Time         :   2023/07/22 14:41:21
@Author       :   yjn
@Contact      :   yinjunnan1@gmail.com
@Version      :   1.0
@Desc         :   分析函数执行的入口文件
"""
from Analysis.train_val_test_data_names.gen_data_names import gen_data_names_main


def main():
    gen_data_names_main()


if __name__ == '__main__':
    main()
