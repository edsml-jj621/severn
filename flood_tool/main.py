"""Script for user interface"""
import sys
import argparse
from argparse import RawTextHelpFormatter
import textwrap
import pandas as pd
from flood_tool.tool import Tool


def main():
    parser = argparse.ArgumentParser(
        description="Flood Risk Tool", formatter_class=RawTextHelpFormatter)
    parser.add_argument('-t', '--label_type',
                        type=str,
                        nargs=1,
                        required=True,
                        help='''
Type of labelling.
-t flood_risk: label flood risk class
-t house_price: label median house price
                        ''')
    parser.add_argument('-f', '--unlabelled_file',
                        type=str,
                        nargs=1,
                        required=True,
                        help='''
Unlabelled postcodes file.
-f postcodes.csv
                        ''')
    parser.add_argument('-m', '--method',
                        type=str,
                        nargs=1,
                        help='''
The regression / classifier method.

Flood Risk

default knn
-m dt:  Decision Tree
-m knn:  KNN
-m rmdf:  Random Forest
-m ada:  AdaBoost

House Price

default rfr
-m lr:  Linear Regression
-m dt:  Dscision Tree
-m rfr: Random Forest Regression
-m sv:  SVR Support Vector Regression
''')
    parser.add_argument('-s', '--save',
                        type=str,
                        nargs=1,
                        help='''
default labelled.csv

output filename / filepath
''')
    args = parser.parse_args()
    (type, file) = (args.label_type[0], args.unlabelled_file[0])
    save = "labelled.csv" if not args.save else args.save[0]
    if type == "flood_risk":
        method = "knn" if not args.method else args.method[0]
        if method not in ["dt", "knn", "rdmf", "ada"]:
            print("Method not supported")
        else:
            label(type, file, method, save)
    elif type == "house_price":
        method = "rfr" if not args.method else args.method[0]
        if method not in ["lr", "dt", "rfr", "sv"]:
            print("Method not supported")
        else:
            label(type, file, method, save)
    else:
        print("Type not supported")


def label(type, file_path, method, save):
    tool = Tool(file_path)
    postcodes = tool.postcode_df["postcode"]
    if type == "flood_risk":
        df = tool.get_flood_class(postcodes, method).to_frame()
    else:
        df = tool.get_median_house_price_estimate(postcodes, method).to_frame()
    df.reset_index(drop=True, inplace=True)
    df = pd.concat(
        [tool.postcode_df, df], axis=1)
    print(df)
    df.to_csv(save, index=False)


if __name__ == '__main__':
    main()
