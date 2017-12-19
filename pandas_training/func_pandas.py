import os
import pandas as pd

basedir = os.path.abspath(os.path.dirname(__file__))

def hundredth_row(column):
    hundredth_item = column.loc[99]
    return hundredth_item

def not_null_count(column):
    column_null = pd.isnull(column)
    null = column[column_null]
    return len(null)