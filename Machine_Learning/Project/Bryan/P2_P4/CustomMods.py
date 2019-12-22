#!/usr/bin/env python3

def merge_data(dfl, dfr, col_name):
	for column in dfr.columns.values:
		col = col_name + str(column)
		dfl[col] = dfr[column]
