import csv

def same_lines_csvs(path1, path2):
    with open(path1, newline='', encoding="utf-8") as f1, \
        open(path2, newline='', encoding="utf-8") as f2:
        rows1 = list(csv.reader(f1))
        rows2 = list(csv.reader(f2))
        same_first_column = sorted(row[0] for row in rows1) == sorted(row[0] for row in rows2)
        same_rows = sorted(rows1) == sorted(rows2)
        print(f"first column {same_first_column}, all csv {same_rows}")
        return same_first_column, same_rows
    
