import pyarrow.parquet as pq

path = "data/cleaned/helios/HEL1OS_FILTERED.parquet"

pf = pq.ParquetFile(path)

print("\nROWS:")
print(pf.metadata.num_rows)

print("\nROW GROUPS:")
print(pf.num_row_groups)

print("\nCOLUMNS:")
print(pf.schema.names)

print("\nFIRST ROW GROUP:")
table = pf.read_row_group(0)

print(table.to_pandas().head())