from barcode_decoding import run_decoding

df = run_decoding(
    "../example_data/FAM_example.tif",
    "../example_data/Cy3_example.tif"
)

df.to_csv(
    "../example_output/decoding_result.csv",
    index=False
)

print(df.head())
