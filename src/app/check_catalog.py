from .m01_data_ingestion import ingest

def main():
    df = ingest()
    print("\n📊 DataFrame obtenido:")
    print(df.head())

if __name__ == "__main__":
    main()
