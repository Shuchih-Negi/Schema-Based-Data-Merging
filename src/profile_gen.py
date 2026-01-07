from pyspark.sql import functions as F

def profile_column(df):
    result = {}
    for col_name in df.columns:
        total_count = df.count()

        # NULL count
        null_count = (
            df.select(col_name).where(F.col(col_name).isNull()).count()
        )

        null_pct = (null_count / total_count * 100) if total_count > 0 else 0.0

        # non-null examples
        examples = (
            df
            .select(col_name)
            .where(F.col(col_name).isNotNull())
            .limit(2)
            .collect()
        )

        ex0 = examples[0][0] if len(examples) > 0 else None
        ex1 = examples[1][0] if len(examples) > 1 else None

        result[col_name]  = {
            "type": dict(df.dtypes)[col_name],
            "ex0": ex0,
            "ex1": ex1,
            "NA": round(null_pct, 2)
            }
    return  result
