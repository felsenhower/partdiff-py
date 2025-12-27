import polars as pl
from io import StringIO
from uncertainties import ufloat

def preprocess_file() -> StringIO:
    sio = StringIO()
    with open("benchmark_results.csv") as f:
        for line in f:
            line = line.strip()
            line = line.split(",")
            line = [f.strip() for f in line]
            line = ",".join(line)
            sio.write(f"{line}\n")
    sio.seek(0)
    return sio

def extract_baseline_runtime(df, method: str, col_name: str):
    return ufloat(*(df.filter(pl.col("variant") == "reference", pl.col("method") == method).to_dicts()[0][col_name]))

def main():
    df = pl.read_csv(preprocess_file())
    df = df.filter(
        pl.col("i") > 0
    )
    df = (
        df
        .group_by(["variant", "method"], maintain_order=True)
        .agg(
            pl.col("runtime_internal").mean().alias("runtime_internal_mean"),
            pl.col("runtime_internal").std().alias("runtime_internal_std"),
            pl.col("runtime_total").mean().alias("runtime_total_mean"),
            pl.col("runtime_total").std().alias("runtime_total_std"),
        )
    )
    df = df.select(
        pl.col("variant"),
        pl.col("method"),
        pl.concat_list("runtime_internal_mean", "runtime_internal_std").alias("runtime_internal"),
        pl.concat_list("runtime_total_mean", "runtime_total_std").alias("runtime_total")
    )

    def runtime_factor(method, runtime, time_type):
        res = ufloat(*runtime) * 100  / extract_baseline_runtime(df, method, time_type)
        return res.n # [res.n, res.s]
    
    df = df.with_columns(
        
        pl.struct(["method", "runtime_internal"]).map_elements(
            lambda x: runtime_factor(x["method"], x["runtime_internal"], "runtime_internal"),
            return_dtype=pl.Float64(),
            # return_dtype=pl.List(pl.Float64()),
        ).alias("runtime_internal_factor"),
        
        pl.struct(["method", "runtime_total"]).map_elements(
            lambda x: runtime_factor(x["method"], x["runtime_total"], "runtime_total"),
            return_dtype=pl.Float64(),
            # return_dtype=pl.List(pl.Float64()),
        ).alias("runtime_total_factor"),
        
    )
        
        
    def format_ufloat_runtime(x) -> str:
        x = ufloat(*x)
        return "({:.3f} ± {:.3f}) s".format(x.n, x.s)
        
    df = df.with_columns(
        pl.col("runtime_internal").map_elements(lambda x: format_ufloat_runtime(x), return_dtype=pl.String()),
        pl.col("runtime_total").map_elements(lambda x: format_ufloat_runtime(x), return_dtype=pl.String()),
        pl.col("runtime_internal_factor").map_elements(lambda x: f"{x:.2f}%", return_dtype=pl.String()),
        pl.col("runtime_total_factor").map_elements(lambda x: f"{x:.2f}%", return_dtype=pl.String()),
    )
    
    df = df.select(
        pl.col("variant"),
        pl.col("method"),
        pl.col("runtime_internal"),
        pl.col("runtime_internal_factor"),
        pl.col("runtime_total"),
        pl.col("runtime_total_factor")
    )

    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
    ):
        s = str(df)
        for col in ("runtime_internal_factor", "runtime_total_factor"):
            s = s.replace(col, " " * len(col))
        print(s)


if __name__ == "__main__":
    main()
