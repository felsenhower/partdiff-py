#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

[ "${SHELLCHECK:-0}" == "1" ] && shellcheck "$0"

if (( "$#" < 1 )) ; then
    echo 'Error: Must supply path to reference implementation!'
    exit 1
fi

REFERENCE="$1"

GAUSS_SEIDEL_ARGS=(1 1 100 2 2 100)
JACOBI_ARGS=(1 2 100 2 2 100)

VARIANTS=(simple numba np_vectorize)

declare -A PYTHON_VERSIONS=(
    [simple]='cpython3.13'
    [numba]='cpython3.10'
    [np_vectorize]='cpython3.13'
)

OUTPUT_FILE="$(realpath benchmark_results.csv)"

cd ..

export LANG=C
export LC_NUMERIC=C

function format_runtime {
    awk '
        /Calculation time/{printf(" %-16.2f,", $3);};
        /real/{printf(" %-13.2f\n", $2);};
    '
}

{
    printf '%-12s, %s, %-11s, %-16s, %-13s\n' variant i method runtime_internal runtime_total
    
    for i in {1..3} ; do
        printf '%-12s, %s, %-11s,' 'reference' "$i" 'Gauß-Seidel'
        { time -p "${REFERENCE}" "${GAUSS_SEIDEL_ARGS[@]}" ; } 2>&1 | format_runtime
        printf '%-12s, %s, %-11s,' 'reference' "$i" 'Jacobi'
        { time -p "${REFERENCE}" "${JACOBI_ARGS[@]}" ; } 2>&1 | format_runtime
    done
    
    for variant in "${VARIANTS[@]}" ; do
        pushd "$variant" > /dev/null
        python="${PYTHON_VERSIONS["$variant"]}"
        # i=0 is added here to warm up the JIT for the numba version, if needed.
        for i in {0..3} ; do
            printf '%-12s, %s, %-11s,' "$variant" "$i" 'Gauß-Seidel'
            { time -p uv run --python "$python" main.py "${GAUSS_SEIDEL_ARGS[@]}" ; } 2>&1 | format_runtime
            printf '%-12s, %s, %-11s,' "$variant" "$i" 'Jacobi'
            { time -p uv run --python "$python" main.py "${JACOBI_ARGS[@]}" ; } 2>&1 | format_runtime
        done
        popd > /dev/null
    done
} | tee "$OUTPUT_FILE"
