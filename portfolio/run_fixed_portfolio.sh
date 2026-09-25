#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FACTOR_BASE="$ROOT_DIR/factor_analyse/factor_mining"
PYTHON_BIN="${PORTFOLIO_PYTHON:-$ROOT_DIR/.venv/bin/python}"

factor_file="$SCRIPT_DIR/factor_selection.txt"
h5_file="$ROOT_DIR/data/crypto_quant.h5"
output_root="$ROOT_DIR/reports/portfolio"
start_date=""
end_date=""
as_of_date=""
rebalance_days=5
cutoffs=()

usage() {
    cat <<'EOF'
Usage:
  portfolio/run_fixed_portfolio.sh \
    --start YYYY-MM-DD --end YYYY-MM-DD --as-of YYYY-MM-DD [options]

Required:
  --start DATE             First signal date
  --end DATE               Last signal date
  --as-of DATE             Knowledge cutoff; include enough tail for liquidation

Options:
  --factors FILE           Factor list, relative names under factor_mining/
                           (default: portfolio/factor_selection.txt)
  --h5 FILE                CryptoQuant H5 store (default: data/crypto_quant.h5)
  --rebalance-days N       Rebalance interval (default: 5)
  --cutoff DATE            Cutoff audit date; may be repeated
  --output-root DIR        Run/config output root (default: reports/portfolio)
  -h, --help               Show this help

Every selected factor receives equal allocation. Blank lines and lines starting
with # are ignored. The script runs freeze, run, and (when cutoffs are supplied)
audit in fresh Python processes.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "missing value for $1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --factors)
            require_value "$@"
            factor_file="$2"
            shift 2
            ;;
        --h5)
            require_value "$@"
            h5_file="$2"
            shift 2
            ;;
        --start)
            require_value "$@"
            start_date="$2"
            shift 2
            ;;
        --end)
            require_value "$@"
            end_date="$2"
            shift 2
            ;;
        --as-of)
            require_value "$@"
            as_of_date="$2"
            shift 2
            ;;
        --rebalance-days)
            require_value "$@"
            rebalance_days="$2"
            shift 2
            ;;
        --cutoff)
            require_value "$@"
            cutoffs+=("$2")
            shift 2
            ;;
        --output-root)
            require_value "$@"
            output_root="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$start_date" || -z "$end_date" || -z "$as_of_date" ]]; then
    echo "--start, --end, and --as-of are required" >&2
    usage >&2
    exit 2
fi
if [[ ! "$rebalance_days" =~ ^[1-9][0-9]*$ ]]; then
    echo "--rebalance-days must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

case "$factor_file" in
    /*) ;;
    *) factor_file="$ROOT_DIR/$factor_file" ;;
esac
case "$h5_file" in
    /*) ;;
    *) h5_file="$ROOT_DIR/$h5_file" ;;
esac
case "$output_root" in
    /*) ;;
    *) output_root="$ROOT_DIR/$output_root" ;;
esac

if [[ ! -f "$factor_file" ]]; then
    echo "factor list not found: $factor_file" >&2
    exit 1
fi
if [[ ! -f "$h5_file" ]]; then
    echo "H5 store not found: $h5_file" >&2
    exit 1
fi

factor_paths=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    case "$line" in
        /*|../*|*/../*|*/..)
            echo "unsafe factor path: $line" >&2
            exit 1
            ;;
        *.py) ;;
        *)
            echo "factor entry must end in .py: $line" >&2
            exit 1
            ;;
    esac
    factor_path="$FACTOR_BASE/$line"
    if [[ ! -f "$factor_path" ]]; then
        echo "factor file not found: $factor_path" >&2
        exit 1
    fi
    factor_paths+=("$factor_path")
done < "$factor_file"

factor_count=${#factor_paths[@]}
if [[ "$factor_count" -eq 0 ]]; then
    echo "factor list contains no factors: $factor_file" >&2
    exit 1
fi
allocation="$(awk -v count="$factor_count" 'BEGIN { printf "%.15g", 1 / count }')"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
config_dir="$output_root/configs"
mkdir -p "$config_dir"
frozen_config="$config_dir/frozen_${timestamp}.json"
audit_receipt="$config_dir/audit_${timestamp}.json"

freeze_args=(
    -m portfolio.main_fixed freeze
    --h5 "$h5_file"
    --start "$start_date"
    --end "$end_date"
    --as-of "$as_of_date"
    --rebalance-days "$rebalance_days"
    --anchor-date "$start_date"
    --output "$frozen_config"
)
for factor_path in "${factor_paths[@]}"; do
    freeze_args+=(--factor "$factor_path" --allocation "$allocation")
done

echo "Selected factors: $factor_count"
echo "Equal allocation:  $allocation"
printf '  %s\n' "${factor_paths[@]}"
echo "Frozen config:     $frozen_config"

cd "$ROOT_DIR"
"$PYTHON_BIN" "${freeze_args[@]}"
"$PYTHON_BIN" -m portfolio.main_fixed run \
    --config "$frozen_config" \
    --output-root "$output_root"

if [[ ${#cutoffs[@]} -gt 0 ]]; then
    audit_args=(
        -m portfolio.main_fixed audit
        --config "$frozen_config"
        --output "$audit_receipt"
    )
    for cutoff in "${cutoffs[@]}"; do
        audit_args+=(--cutoff "$cutoff")
    done
    "$PYTHON_BIN" "${audit_args[@]}"
    echo "Audit receipt:     $audit_receipt"
else
    echo "Audit skipped: no --cutoff supplied"
fi
