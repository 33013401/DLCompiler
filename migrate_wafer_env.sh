#!/usr/bin/env bash
# 使用 Bash 执行本脚本。

# 任何命令失败、使用未定义变量或管道中任意命令失败时立即退出。
set -euo pipefail

# 当前脚本所在目录；脚本可以放在工作目录，也可以放在 DLCompiler 仓库内。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 新版 DLCompiler 的 Git 地址。
DL_REPO_URL=${DL_REPO_URL:-git@github.com:33013401/DLCompiler.git}
# 默认使用已经完成 Wafer 迁移的远程分支。
DL_REPO_BRANCH=${DL_REPO_BRANCH:-wafer-migration-squashed}
# 脚本在仓库内时复用该仓库；脚本在工作目录时将仓库克隆到 ./DLCompiler。
if [[ -d "$SCRIPT_DIR/.git" || -f "$SCRIPT_DIR/.git" ]]; then
    WORK_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
    REPO_DIR=${REPO_DIR:-$SCRIPT_DIR}
else
    WORK_ROOT="$SCRIPT_DIR"
    REPO_DIR=${REPO_DIR:-$WORK_ROOT/DLCompiler}
fi
# 解压后的 LLVM、TX8、Kuiper 等依赖的根目录，可由外部覆盖。
DEPS_ROOT=${DEPS_ROOT:-$WORK_ROOT/deps}
# 下载归档、wheel 和校验清单的缓存目录，可由外部覆盖。
PACKAGE_ROOT=${PACKAGE_ROOT:-$WORK_ROOT/packages}
# 默认创建或使用名为 wafer310 的 Conda 环境。
ENV_NAME=${ENV_NAME:-wafer310}
# 已有 Conda 的安装根目录；为空时后面默认查找 $HOME/miniconda3。
CONDA_ROOT=${CONDA_ROOT:-}
# 已有 Python 3.10 可执行文件；非空时跳过 Conda 环境创建。
PYTHON=${PYTHON:-}
# 0 表示完整迁移，1 表示只准备编译器，不安装硬件 runtime。
COMPILER_ONLY=0
# 当前变量预留给后续清理下载缓存逻辑，默认保留下载文件。
KEEP_DOWNLOADS=${KEEP_DOWNLOADS:-1}

# 当前已知的唯一 TXDA 发布包；只有供应方发布匹配 manifest 时才允许覆盖。
TORCH_TXDA_TAR_URL=${TORCH_TXDA_TAR_URL:-https://toolchain-jfrog.wafer.xyz/artifactory/tx8-generic-dev/torch_txda/torch_txda%2Btxops-20251230-03541ed8%2B71a1e5a.tar.gz}
# TX8 依赖归档地址；脚本同目录存在本地归档时不需要设置。
TX8_DEPS_TAR_URL=${TX8_DEPS_TAR_URL:-}
# 可显式指定 TX8 本地归档；未指定时自动查找脚本同目录下的 tx8_depends*.tar.gz。
TX8_DEPS_ARCHIVE=${TX8_DEPS_ARCHIVE:-}
# Kuiper SDK/runtime 归档地址，完整模式必须由执行人显式提供。
KUIPER_TAR_URL=${KUIPER_TAR_URL:-}
# 与 TXDA/txops 同一发布组合的 PyTorch wheel 地址。
TORCH_WHEEL_URL=${TORCH_WHEEL_URL:-}

# 打印脚本用法和可覆盖的环境变量。
usage() {
    cat <<'EOF'
Usage: bash migrate_wafer_env.sh [--compiler-only]

The script clones the repository into ./DLCompiler when it is not already
running inside a DLCompiler Git checkout.

Repository branch:
    DL_REPO_BRANCH    Branch to clone (default: wafer-migration-squashed)

Required in all modes unless a local archive is present beside this script:
    TX8_DEPS_TAR_URL  URL of the TX8 dependency archive
    TX8_DEPS_ARCHIVE  Local TX8 archive path; overrides automatic local discovery

Required for the full hardware environment:
  KUIPER_TAR_URL    URL of the matching Kuiper SDK/runtime archive
    TORCH_WHEEL_URL   URL of the PyTorch wheel from the same TXDA release

Optional variables:
  DEPS_ROOT         Extracted dependency root (default: ../../deps)
  PACKAGE_ROOT      Download/cache root (default: ../../packages)
  ENV_NAME          Conda environment name (default: wafer310)
  CONDA_ROOT        Existing Conda installation root
    PYTHON            Existing Python 3.10 executable; skips Conda setup
    DL_REPO_URL       DLCompiler Git URL (default: git@github.com:33013401/DLCompiler.git)
    REPO_DIR          Clone/reuse destination (default: ./DLCompiler in workspace mode)
  KEEP_DOWNLOADS    Keep downloaded archives, default 1

The default TXDA URL is the documented 20251230 release and is CPython 3.10
only. Do not replace it with an unverified torch_txda or txops package.
EOF
}

# 逐个解析命令行参数。
while [[ $# -gt 0 ]]; do
    case "$1" in
        # 只执行编译器迁移，不下载/安装 TXDA、txops 和 Kuiper。
        --compiler-only)
            COMPILER_ONLY=1
            shift
            ;;
        # 打印帮助后正常退出。
        -h|--help)
            usage
            exit 0
            ;;
        # 未知参数直接报错，避免用户以为参数已生效。
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# 当前预编译 LLVM 和 TXDA 制品只针对 Linux x86_64。
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "ERROR: Wafer prebuilt dependencies currently require Linux x86_64" >&2
    exit 1
fi

# Wafer 编译器本身就需要 TX8 的 instr_def.h。优先使用脚本同目录的本地归档，
# 这样迁移脚本和 tx8 dev 包放在一起时无需配置内部制品库 URL。
if [[ -z "$TX8_DEPS_ARCHIVE" ]]; then
    TX8_DEPS_ARCHIVE=$(find "$SCRIPT_DIR" -maxdepth 1 -type f \
        \( -name 'tx8_depends*.tar.gz' -o -name 'tx8_deps*.tar.gz' \) \
        -print -quit)
fi
if [[ -n "$TX8_DEPS_ARCHIVE" && ! -f "$TX8_DEPS_ARCHIVE" ]]; then
    echo "ERROR: TX8_DEPS_ARCHIVE does not exist: $TX8_DEPS_ARCHIVE" >&2
    exit 1
fi
if [[ -z "$TX8_DEPS_ARCHIVE" && -z "$TX8_DEPS_TAR_URL" ]]; then
    echo "ERROR: place a tx8_depends*.tar.gz beside this script, or set TX8_DEPS_TAR_URL" >&2
    exit 1
fi
# 完整模式必须有 Kuiper 归档地址。
if [[ $COMPILER_ONLY == 0 && -z "$KUIPER_TAR_URL" ]]; then
    echo "ERROR: set KUIPER_TAR_URL for the full migration, or use --compiler-only" >&2
    exit 1
fi
# 完整模式必须有与 TXDA 发布组合匹配的 PyTorch wheel 地址。
if [[ $COMPILER_ONLY == 0 && -z "$TORCH_WHEEL_URL" ]]; then
    echo "ERROR: set TORCH_WHEEL_URL for the full migration, or use --compiler-only" >&2
    exit 1
fi

# 创建依赖解压目录、下载清单目录和各类制品缓存目录。
mkdir -p "$DEPS_ROOT" "$PACKAGE_ROOT/manifests" "$PACKAGE_ROOT/llvm" \
    "$PACKAGE_ROOT/tx8" "$PACKAGE_ROOT/kuiper" "$PACKAGE_ROOT/python"

# 统一打印阶段标题，便于查看迁移日志。
log() {
    printf '\n==> %s\n' "$*"
}

# 下载一个 URL；如果目标文件已存在且非空，则复用本地缓存。
download() {
    # 第一个参数是下载地址。
    local url=$1
    # 第二个参数是本地目标文件。
    local destination=$2
    # 已有缓存不重复下载，避免浪费带宽。
    if [[ -s "$destination" ]]; then
        echo "    exists: $destination"
        return
    fi
    echo "    downloading: $url"
    # --fail 遇到 HTTP 错误时失败，--location 跟随重定向，失败时重试三次。
    curl --fail --location --retry 3 --retry-delay 2 --output "$destination" "$url"
}

# 计算并追加下载文件的 SHA256，形成新机迁移记录。
record_sha256() {
    # 接收待记录的本地文件路径。
    local file=$1
    # 同时打印到终端并追加到 downloads.sha256。
    sha256sum "$file" | tee -a "$PACKAGE_ROOT/manifests/downloads.sha256"
}

# 如果脚本放在工作目录，则从 Git 克隆新版 DLCompiler；如果已在仓库内则直接复用。
log "Clone or reuse the DLCompiler repository"
if [[ ! -d "$REPO_DIR/.git" && ! -f "$REPO_DIR/.git" ]]; then
    if [[ -e "$REPO_DIR" && -n "$(find "$REPO_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "ERROR: target directory exists but is not a Git checkout: $REPO_DIR" >&2
        echo "Move it away or set REPO_DIR to an empty clone destination." >&2
        exit 1
    fi
    echo "    cloning $DL_REPO_URL branch $DL_REPO_BRANCH into $REPO_DIR"
    # 不使用 --recurse-submodules：它会把 ascendnpu-ir 及其 llvm-project 一并拉下来。
    git clone --branch "$DL_REPO_BRANCH" --single-branch \
        "$DL_REPO_URL" "$REPO_DIR"
fi

# 初始化 Triton 3.5 子模块，并确保它是项目锁定的提交。
log "Initialize the pinned Triton submodule"
cd "$REPO_DIR"
# 只初始化 Wafer 编译所需的 Triton，不初始化 ascendnpu-ir。
git submodule update --init --recursive third_party/triton
# 读取实际检出的 Triton commit。
TRITON_COMMIT=$(git -C third_party/triton rev-parse HEAD)
# Wafer 使用官方 Triton v3.5.0，并在构建时应用 DLCompiler 内维护的最小 patch。
EXPECTED_TRITON_COMMIT=c3c476f357f1e9768ea4e45aa5c17528449ab9ef
if [[ "$TRITON_COMMIT" != "$EXPECTED_TRITON_COMMIT" ]]; then
    echo "ERROR: unexpected Triton commit: $TRITON_COMMIT" >&2
    echo "       expected official Triton v3.5.0: $EXPECTED_TRITON_COMMIT" >&2
    exit 1
fi

# 准备 Python 3.10；这是当前 torch_txda cp310 wheel 的 ABI 要求。
log "Prepare the Python 3.10 build/runtime environment"
if [[ -z "$PYTHON" ]]; then
    # 没有显式 Python 时，使用 Conda 创建/激活隔离环境。
    if [[ -z "$CONDA_ROOT" ]]; then
        # 未指定 Conda 根目录时使用用户目录下的默认安装位置。
        CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
    fi
    # 检查 Conda 是否存在；本脚本不负责安装 Miniconda。
    if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
        echo "ERROR: Conda not found at $CONDA_ROOT" >&2
        echo "Install Miniconda first, or set PYTHON=/path/to/python3.10" >&2
        exit 1
    fi
    # 加载 Conda shell 函数，使 conda activate 在脚本中生效。
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    # 如果环境不存在，就创建 Python 3.10 环境。
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda create -n "$ENV_NAME" python=3.10 -y
    fi
    # 激活目标环境，并记录其中的 python 路径。
    conda activate "$ENV_NAME"
    PYTHON=$(command -v python)
else
    # 用户指定了 Python 时，只检查它确实是可执行文件。
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: Python executable not found: $PYTHON" >&2
        exit 1
    fi
fi

# 读取 Python 主次版本号。
PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
# cp310 的 torch_txda 不能用于 Python 3.12 或其他版本。
if [[ "$PYTHON_VERSION" != 3.10 ]]; then
    echo "ERROR: this migration script requires Python 3.10, got $PYTHON_VERSION" >&2
    exit 1
fi

# 更新基础打包工具。
$PYTHON -m pip install --upgrade pip setuptools wheel
# 安装编译 Wafer/Triton 所需的 Python 依赖。
$PYTHON -m pip install -r "$REPO_DIR/requirements.txt"

# 调用已有脚本下载并校验固定版本的 LLVM/MLIR 22。
log "Download and configure the pinned LLVM 22 package"
# setup_llvm22_env.sh 用 WORK_ROOT 拼出 LLVM 默认安装路径。
export WORK_ROOT="$DEPS_ROOT"
# 显式指定解压后的 LLVM 目录。
export LLVM_SYSPATH="$DEPS_ROOT/llvm-7d5de303-ubuntu-x64"
# 执行 LLVM 下载、版本和 CMake package 校验。
bash "$REPO_DIR/setup_llvm22_env.sh"
# 导入 LLVM、MLIR、PATH 等后续构建需要的环境变量。
source "$REPO_DIR/llvm22_env.sh"

# TX8 是编译器依赖，所有模式都下载；Kuiper 只在完整硬件模式下载。
log "Download and extract TX8 and Kuiper packages"
if [[ $COMPILER_ONLY == 0 ]]; then
    # 用 URL 的文件名作为本地缓存文件名。
    KUIPER_ARCHIVE="$PACKAGE_ROOT/kuiper/$(basename "${KUIPER_TAR_URL%%\?*}")"
    # 下载 Kuiper SDK/runtime 归档。
    download "$KUIPER_TAR_URL" "$KUIPER_ARCHIVE"
    record_sha256 "$KUIPER_ARCHIVE"
    # 删除旧 Kuiper 解压目录，避免新旧 SDK 文件混杂。
    rm -rf "$DEPS_ROOT/firmware_kuiper"
    # 解压 Kuiper 归档到依赖根目录。
    tar -xzf "$KUIPER_ARCHIVE" -C "$DEPS_ROOT"
fi

# 准备并解压 TX8；这是 compiler-only 也必须执行的步骤。
if [[ -z "$TX8_DEPS_ARCHIVE" ]]; then
    TX8_ARCHIVE="$PACKAGE_ROOT/tx8/$(basename "${TX8_DEPS_TAR_URL%%\?*}")"
    download "$TX8_DEPS_TAR_URL" "$TX8_ARCHIVE"
else
    TX8_ARCHIVE="$TX8_DEPS_ARCHIVE"
    echo "    using local TX8 archive: $TX8_ARCHIVE"
fi
record_sha256 "$TX8_ARCHIVE"
rm -rf "$DEPS_ROOT/tx8_deps"
tar -xzf "$TX8_ARCHIVE" -C "$DEPS_ROOT"

# Wafer lowering 即使在 compiler-only 模式也需要 TX8 指令定义头文件。
if [[ ! -f "$DEPS_ROOT/tx8_deps/include/instr_def.h" ]]; then
    if [[ $COMPILER_ONLY == 1 ]]; then
        echo "ERROR: compiler build still requires TX8_DEPS_ROOT/include/instr_def.h" >&2
        echo "Place a validated TX8 archive beside this script or set TX8_DEPS_TAR_URL" >&2
        exit 1
    fi
    echo "ERROR: TX8 archive did not produce $DEPS_ROOT/tx8_deps/include/instr_def.h" >&2
    exit 1
fi

# 完整模式才安装独家的 TXDA/txops 发布组合。
log "Download and install the unique torch_txda/txops release"
if [[ $COMPILER_ONLY == 0 ]]; then
    # 计算 TXDA 发布归档的本地缓存路径。
    TXDA_ARCHIVE="$PACKAGE_ROOT/python/$(basename "${TORCH_TXDA_TAR_URL%%\?*}")"
    # 下载 TXDA/txops 归档并记录校验和。
    download "$TORCH_TXDA_TAR_URL" "$TXDA_ARCHIVE"
    record_sha256 "$TXDA_ARCHIVE"
    # 下载同一发布组合的 PyTorch wheel，不能随意使用公共版本替代。
    TORCH_WHEEL="$PACKAGE_ROOT/python/$(basename "${TORCH_WHEEL_URL%%\?*}")"
    download "$TORCH_WHEEL_URL" "$TORCH_WHEEL"
    record_sha256 "$TORCH_WHEEL"
    # 清理上一次解压的 pack，防止混入旧版 wheel。
    rm -rf "$PACKAGE_ROOT/python/pack"
    # TXDA 归档预期会解压出 pack 目录。
    tar -xzf "$TXDA_ARCHIVE" -C "$PACKAGE_ROOT/python"
    TXDA_PACK="$PACKAGE_ROOT/python/pack"
    # 查找与当前 Python 3.10 匹配的 txops wheel。
    TXOPS_WHEEL=$(find "$TXDA_PACK" -maxdepth 1 -type f -name 'txops-*-cp310-*.whl' -print -quit)
    # 查找与当前 Python 3.10 匹配的 torch_txda wheel。
    TORCH_TXDA_WHEEL=$(find "$TXDA_PACK" -maxdepth 1 -type f -name 'torch_txda-*-cp310-*.whl' -print -quit)
    if [[ -z "$TXOPS_WHEEL" || -z "$TORCH_TXDA_WHEEL" ]]; then
        echo "ERROR: TXDA archive does not contain matching cp310 txops and torch_txda wheels" >&2
        exit 1
    fi
    # 按 PyTorch、txops、torch_txda 顺序安装同一发布组合。
    $PYTHON -m pip install "$TORCH_WHEEL" "$TXOPS_WHEEL" "$TORCH_TXDA_WHEEL"
    # 用 import 验证三个 Python runtime 包都能加载。
    $PYTHON - <<'PY'
import torch
import torch_txda
import txops
print("torch:", torch.__version__)
print("torch_txda:", torch_txda.__file__)
print("txops:", txops.__file__)
PY
fi

# 设置 Wafer 构建所需的 TX8 头文件路径和 simulator 编译模式。
log "Build and install Wafer from source"
export TX8_DEPS_ROOT="$DEPS_ROOT/tx8_deps"
export WAFER_TX8_INCLUDE_DIR="$TX8_DEPS_ROOT/include"
export DICP_BACKEND=wafer
export USE_SIM_MODE=1
# 生成 wafer_env.sh，并再次检查 LLVM 22 和 instr_def.h。
bash "$REPO_DIR/setup_wafer_env.sh"
# 激活 Wafer 编译环境。
source "$REPO_DIR/wafer_env.sh"
# 删除旧 build_manual 后重新编译 wafer-opt 和 libtriton.so。
bash "$REPO_DIR/compile_wafer.sh" --clean
# 使用刚生成的 C++ 产物构建并安装 Python wheel。
bash "$REPO_DIR/install_wafer.sh" --skip-build

# 打印迁移结束信息。
log "Migration completed"
# 到这里至少完成编译器安装和 TTIR 到 ELF object 验收。
echo "Compiler and TTIR-to-ELF validation passed."
if [[ $COMPILER_ONLY == 1 ]]; then
    # compiler-only 模式明确没有验证硬件 runtime。
    echo "Hardware runtime was intentionally skipped (--compiler-only)."
else
    echo "TXDA/txops imports passed; hardware execution still requires the runtime/linker/launcher gates in the migration SOP."
fi
