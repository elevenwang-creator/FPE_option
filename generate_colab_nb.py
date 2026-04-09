import json
import os

notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "intro_title"
      },
      "source": [
        "# 🚀 FPE Option Pricing Engine - Colab NVPTX Benchmark\n",
        "\n",
        "This notebook acts as an orchestration entry point to test the Grid-Block parameterized GPU batch pricing benchmark natively on Nvidia accelerators via `Pixi` & `Mojo Nightly`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "install_pixi"
      },
      "outputs": [],
      "source": [
        "!echo \"Downloading and Installing Pixi...\"\n",
        "!curl -fsSL https://pixi.sh/install.sh | bash\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mount_drive_md"
      },
      "source": [
        "### 📦 Mount Your Source Project\n",
        "\n",
        "Upload or mount your `FPE_option` project folder here. If you zipped it and uploaded it via the sidebar, unzip it FIRST."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "cd_path",
        "colab": {}
      },
      "outputs": [],
      "source": [
        "import os\n",
        "\n",
        "# MODIFY this path if you unpacked the FPE_option folder to a different path\n",
        "# e.g. /content/drive/MyDrive/FPE_option\n",
        "PROJECT_DIR = \"/content/FPE_option\"\n",
        "\n",
        "if not os.path.exists(PROJECT_DIR):\n",
        "    print(f\"⚠️ Cannot find {PROJECT_DIR}. Please make sure you uploaded the FPE_option folder to Colab!\")\n",
        "    print(\"HINT: Zip your FPE_option folder on Mac, upload it to Colab, and run `!unzip FPE_option.zip`\")\n",
        "else:\n",
        "    os.chdir(PROJECT_DIR)\n",
        "    print(\"Successfully entered project directory:\", os.getcwd())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "run_benchmark"
      },
      "source": [
        "### ⚡️ Run FPE GPU Multiple Allocations Benchmark"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "mojo_run"
      },
      "outputs": [],
      "source": [
        "%%bash\n",
        "export PATH=\"/root/.pixi/bin:$PATH\"\n",
        "# Install the environment natively on Linux-64 according to pixi.toml\n",
        "pixi install\n",
        "\n",
        "# Run the benchmark logic testing multi-batch block/grid logic!\n",
        "pixi run mojo run -I src benchmarks/bench_gpu_batch_pricing.mojo\n"
      ]
    }
  ]
}

os.makedirs('benchmarks', exist_ok=True)
with open('benchmarks/Colab_FPE_Benchmark.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
