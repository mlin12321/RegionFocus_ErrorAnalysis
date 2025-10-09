# Script Improvements Summary

## Changes Made to `run_qwen3_30b_slurm.sh`

### 1. **Extended Time Limits**
- SLURM job time: `00:10:00` → `00:30:00` (30 minutes)
- vLLM server timeout: `300s` → `1200s` (20 minutes)
- VLLM_RAY_TIMEOUT: `300s` → `600s` (10 minutes)

**Reason**: Loading a 30B parameter model across 4 GPUs requires significant time (typically 5-15 minutes).

### 2. **Robust File Copying with Exit Trap**
Implemented a `cleanup_and_copy()` function that:
- Automatically executes on script exit (success OR failure)
- Copies all output files to the save folder
- Provides detailed logging of what's being copied
- Handles missing files gracefully

**Files copied:**
- `vllm_server.log` - vLLM server startup and runtime logs
- `eval_output.log` - Full evaluation progress and output
- `results_qwen3vl_30b/*.json` - Final evaluation results
- `results_mid/*.json` - Checkpoint files (saved to subdirectory)

**Key benefit**: Files are copied even if the script crashes or times out.

### 3. **Evaluation Output Logging**
```bash
bash scripts/run_ss_pro_qwen3vl_RegionFocus_30b.sh 2>&1 | tee eval_output.log
```

This captures:
- All progress output from the evaluation script
- Both stdout and stderr
- Real-time display (via `tee`) while also saving to log
- Complete Python traceback if errors occur

### 4. **Automatic vLLM Server Cleanup**
The cleanup function now:
- Kills the vLLM server process on exit
- Removes the PID file
- Prevents zombie processes

### 5. **Directory Path Consistency**
- Fixed output directory: `results_qwen25vl_72b` → `results_qwen3vl_30b`
- Matches the model name being used (Qwen3-VL-30B)

## Changes Made to `run_ss_pro_qwen3vl_RegionFocus_30b.sh`

### 1. **Auto-create Output Directories**
```bash
mkdir -p ./results_qwen3vl_30b
mkdir -p ./results_mid
```

Ensures directories exist before writing.

### 2. **Enhanced Progress Logging**
Added clear section markers:
```
=====================================
Starting evaluation for model: qwen3vl_RegionFocus_30b
=====================================
```

Makes it easier to parse the log and track progress.

### 3. **Corrected Output Path**
Updated `--log_path` to use `results_qwen3vl_30b` instead of `results_qwen25vl_72b`.

## Expected Output Structure

After running the script, `results_qwen3_30b/` will contain:

```
results_qwen3_30b/
├── vllm_server.log                        # vLLM startup and runtime logs
├── eval_output.log                        # Complete evaluation output
├── qwen3vl_RegionFocus_30b.json          # Final results
└── results_mid/
    └── qwen3vl_RegionFocus_30b.json      # Checkpoint file
```

## Key Improvements

1. ✅ **Reliable file copying** - Works even if script fails
2. ✅ **Complete logging** - All progress captured in eval_output.log
3. ✅ **Proper cleanup** - vLLM server always terminated
4. ✅ **Extended timeouts** - Sufficient time for 30B model loading
5. ✅ **Better error messages** - Clear warnings for missing files
6. ✅ **Consistent naming** - All paths match qwen3_30b

## How to Run

```bash
cd /fs/ess/PAS1576/boyu_gou/gui_pretrain/RegionFocus_EA
sbatch scripts/run_qwen3_30b_slurm.sh
```

## Monitoring Progress

Check the SLURM output file:
```bash
tail -f <JOBID>_qwen3vl_rf.slurm.out
```

This will show:
- vLLM server startup progress
- Evaluation progress (real-time)
- File copying status
- Any errors or warnings

