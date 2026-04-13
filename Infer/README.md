# Infer

当前目录使用单文件方案：
- [EggInfer.py](/D:/gpt-codex/ovo/Infer/EggInfer.py:1) 是唯一的核心实现，负责推理和模型加密
- [tools.py](/D:/gpt-codex/ovo/Infer/tools.py:1)、[model_crypto.py](/D:/gpt-codex/ovo/Infer/model_crypto.py:1)、[encrypt_models.py](/D:/gpt-codex/ovo/Infer/encrypt_models.py:1) 是兼容层
- [run_infer_test.py](/D:/gpt-codex/ovo/Infer/run_infer_test.py:1) 用于本地测试
- [build_protected_infer.py](/D:/gpt-codex/ovo/Infer/build_protected_infer.py:1) 用于手动构建保护版
- [publish_release.py](/D:/gpt-codex/ovo/Infer/publish_release.py:1) 用于自动发布
- [verify_release.py](/D:/gpt-codex/ovo/Infer/verify_release.py:1) 用于验证发布目录和源码结果是否一致

## 目录说明

- [model](/D:/gpt-codex/ovo/Infer/model) 存放明文模型
- `release_pyarmor_singlefile_YYYYMMDD_HHMMSS` 是自动发布后的目录
- 发布目录内：
  - `code/EggInfer.py` 是受保护入口
  - `code/run_infer_test.py` 保持明文，便于调试
  - `models/*.pt.enc` 是加密模型

## 运行环境

- Python：`C:\Users\liu\Anaconda3\envs\py312\python.exe`
- 依赖：`torch`、`cv2`、`scipy`、`osgeo/gdal`
- 保护工具：`pyarmor`

## 常用命令

源码版测试：

```powershell
& "C:\Users\liu\Anaconda3\envs\py312\python.exe" "D:\gpt-codex\ovo\Infer\run_infer_test.py" --case 7x5 --model-dir "D:\gpt-codex\ovo\Infer\model\0413"
```

手动加密模型：

```powershell
& "C:\Users\liu\Anaconda3\envs\py312\python.exe" "D:\gpt-codex\ovo\Infer\EggInfer.py" encrypt-models --input-dir "D:\gpt-codex\ovo\Infer\model\0413" --output-dir "D:\gpt-codex\ovo\Infer\your_models_enc" --key "your-model-key" --overwrite
```

自动发布：

```powershell
& "C:\Users\liu\Anaconda3\envs\py312\python.exe" "D:\gpt-codex\ovo\Infer\publish_release.py"
```

说明：
- 自动发布会生成带时间的发布目录
- 自动发布固定使用密钥 `WL-20260226`
- 自动发布默认取 [model\0413](/D:/gpt-codex/ovo/Infer/model/0413) 作为明文模型目录
- 发布目录内置这个 key，正常使用 bundled models 时不需要额外传 `--model-key`
- 如果你换成了其他 key 加密的模型，再显式传 `--model-key` 或设置 `EGG_INFER_MODEL_KEY`

验证最新发布目录：

```powershell
& "C:\Users\liu\Anaconda3\envs\py312\python.exe" "D:\gpt-codex\ovo\Infer\verify_release.py"
```

指定发布目录验证：

```powershell
& "C:\Users\liu\Anaconda3\envs\py312\python.exe" "D:\gpt-codex\ovo\Infer\verify_release.py" --release-dir "D:\gpt-codex\ovo\Infer\release_pyarmor_singlefile_20260413_201258"
```

## 当前默认结果

在默认样本和 `7x5` 坐标下，当前 `0413` 模型的预期结果是：
- `female=21`
- `male=14`

## 备注

- 运行时模型保持外置，替换模型不需要重新编译代码
- 发布目录中的测试脚本不混淆，这是有意保留的
- 如果需要发布新版本，优先使用 [publish_release.py](/D:/gpt-codex/ovo/Infer/publish_release.py:1)，不要手工拼目录
