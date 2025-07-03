import sys
import os
from pathlib import Path

# 获取当前包的绝对路径（即 WaveRNN_master 的目录）
package_dir = Path(__file__).resolve().parent

# 将该路径加入 sys.path，使其成为模块搜索的根目录
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

# 设置环境变量 PYTHONPATH（可选，增强兼容性）
os.environ["PYTHONPATH"] = str(package_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
