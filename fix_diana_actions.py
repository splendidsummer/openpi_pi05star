#!/usr/bin/env python3
"""
修复Diana数据集的action：用下一时刻的state替换IK计算的action

原理：在模仿学习中，action[t]应该是导致state[t+1]的命令
     最准确的做法是直接用state[t+1]作为action[t]（目标位置）
     训练时DeltaActions transform会自动转换为delta

用法：
    python fix_diana_actions.py --data-dir /path/to/data_lerobot/YYYYMMDD
"""

import pickle
import argparse
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def fix_episode_actions(episode_dir: Path):
    """修复单个episode的actions"""
    pkl_path = episode_dir / "episode_data.pkl"
    
    if not pkl_path.exists():
        logging.warning(f"跳过 {episode_dir.name}: 未找到episode_data.pkl")
        return False
    
    # 加载数据
    with open(pkl_path, 'rb') as f:
        episode_data = pickle.load(f)
    
    frames = episode_data["frames"]
    num_frames = len(frames)
    
    if num_frames < 2:
        logging.warning(f"跳过 {episode_dir.name}: 帧数不足 ({num_frames})")
        return False
    
    # 修复actions：action[t] = state[t+1]
    fixed_count = 0
    for i in range(num_frames - 1):
        # 取下一帧的state作为当前帧的action（目标位置）
        frames[i]["action"] = frames[i + 1]["observation"]["state"].copy()
        fixed_count += 1
    
    # 最后一帧：使用当前state（保持不动）
    frames[-1]["action"] = frames[-1]["observation"]["state"].copy()
    
    # 保存修复后的数据
    backup_path = pkl_path.with_suffix('.pkl.bak')
    pkl_path.rename(backup_path)  # 备份原文件
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(episode_data, f)
    
    logging.info(f"✓ {episode_dir.name}: 修复了 {fixed_count+1} 帧 (备份: {backup_path.name})")
    return True


def main(data_dir: str, episode_id: int = None, restore_backup: bool = False):
    """
    修复Diana数据集的actions
    
    Args:
        data_dir: 数据目录（如 data_lerobot/20260105）
        episode_id: 只修复指定episode（可选）
        restore_backup: 从备份恢复（撤销修复）
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logging.error(f"错误：数据目录不存在: {data_dir}")
        return
    
    # 获取所有episode目录
    if episode_id is not None:
        episode_dirs = [data_path / f"episode_{episode_id:04d}"]
    else:
        episode_dirs = sorted([d for d in data_path.iterdir() 
                              if d.is_dir() and d.name.startswith("episode_")])
    
    if len(episode_dirs) == 0:
        logging.error(f"错误：未找到任何episode目录")
        return
    
    logging.info(f"找到 {len(episode_dirs)} 个episodes")
    logging.info(f"数据目录: {data_path}")
    
    if restore_backup:
        # 恢复备份
        logging.info("\n开始恢复备份...")
        restored = 0
        for episode_dir in episode_dirs:
            pkl_path = episode_dir / "episode_data.pkl"
            backup_path = episode_dir / "episode_data.pkl.bak"
            
            if backup_path.exists():
                pkl_path.unlink(missing_ok=True)
                backup_path.rename(pkl_path)
                logging.info(f"✓ {episode_dir.name}: 已恢复备份")
                restored += 1
        
        logging.info(f"\n完成！恢复了 {restored} 个episodes")
    else:
        # 修复actions
        logging.info("\n开始修复actions...")
        logging.info("策略: action[t] = state[t+1] (下一时刻的真实状态)")
        logging.info("")
        
        confirm = input(f"确认修复 {len(episode_dirs)} 个episodes？(原文件会备份为.pkl.bak) [y/N]: ")
        if confirm.lower() != 'y':
            logging.info("取消操作")
            return
        
        fixed = 0
        for episode_dir in episode_dirs:
            if fix_episode_actions(episode_dir):
                fixed += 1
        
        logging.info(f"\n完成！成功修复 {fixed}/{len(episode_dirs)} 个episodes")
        logging.info("\n提示:")
        logging.info("  - 原文件已备份为 episode_data.pkl.bak")
        logging.info("  - 如需恢复，运行: python fix_diana_actions.py --data-dir {} --restore-backup".format(data_dir))
        logging.info("  - 确认无误后，可删除备份文件释放空间")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="修复Diana数据集的actions（用state[t+1]替换IK输出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 修复整个日期目录下的所有episodes
  python fix_diana_actions.py --data-dir data_lerobot/20260105
  
  # 只修复指定episode
  python fix_diana_actions.py --data-dir data_lerobot/20260105 --episode-id 3
  
  # 恢复备份（撤销修复）
  python fix_diana_actions.py --data-dir data_lerobot/20260105 --restore-backup
        """
    )
    parser.add_argument("--data-dir", type=str, required=True,
                       help="数据目录路径（如 data_lerobot/20260105）")
    parser.add_argument("--episode-id", type=int, default=None,
                       help="只修复指定的episode ID（可选）")
    parser.add_argument("--restore-backup", action="store_true",
                       help="从备份恢复（撤销修复）")
    
    args = parser.parse_args()
    main(args.data_dir, args.episode_id, args.restore_backup)
