# 测试最新权重文件 (exp_baseline_20251030_112631)
# 使用不同的follower数量进行测试

python scripts/baseline/test.py --leader_model_path runs/exp_baseline_20251030_112631/leader.pth --follower_model_path runs/exp_baseline_20251030_112631/follower.pth --n_follower 1 --test_episode 10
python scripts/baseline/test.py --leader_model_path runs/exp_baseline_20251030_112631/leader.pth --follower_model_path runs/exp_baseline_20251030_112631/follower.pth --n_follower 2 --test_episode 10
python scripts/baseline/test.py --leader_model_path runs/exp_baseline_20251030_112631/leader.pth --follower_model_path runs/exp_baseline_20251030_112631/follower.pth --n_follower 3 --test_episode 10
python scripts/baseline/test.py --leader_model_path runs/exp_baseline_20251030_112631/leader.pth --follower_model_path runs/exp_baseline_20251030_112631/follower.pth --n_follower 4 --test_episode 10
