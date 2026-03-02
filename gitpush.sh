#!/bin/bash

# comment="1125 效果有好转 v1.6=2% v2.6=1.03% ibm-guided=1.97% ibm-macro=-4.62% ispd-macro=-3.03%"

# comment="1203 v2.6 DDPO  timesteps=50 batch=1  actual_batch=2  num_batch_per_epoch=16 inner_epochs=3 num_epochs=60 lr=1e-6  eta=0.3 v2.61优化率0.47%"

comment="260302 2602281113-best 不考虑合法化 使用中间奖励"

echo "----------------------------------------"
echo "目前提交的 comment 如下，提醒是否需要更新："
echo "$comment"
echo "----------------------------------------"
read -p "确认提交吗？(y/n): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "❌ 已取消提交。"
    exit 0
fi

git add -A .
git commit -m "$comment"
git push
