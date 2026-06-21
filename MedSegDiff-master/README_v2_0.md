# v2.0 — Dynamic Relation-to-Field Constraint Router

v2.0 在 **不推翻 v1.5** 的前提下，把 v1.5 的“固定 penalty 场 + 固定 risk
mask 注入”升级为：

> 可学习约束场编译器（Learnable Constraint-Field Compiler）
> ＋ 动态时间/风险感知约束路由器（Dynamic Time/Risk-aware Constraint Router）
> ＋ 可选形态保持约束目标（Constraint-aware Morphology Loss）。

核心范式：**符号/关系约束 → 可学习连续条件场 → 动态风险感知约束生成**。
模型先把多通道 penalty basis 编译成连续约束场，再根据空间风险、生成
时间步和当前特征状态，自适应决定在哪里、何时、以多强程度注入约束。

> 本版本所有文件均为新增（`*_v2_0.*`），**未修改任何 v1.5 文件**。

---

## 1. 注入公式的升级

```
v1.5:  h_x = h_x + W_risk          * ZeroConv(penalty_feat)      # W_risk 固定
v2.0:  h_x = h_x + router_weight_l * ZeroConv(compiled_feat_l)   # 可学习/动态
```

数据流：

```
penalty_basis [S_target, A_oar, B_boundary, P_fused]
  └─> ConstraintFieldCompiler3D (+ RelationEncoder / FiLM)
        └─> compiled field [C_pos, C_neg, C_boundary, C_fused]
              └─> PenaltyAdapter3D (v1.4/v1.5 复用) -> 多尺度特征
                    └─> ZeroConv (1x1x1, 0 初始化, 复用)
                          └─> DynamicConstraintRouter3D 调制 (逐层/逐体素/随 t)
                                └─> 注入 v1.2 主干 X 流 (encoder + bottleneck)
```

---

## 2. 新增模块

| 文件 | 类 | 作用 |
|------|----|------|
| `guided_diffusion/constraint_field_compiler_v2_0.py` | `ConstraintFieldCompiler3D` | 把 penalty basis 编译为可学习约束场，输出 `[C_pos, C_neg, C_boundary, C_fused]`；空间自适应权重 `alpha(x),beta(x),gamma(x)` 取代 v1.5 固定公式 `P=αS-βA+γB` |
| 同上 | `RelationEncoder` | 从 DIS mask + penalty 统计派生 global relation embedding，FiLM 调制 compiler（无 GNN，无 GT 泄漏） |
| `guided_diffusion/dynamic_constraint_router_v2_0.py` | `DynamicConstraintRouter3D` | 以 `h_x / compiled adapter feat / M_risk_base / t_emb` 预测逐体素注入权重，支持 time-dependent；零初始化使 `delta=0`，初始等价 v1.5 |
| `guided_diffusion/unet_3d_v2_0.py` | `UNetModel_DynamicConstraintRouter_v2_0` | 继承 v1.5，整合 compiler + router + 复用的 adapter/ZeroConv；ablation 开关；checkpoint 兼容 |
| `flow_matching_v2_0.py` | `FlowMatchingV20` | 主损失同 v1.5；新增 router 正则 + 可选形态损失（risk-region 梯度一致性 / 高频保持），含稳定 clean-dose 估计 |
| `guided_diffusion/dose_loader_3d_v2_0.py` | `Dataset_PSDM_3D_Train_v2_0` | 复用 v1.5 数据格式 `(ct, syn, dis, penalty, dose)`；优先使用真实 `P_fused.nii.gz` |

---

## 3. 基础风险图（保留为 Router 先验）

v1.5 的 `M_risk_base = f(A_oar, abs(B_boundary), abs(P_fused))` 完整保留，
但在 v2.0 中**只作为 Router 的先验输入**，不再是最终注入权重：

```
W_floor       = risk_base + (1 - risk_base) * M_risk_base
router_weight = clamp(W_floor + tanh(conv(...)) * delta_scale, risk_base, 1)
```

`risk_base` / `risk_use_channels` 配置保留。Router 最终决定注入强度，
但初始（delta=0）严格回到 v1.5 行为。

---

## 4. ZeroConv 安全注入 + Checkpoint 兼容

- ZeroConv 仍为权重&bias 全 0 初始化 → 初始 residual = 0。
- Router 最终 1×1×1 conv 零初始化 → 初始 `delta = 0` → `router_weight = W_floor`。
- 二者叠加：**v2.0 初始化时与 v1.5 / v1.2 主干行为逐位一致**。
- 从 v1.5 checkpoint `strict=False` 加载：主干 / PenaltyAdapter / ZeroConv
  权重复用（默认 `compiler_out_channels == penalty_channels == 4`，adapter
  输入通道不变也复用）；compiler / relation encoder / routers 从头初始化。

---

## 5. Ablation 模式（`--mode`）

| mode | compiler | router | 说明 |
|------|----------|--------|------|
| `v1_5` | ✗ | ✗ | 固定 W_risk（等价 v1.5） |
| `compiler_only` | ✓ | ✗ | 可学习 compiled field + 固定 W_risk |
| `router_only` | ✗ | ✓ | 原始 basis + 动态 router |
| `full_v2_0` | ✓ | ✓ | 全启用（默认） |

其它开关：`--disable_relation_embedding`、`--disable_morphology_loss`、
`--router_time_independent`、`--disable_risk_aware_injection`。

---

## 6. 可选形态损失与 Router 正则（默认关闭 / 极小）

| 参数 | 默认 | 含义 |
|------|------|------|
| `--lambda_grad_loss` | 0.0 | 风险区域梯度一致性 `M_risk*|grad(pred_clean)-grad(gt)|` |
| `--lambda_hf_loss` | 0.0 | 高频结构保持 `x - avgpool(x)` 的 L1（缓解过平滑） |
| `--router_reg_weight` | 0.0 | 平均注入惩罚（防止 router 全图打开退化为全局注入） |
| `--router_smooth_weight` | 0.0 | router 权重 total-variation（平滑正则） |

`pred_clean ≈ x_t + (1-t) * v_pred`（OT 路径在 σ_min→0 下精确），用于稳定
恢复 clean dose。建议实验时设 0.01 量级，默认 0 保证训练初期稳定。

---

## 7. P_fused 语义（重要）

`P_fused` 当前复用 `dose_synthetic.nii.gz`，在 v2.0 中明确为
**coarse / synthetic prior**，**不是真实 dose**。Compiler 仅将其作为
`C_fused` 的 coarse-prior 项；**任何 penalty / compiled field 都不使用
ground-truth dose**（无 label leakage）。若存在真实 `P_fused.nii.gz`，
数据集会自动优先采用。SYN 流与 penalty 的 P_fused 通道可并存，其冗余性
请用 `router_only` / `disable_relation_embedding` ablation 验证。

---

## 8. 训练 / 推理

```bash
# 训练 (full_v2_0)，可选从 v1.5 续训
bash scripts/train_v2_0_local_mednext.sh \
    --resume_from trained_models/v1_5_risk_penalty_adapter/<run>/model_best_mae.pth \
    --resume_ema  trained_models/v1_5_risk_penalty_adapter/<run>/ema_best_mae.pth

# ablation 例
bash scripts/train_v2_0_local_mednext.sh --mode router_only
bash scripts/train_v2_0_local_mednext.sh --mode compiler_only --disable_relation_embedding

# 推理 + 保存可视化 (M_risk_base / compiled field / 各层 router 权重)
bash scripts/run_test_v2_0.sh \
    trained_models/v2_0_dynamic_router/<run>/ema_best_mae.pth --save_aux
```

---

## 9. 日志与可视化

- 训练日志：weighted MAE、risk-region MAE，以及损失分量（base / router_mean /
  router_tv / grad / hf）。
- 推理 `--save_aux`：对前 `--aux_num` 个病例在中心 patch（`--aux_t`）保存
  `m_risk_base.nii.gz`、`compiled_field_c*.nii.gz`、`router_weight_l*.nii.gz`，
  用于验证 router 是否真的在 OAR / boundary / conflict 区域更强使用先验，
  并减少普通区域的过约束。

v2.0 的目标不是仅追求 Dose Score 小幅提升，而是证明动态约束路由能改善
OAR / boundary / high-gradient 区域，同时降低普通区域的过约束。
