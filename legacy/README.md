# legacy/ — 历史层（只读归档，不再维护）

| 目录 | 时期 | 说明 |
|---|---|---|
| 根目录（docs/ notebooks/ outputs/ scripts/ 等） | 前身项目 | Qwen3-4B 文本轨迹预测实验（由此转向本项目，见根 README「仓库沿革」） |
| `phase_v_synth/` | Phase V 早期（PhysicalAI-AV 授权前） | 合成数据时代的评测脚手架：CVM 基线 / ADE·FDE 指标 / action→轨迹积分 / BEV 可视化 / 合成测试集生成 + 占位产物。取得数据集授权、R1 全链跑通后被 `eval/r1/` 真实管线取代——积分改官方 `compute_minade`，可视化改 dgural 真实帧叠加（`eval/r1/dgural_demo/`） |

归档原则：只进不出（历史可查、不再演进）；现役资产见根 README「仓库布局」。
