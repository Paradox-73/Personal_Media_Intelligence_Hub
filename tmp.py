import json
res = json.load(open('reports/distillation_ablation_results.json'))
table = '| Domain | MAE (No) | Skill (No) | MAE (With) | Skill (With) | p-value | Verdict |\n|:---|:---|:---|:---|:---|:---|:---|\n'
for r in res:
    table += f"| {r['Domain'].capitalize()} | {r['MAE_NoPrior']:.3f} | {r['Skill_NoPrior']:.3f} | {r['MAE_WithPrior']:.3f} | {r['Skill_WithPrior']:.3f} | {r['p_value']:.3f} | **{r['Verdict']}** |\n"
with open('reports/FINAL_ML_TECHNICAL_REPORT.md', 'a') as f:
    f.write('\n\n### Distillation Prior Ablation\n\n' + table + '\n\n*Note: The unified model prior is not significantly helpful in either domain (p >= 0.05). It has been dropped.*\n')
