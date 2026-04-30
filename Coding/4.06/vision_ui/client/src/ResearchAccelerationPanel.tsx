import { PanelTitle } from "./uiPrimitives";
import { devLaneCategoryLabel } from "./displayHelpers";
import type { DevelopmentLaneItem } from "./types";

interface ResearchAccelerationPanelProps {
  items: DevelopmentLaneItem[];
}

export function ResearchAccelerationPanel({ items }: ResearchAccelerationPanelProps) {
  const researchItems = items.filter(item => 
    item.category === "paper_reproduction" || 
    item.category === "prototype" || 
    item.category === "evaluation"
  );

  return (
    <article className="panel research-acceleration-panel">
      <PanelTitle eyebrow="Acceleration" title="研究与原型加速" />
      <div className="research-acceleration-info">
        <p className="status-honest-note">
          <strong>状态提示：</strong>当前研发车道处于“设计就绪”状态。正在定义输入/输出契约，以支持从论文复现到 KYKT 核心流的自动合入。
        </p>
      </div>
      <div className="research-lane-grid">
        {researchItems.map((item) => (
          <div key={item.id} className="research-lane-card">
            <div className="research-lane-head">
              <span className="mini-label">{devLaneCategoryLabel(item.category)}</span>
              <strong>{item.title}</strong>
            </div>
            <div className="research-lane-next">
              <span className="mini-label">Next Action</span>
              <p>{item.nextAction}</p>
            </div>
            <div className="research-lane-target">
              <span className="mini-label">合入目标</span>
              <code>{item.mergeTarget ?? "research_context"}</code>
            </div>
          </div>
        ))}
      </div>
      <div className="seed-categories-strip">
        <span className="mini-label">快速启动种子</span>
        <div className="seed-pills">
          <button className="ghost-button small">论文复现</button>
          <button className="ghost-button small">新 3R 模型 Runner</button>
          <button className="ghost-button small">UI/评测流原型</button>
          <button className="ghost-button small">研究报告/实验设计</button>
        </div>
      </div>
    </article>
  );
}
