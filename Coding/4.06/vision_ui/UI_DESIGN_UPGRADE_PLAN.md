# KYKT Vision UI 设计升级方案

> 目标：从"玩具感"升级为"专业工具软件"

---

## 一、当前问题诊断

### 1.1 视觉元素问题

| 问题 | 当前状态 | 专业软件参考 |
|------|----------|--------------|
| **图标** | Emoji (📋 ➕ 🔬) | SVG 图标 (Lucide/Heroicons) |
| **圆角** | 16-20px 大圆角 | 6-8px 克制圆角 |
| **配色** | 高饱和蓝色渐变 | 中性灰 + 单色强调 |
| **阴影** | 明显的投影 | 微妙或无阴影 |
| **字体** | 偏大、松散 | 紧凑、层次分明 |

### 1.2 布局问题

| 问题 | 当前状态 | 专业软件参考 |
|------|----------|--------------|
| **信息密度** | 低，空白多 | 高，内容紧凑 |
| **侧边栏** | 宽松 padding | 紧凑，图标+文字对齐 |
| **面板** | 卡片套卡片 | 扁平化层级 |
| **间距** | 随意变化 | 严格 4/8px 栅格 |

### 1.3 交互问题

| 问题 | 当前状态 | 专业软件参考 |
|------|----------|--------------|
| **状态反馈** | 颜色块 badge | 细线 + 微妙色彩 |
| **按钮** | 圆润、渐变 | 扁平、边框清晰 |
| **hover** | 背景色变化 | 微妙边框/下划线 |

---

## 二、设计方向对比

### 方案 A：Linear 风格 (深色专业)

```
特点：
- 深色背景 (#0D0D0D ~ #1A1A1A)
- 高对比度文字 (#FFFFFF / #A1A1A1)
- 紫/蓝强调色用于关键操作
- 极简图标，线条统一
- 紧凑行高，密集信息

适用场景：
- 面向开发者/专业用户
- 长时间使用，减少视觉疲劳
- 数据密集型界面
```

### 方案 B：Figma 风格 (浅色专业)

```
特点：
- 浅灰背景 (#F5F5F5 ~ #FFFFFF)
- 中性文字 (#1E1E1E / #6B6B6B)
- 最小化色彩使用，蓝色仅用于交互
- 清晰的边界线分隔区域
- 工具栏紧凑，内容区宽松

适用场景：
- 设计/创意工具
- 需要大量预览内容
- 白天使用为主
```

### 方案 C：VS Code 风格 (开发者工具)

```
特点：
- 深灰背景 (#1E1E1E ~ #252526)
- 紧凑侧边栏（可折叠为图标）
- 标签页/面板系统
- 状态栏底部信息条
- 高信息密度，12-13px 字体

适用场景：
- 面向工程师
- 多任务并行
- 需要键盘快捷键支持
```

---

## 三、推荐方案：混合方案 (Figma 基底 + Linear 细节)

### 3.1 设计 Token 定义

```css
:root {
  /* ═══════════════════════════════════════════
     色彩系统 - 专业中性调
     ═══════════════════════════════════════════ */
  
  /* 背景层级 */
  --bg-app: #F7F8FA;           /* 最底层背景 */
  --bg-surface: #FFFFFF;        /* 卡片/面板 */
  --bg-elevated: #FFFFFF;       /* 弹层/模态 */
  --bg-subtle: #F3F4F6;         /* 次级区域 */
  --bg-muted: #E5E7EB;          /* 禁用/占位 */
  
  /* 文字层级 */
  --text-primary: #111827;      /* 主文字 */
  --text-secondary: #4B5563;    /* 次要文字 */
  --text-tertiary: #9CA3AF;     /* 辅助文字 */
  --text-disabled: #D1D5DB;     /* 禁用文字 */
  
  /* 边框 */
  --border-default: #E5E7EB;    /* 默认边框 */
  --border-subtle: #F3F4F6;     /* 微弱边框 */
  --border-strong: #D1D5DB;     /* 强调边框 */
  
  /* 强调色 - 克制使用 */
  --accent-primary: #2563EB;    /* 主操作 */
  --accent-success: #059669;    /* 成功 */
  --accent-warning: #D97706;    /* 警告 */
  --accent-danger: #DC2626;     /* 危险 */
  
  /* ═══════════════════════════════════════════
     尺寸系统 - 严格 4px 栅格
     ═══════════════════════════════════════════ */
  
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  
  /* 圆角 - 克制 */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  
  /* 字体 */
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", Consolas, monospace;
  
  --text-xs: 11px;
  --text-sm: 12px;
  --text-base: 13px;
  --text-lg: 14px;
  --text-xl: 16px;
  
  /* 阴影 - 极微妙 */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-md: 0 2px 4px rgba(0, 0, 0, 0.06);
}
```

### 3.2 组件改造示例

#### Sidebar 改造

**Before (当前)**
```tsx
// 使用 emoji，padding 过大
<button className="nav-item">
  <span>📋</span>
  工作队列
</button>
```

**After (专业化)**
```tsx
// 使用 SVG 图标，紧凑布局
import { ListTodo, Plus, FlaskConical, Grid3X3, Rocket, Settings } from "lucide-react";

<button className="nav-item">
  <ListTodo size={16} strokeWidth={1.5} />
  <span>工作队列</span>
  {badge > 0 && <span className="nav-badge">{badge}</span>}
</button>
```

```css
.nav-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 450;
  color: var(--text-secondary);
  transition: background 0.1s, color 0.1s;
}

.nav-item:hover {
  background: var(--bg-subtle);
  color: var(--text-primary);
}

.nav-item.active {
  background: var(--bg-subtle);
  color: var(--accent-primary);
}

.nav-badge {
  margin-left: auto;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: 9px;
  background: var(--accent-primary);
  color: white;
  font-size: 11px;
  font-weight: 500;
  line-height: 18px;
  text-align: center;
}
```

#### 状态 Badge 改造

**Before**
```css
.status-badge {
  padding: 6px 12px;
  border-radius: 999px;
  background: var(--blue-soft);
  font-size: 12px;
}
```

**After**
```css
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.01em;
}

.status-badge::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
}

.status-badge.running {
  background: #EFF6FF;
  color: #1D4ED8;
}
.status-badge.running::before {
  background: #3B82F6;
  animation: pulse 1.5s infinite;
}

.status-badge.finished {
  background: #ECFDF5;
  color: #047857;
}
.status-badge.finished::before {
  background: #10B981;
}

.status-badge.failed {
  background: #FEF2F2;
  color: #B91C1C;
}
.status-badge.failed::before {
  background: #EF4444;
}
```

#### 按钮改造

**Before**
```css
.primary-button {
  padding: 10px 20px;
  border-radius: 12px;
  background: linear-gradient(180deg, #5e8cf4 0%, #3f74e6 100%);
  font-size: 14px;
}
```

**After**
```css
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 32px;
  padding: 0 12px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.1s;
}

.btn-primary {
  background: var(--accent-primary);
  color: white;
  border: none;
}
.btn-primary:hover {
  background: #1D4ED8;
}

.btn-secondary {
  background: white;
  color: var(--text-primary);
  border: 1px solid var(--border-default);
}
.btn-secondary:hover {
  background: var(--bg-subtle);
  border-color: var(--border-strong);
}

.btn-ghost {
  background: transparent;
  color: var(--text-secondary);
  border: none;
}
.btn-ghost:hover {
  background: var(--bg-subtle);
  color: var(--text-primary);
}
```

### 3.3 布局改造

#### 当前 Sidebar 宽度
```css
.workspace-sidebar { width: 240px; }  /* 太宽 */
```

#### 改造后
```css
.workspace-sidebar { 
  width: 200px;        /* 收窄 */
  border-right: 1px solid var(--border-default);
  background: var(--bg-surface);
}
```

#### 面板间距

**Before**
```css
.panel { padding: 18px; border-radius: 18px; }
```

**After**
```css
.panel { 
  padding: 16px; 
  border-radius: 8px;
  border: 1px solid var(--border-default);
  background: var(--bg-surface);
  box-shadow: var(--shadow-sm);
}
```

---

## 四、具体改造步骤

### Phase 1: 基础 Token 替换 (30 min)
1. 更新 `:root` CSS 变量
2. 统一圆角为 4/6/8px
3. 移除渐变背景
4. 调整阴影为微妙版本

### Phase 2: 图标系统 (20 min)
1. 安装 `lucide-react`
2. 替换 Sidebar emoji
3. 替换按钮/状态图标

### Phase 3: 组件精修 (40 min)
1. Sidebar 紧凑化
2. Status Badge 重设计
3. 按钮系统统一
4. 表格/列表行高收紧

### Phase 4: 布局调整 (30 min)
1. Sidebar 收窄
2. 面板间距统一
3. 命令栏高度调整
4. 整体信息密度提升

---

## 五、视觉对比预览

### Before (当前)
```
┌─────────────────────────────────────────────────────┐
│ [K] KYKT Vision                    [刷新] [+ 新建]  │
├──────────────┬──────────────────────────────────────┤
│              │                                      │
│  📋 工作队列  │   ┌─────────────────────────────┐   │
│              │   │                             │   │
│  ➕ 新建任务  │   │      大面积留白             │   │
│              │   │                             │   │
│  🔬 对比面板  │   │   [  圆润的大按钮  ]        │   │
│              │   │                             │   │
│  📊 样例矩阵  │   └─────────────────────────────┘   │
│              │                                      │
│  🚀 研发加速  │                                      │
│              │                                      │
│  ⚙️ 系统配置  │                                      │
│              │                                      │
└──────────────┴──────────────────────────────────────┘
```

### After (专业化)
```
┌────────────────────────────────────────────────────┐
│ K  KYKT Vision              [↻] [Cmd+N 新建任务]   │
├───────────┬────────────────────────────────────────┤
│ ☰ 队列  3 │ 任务队列                               │
│ + 新建    │ ┌────────┬────────┬────────┬────────┐ │
│ ⚗ 对比    │ │ ID     │ 模型   │ 状态   │ 时间   │ │
│ ⊞ 矩阵    │ ├────────┼────────┼────────┼────────┤ │
│ ◇ 研发    │ │ abc123 │ MASt3R │ ● 运行 │ 2m ago │ │
│ ⚙ 系统    │ │ def456 │ Fast3R │ ● 完成 │ 5m ago │ │
│           │ │ ghi789 │ DUSt3R │ ● 失败 │ 8m ago │ │
│           │ └────────┴────────┴────────┴────────┘ │
│           │                                        │
│───────────│ 快速操作                               │
│ v0.3.0    │ [派发选中] [重试失败] [导出报告]       │
└───────────┴────────────────────────────────────────┘
```

---

## 六、实施确认

以上方案是否符合预期？确认后我将按 Phase 1-4 顺序逐步实施。

**预计改动文件：**
- `client/src/styles.css` - 核心样式
- `client/src/Sidebar.tsx` - 侧边栏组件
- `client/src/uiPrimitives.tsx` - 基础组件
- `client/src/App.tsx` - 主布局
- `package.json` - 添加 lucide-react

**预计耗时：** ~2 小时完整改造
