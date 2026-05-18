from pathlib import Path
import shutil

import win32com.client
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\kykt\Dream\planning\proposal_dream3r")
WORK = ROOT / "deliverables" / "ppt_work"
REF = Path(r"E:\Work\HSY\气动系统泄漏检测迁移学习.pptx")
OUT = WORK / "proposal_dream3r_opening_report_reference_mode_v13.pptx"
PREVIEW = WORK / "previews_reference_mode_v13"
CONTACT = WORK / "contact_sheet_reference_mode_v13.png"
AI = ROOT / "ppt_assets" / "ai"
ASSETS = WORK / "reference_mode_assets"


SLIDE_SOURCES = [1, 2, 3, 4, 5, 5, 2, 10, 10, 10, 10, 5, 2, 10, 10, 10, 8, 5, 2, 10, 10, 10, 5, 7, 5, 12]

TITLES = [
    "",
    "汇报提纲",
    "研究背景：前馈式三维重建的兴起",
    "代表模型与方法谱系",
    "问题提出：失败模式与研究问题",
    "研究目标与两大支柱",
    "02  候选架构研究",
    "总体架构：六模块设计",
    "记忆模块：长序列内存机制",
    "校验模块：几何自检查与修复",
    "编排模块：专家池与路由策略",
    "跨模块信号合同与实现基础",
    "03  软件平台建设",
    "工具链现状与平台动机",
    "平台架构：四层分离设计",
    "统一执行合同：核心学术抽象",
    "平台进展：模型接入与实测数据",
    "平台与候选架构的协同",
    "04  实验计划与总结",
    "实验设计：架构证据链",
    "实验设计：平台证据链",
    "四个创新点",
    "已完成工作总览",
    "研究计划与时间安排",
    "计划风险与应对",
    "",
]


def rgb(r, g, b):
    return r + g * 256 + b * 65536


BLUE = rgb(0, 70, 142)
DARK_BLUE = rgb(0, 40, 95)
MID_BLUE = rgb(40, 111, 185)
LIGHT_BLUE = rgb(232, 242, 251)
GRAY = rgb(245, 247, 250)
LINE = rgb(70, 120, 180)
ORANGE = rgb(235, 138, 46)
TEXT = rgb(8, 24, 48)
WHITE = rgb(255, 255, 255)


def make_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    font_path = next((p for p in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"] if Path(p).exists()), None)
    f_title = ImageFont.truetype(font_path, 34) if font_path else ImageFont.load_default()
    f_head = ImageFont.truetype(font_path, 24) if font_path else ImageFont.load_default()
    f_body = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()

    # Coverage matrix, simplified to stay readable in the reference deck style.
    im = Image.new("RGB", (1400, 760), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1400, 70), fill=(0, 83, 155))
    d.text((40, 18), "F3 四轴覆盖矩阵（精简）", font=f_title, fill="white")
    rows = [
        ("几何校验", "absent", "作为候选架构重点补足"),
        ("长序列内存", "first-class", "保留显式对照实验"),
        ("多专家组合", "partial", "从孤立比较转为组合机制"),
        ("动态场景", "partial", "作为扩展场景验证"),
        ("统一平台", "absent", "支柱 B 的直接动机"),
        ("测试时机制", "partial", "作为后续可接入分支"),
    ]
    x0, y0 = 60, 120
    widths = [360, 260, 620]
    headers = ["能力维度", "覆盖状态", "对本研究的启示"]
    for j, h in enumerate(headers):
        x = x0 + sum(widths[:j])
        d.rectangle((x, y0, x + widths[j], y0 + 55), fill=(232, 242, 251), outline=(70, 120, 180), width=2)
        d.text((x + 20, y0 + 15), h, font=f_head, fill=(0, 83, 155))
    colors = {"first-class": (64, 158, 105), "partial": (235, 138, 46), "absent": (205, 80, 70)}
    for i, row in enumerate(rows):
        y = y0 + 55 + i * 72
        for j, val in enumerate(row):
            x = x0 + sum(widths[:j])
            d.rectangle((x, y, x + widths[j], y + 72), fill="white", outline=(185, 210, 230), width=1)
            if j == 1:
                d.rounded_rectangle((x + 28, y + 18, x + 200, y + 52), radius=8, fill=colors[val])
                d.text((x + 48, y + 23), val, font=f_body, fill="white")
            else:
                d.text((x + 20, y + 22), val, font=f_body, fill=(35, 50, 75))
    d.text((60, 675), "结论：能力覆盖分布不均，研究目标是建立可对照、可复用的评测入口。", font=f_head, fill=(0, 83, 155))
    im.save(ASSETS / "coverage_matrix.png")

    # Timing bar chart.
    vals = [("Spann3R", 24.8), ("CUT3R", 26.2), ("Fast3R", 28.5), ("DUSt3R", 91.2), ("MASt3R", 95.0), ("MonST3R", 223.3)]
    im = Image.new("RGB", (1400, 760), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1400, 70), fill=(0, 83, 155))
    d.text((40, 18), "F11 跨模型推理耗时（秒）", font=f_title, fill="white")
    max_v = max(v for _, v in vals)
    y = 135
    for name, val in vals:
        d.text((95, y + 8), name, font=f_head, fill=(35, 50, 75))
        w = int(880 * val / max_v)
        color = (235, 138, 46) if name == "MonST3R" else (64, 120, 180)
        d.rounded_rectangle((330, y, 330 + w, y + 36), radius=8, fill=color)
        d.text((330 + w + 20, y + 4), f"{val:.1f}s", font=f_head, fill=(35, 50, 75))
        y += 82
    d.text((95, 680), "结论：耗时差异用于调度与实验管理，不直接等同于模型质量排序。", font=f_head, fill=(0, 83, 155))
    im.save(ASSETS / "timing_bars.png")

    # Gantt-style timeline for the research plan.
    im = Image.new("RGB", (1400, 760), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1400, 70), fill=(0, 83, 155))
    d.text((40, 18), "F16 后续研究计划（M1-M8）", font=f_title, fill="white")
    left, top = 210, 145
    cell_w, row_h = 125, 74
    months = [f"M{i}" for i in range(1, 9)]
    for i, m in enumerate(months):
        x = left + i * cell_w
        d.rectangle((x, top - 45, x + cell_w, top), fill=(232, 242, 251), outline=(185, 210, 230))
        d.text((x + 42, top - 34), m, font=f_body, fill=(0, 83, 155))
    tasks = [
        ("基线复现与合同补齐", 1, 2, (64, 120, 180)),
        ("校验标定与模块消融", 2, 5, (235, 138, 46)),
        ("长序列与多专家评测", 4, 7, (64, 158, 105)),
        ("平台对比视图与报告", 2, 6, (86, 132, 191)),
        ("论文整理与系统固化", 7, 8, (45, 88, 145)),
    ]
    for r, (name, start, end, color) in enumerate(tasks):
        y = top + r * row_h
        d.text((42, y + 18), name, font=f_head, fill=(35, 50, 75))
        for i in range(8):
            x = left + i * cell_w
            d.rectangle((x, y, x + cell_w, y + row_h - 10), outline=(225, 232, 240))
        x1 = left + (start - 1) * cell_w + 16
        x2 = left + end * cell_w - 16
        d.rounded_rectangle((x1, y + 18, x2, y + 44), radius=10, fill=color)
    d.text((42, 675), "说明：时间表为候选安排，受算力授权、导师反馈和阶段评测结果影响。", font=f_head, fill=(0, 83, 155))
    im.save(ASSETS / "timeline_gantt.png")


def shape_text(shape):
    try:
        if shape.HasTextFrame and shape.TextFrame.HasText:
            return shape.TextFrame.TextRange.Text.strip()
    except Exception:
        return ""
    return ""


def set_title(slide, title):
    if not title:
        return
    candidates = []
    for s in slide.Shapes:
        t = shape_text(s)
        if t:
            try:
                if s.Top < 80 and s.Left < 760:
                    candidates.append(s)
            except Exception:
                pass
    candidates.sort(key=lambda s: (s.Top, s.Left))
    for s in candidates:
        try:
            if s.Width > 220 and "北京航空航天大学" not in shape_text(s):
                s.TextFrame.TextRange.Text = title
                s.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
                s.TextFrame.TextRange.Font.Size = 31
                s.TextFrame.TextRange.Font.Bold = -1
                s.TextFrame.TextRange.Font.Color.RGB = DARK_BLUE
                return
        except Exception:
            pass


def clear_body(slide):
    doomed = []
    for s in slide.Shapes:
        try:
            if s.Top > 88:
                doomed.append(s)
        except Exception:
            pass
    for s in doomed:
        try:
            s.Delete()
        except Exception:
            pass


def add_box(slide, x, y, w, h, text="", fill=WHITE, line=LINE, font=20, bold=False, color=TEXT, align=1):
    shp = slide.Shapes.AddShape(1, x, y, w, h)
    shp.Fill.ForeColor.RGB = fill
    shp.Line.ForeColor.RGB = line
    shp.Line.Weight = 1
    if text:
        tr = shp.TextFrame.TextRange
        tr.Text = text
        tr.Font.Name = "Microsoft YaHei"
        tr.Font.Size = font
        tr.Font.Bold = -1 if bold else 0
        tr.Font.Color.RGB = color
        shp.TextFrame.MarginLeft = 8
        shp.TextFrame.MarginRight = 8
        shp.TextFrame.MarginTop = 6
        shp.TextFrame.MarginBottom = 4
        tr.ParagraphFormat.Alignment = align
    return shp


def add_bar(slide, text, y=92):
    return add_box(slide, 34, y, 890, 40, text, fill=BLUE, line=BLUE, font=20, bold=True, color=WHITE, align=2)


def add_label(slide, x, y, w, h, text):
    return add_box(slide, x, y, w, h, text, fill=BLUE, line=BLUE, font=20, bold=True, color=WHITE, align=2)


def add_picture(slide, path, x, y, w, h):
    path = Path(path)
    if not path.exists():
        add_box(slide, x, y, w, h, f"待补充：{path.name}", fill=GRAY, line=LINE, font=16, color=TEXT, align=2)
        return
    im = Image.open(path)
    iw, ih = im.size
    scale = min(w / iw, h / ih)
    nw, nh = iw * scale, ih * scale
    slide.Shapes.AddPicture(str(path), False, True, x + (w - nw) / 2, y + (h - nh) / 2, nw, nh)


def add_arrow(slide, x, y, w=36, h=22):
    shp = slide.Shapes.AddShape(33, x, y, w, h)
    shp.Fill.ForeColor.RGB = MID_BLUE
    shp.Line.ForeColor.RGB = MID_BLUE
    return shp


def bullets(slide, x, y, w, items, head=None):
    if head:
        add_label(slide, x, y, w, 30, head)
        y += 34
    for it in items:
        add_box(slide, x, y, w, 42, it, fill=WHITE, line=LINE, font=15, color=TEXT)
        y += 48


def big_card(slide, x, y, w, h, head, body, accent=BLUE):
    add_box(slide, x, y, w, 34, head, fill=accent, line=accent, font=19, bold=True, color=WHITE, align=2)
    return add_box(slide, x, y + 36, w, h - 36, body, fill=WHITE, line=accent, font=19, color=TEXT, align=2)


def slide_cover(slide):
    for s in slide.Shapes:
        if not shape_text(s):
            continue
        try:
            if s.Top < 45 and s.Width > 300:
                s.TextFrame.TextRange.Text = "硕士学位论文开题报告"
            elif 90 < s.Top < 260 and s.Width > 500:
                tr = s.TextFrame.TextRange
                tr.Text = "面向前馈式三维重建的\n候选架构设计与统一聚合管理平台"
                tr.Font.Size = 42
            elif 330 < s.Top < 380 and s.Left < 430:
                s.TextFrame.TextRange.Text = "汇报人"
            elif 330 < s.Top < 380 and s.Left >= 430:
                s.TextFrame.TextRange.Text = "XXX"
            elif s.Top > 390:
                s.TextFrame.TextRange.Text = "2026年5月"
            else:
                s.TextFrame.TextRange.Text = ""
        except Exception:
            pass


AGENDA_ITEMS = ["背景与现状", "候选架构研究", "软件平台建设", "实验计划与总结"]


def clear_agenda_content(slide):
    doomed = []
    for s in slide.Shapes:
        try:
            # Preserve the copied background images and left blue field, rebuild only the agenda widgets.
            if s.Top > 80 and s.Left > 300:
                doomed.append(s)
            elif s.Top > 80 and 80 < s.Left < 230:
                doomed.append(s)
        except Exception:
            pass
    for s in doomed:
        try:
            s.Delete()
        except Exception:
            pass


def agenda_page(slide, active=None):
    clear_agenda_content(slide)
    # Right grey panel, matching the reference outline page.
    panel = slide.Shapes.AddShape(1, 292, 96, 570, 400)
    panel.Fill.ForeColor.RGB = rgb(235, 236, 238)
    try:
        panel.Fill.Transparency = 0.08
    except Exception:
        pass
    panel.Line.ForeColor.RGB = rgb(230, 230, 230)
    panel.Shadow.Visible = -1
    try:
        panel.Shadow.Transparency = 0.55
        panel.Shadow.Blur = 8
    except Exception:
        pass

    # Vertical title on the left blue area.
    v = slide.Shapes.AddTextbox(1, 120, 205, 75, 210)
    tr = v.TextFrame.TextRange
    tr.Text = "汇\n报\n提\n纲"
    tr.Font.Name = "Microsoft YaHei"
    tr.Font.Size = 30
    tr.Font.Bold = -1
    tr.Font.Color.RGB = WHITE
    tr.ParagraphFormat.Alignment = 2
    line = slide.Shapes.AddLine(112, 250, 112, 450)
    line.Line.ForeColor.RGB = rgb(210, 225, 245)
    try:
        line.Line.Transparency = 0.35
    except Exception:
        pass

    y0 = 112
    for i, item in enumerate(AGENDA_ITEMS, 1):
        y = y0 + (i - 1) * 73
        is_active = active == i
        fill = BLUE if is_active else WHITE
        text_color = WHITE if is_active else rgb(190, 190, 190)
        border = BLUE if is_active else rgb(220, 220, 220)
        add_box(slide, 365, y, 460, 58, "", fill=fill, line=border)
        n = slide.Shapes.AddTextbox(1, 405, y + 7, 70, 44)
        nt = n.TextFrame.TextRange
        nt.Text = f"{i:02d}"
        nt.Font.Name = "Microsoft YaHei"
        nt.Font.Size = 27
        nt.Font.Bold = -1
        nt.Font.Color.RGB = text_color
        nt.ParagraphFormat.Alignment = 2
        div = slide.Shapes.AddLine(486, y + 12, 486, y + 46)
        div.Line.ForeColor.RGB = WHITE if is_active else rgb(210, 210, 210)
        title = slide.Shapes.AddTextbox(1, 510, y + 9, 280, 42)
        tt = title.TextFrame.TextRange
        tt.Text = item
        tt.Font.Name = "Microsoft YaHei"
        tt.Font.Size = 25
        tt.Font.Bold = -1
        tt.Font.Color.RGB = text_color


def slide_outline(slide):
    agenda_page(slide, active=1)


def section_slide(slide, no, title, items):
    agenda_page(slide, active=int(no))


def build_slide(slide, idx):
    if idx == 7:
        section_slide(slide, "02", "候选架构研究", [])
        return
    if idx == 13:
        section_slide(slide, "03", "软件平台建设", [])
        return
    if idx == 19:
        section_slide(slide, "04", "实验计划与总结", [])
        return
    clear_body(slide)
    add_box(slide, 0, 130, 960, 410, "", fill=WHITE, line=WHITE)
    if idx == 3:
        add_bar(slide, "前馈式 3R 方法在两年内从单一基准发展为多个子方向并行演进。")
        big_card(slide, 70, 165, 250, 170, "传统几何流程", "特征匹配\n位姿估计\n三角化\n稠密重建", accent=BLUE)
        add_arrow(slide, 340, 230, 50, 30)
        big_card(slide, 415, 165, 250, 170, "3R 前向预测", "单次前向传播\n像素级三维点图\n置信度与相机信息", accent=MID_BLUE)
        add_arrow(slide, 685, 230, 50, 30)
        big_card(slide, 740, 165, 150, 170, "研究延伸", "跨视图几何\n动态 4D\n长序列记忆\n测试时校验", accent=BLUE)
        add_box(slide, 80, 390, 800, 52, "前馈式 3R 提供了新的统一预测入口，长序列稳定性仍需进一步研究。", fill=LIGHT_BLUE, line=LINE, font=20, bold=True, color=DARK_BLUE, align=2)
    elif idx == 4:
        add_bar(slide, "各模型分别推进匹配、多视角、动态场景和长序列记忆等问题。")
        models = [
            ("点图与匹配", "DUSt3R\nMASt3R"),
            ("多视角规模化", "Fast3R\nVGGT"),
            ("动态场景", "MonST3R\n动态掩码"),
            ("长序列记忆", "CUT3R\nSpann3R"),
            ("测试时校验", "Test3R\n几何一致性"),
        ]
        for i, (h, b) in enumerate(models):
            x = 42 + i * 175
            big_card(slide, x, 165, 145, 150, h, b, accent=BLUE if i in (0, 3) else MID_BLUE)
        add_box(slide, 90, 380, 780, 52, "现有模型形成了丰富方法谱系，关键机制仍分散在不同工作和不同评测口径中。", fill=LIGHT_BLUE, line=LINE, font=19, bold=True, color=DARK_BLUE, align=2)
    elif idx == 5:
        add_bar(slide, "现有 3R 方法面临六类几何失败模式和四组架构层研究问题。")
        failures = ["弱纹理", "镜面反射", "快速运动", "长基线", "尺度漂移", "域外场景"]
        for i, f in enumerate(failures):
            add_box(slide, 55, 150 + i * 44, 135, 32, f, fill=LIGHT_BLUE if i % 2 else WHITE, line=LINE, font=16, bold=True, color=DARK_BLUE, align=2)
        qs = [
            ("Q1", "几何校验与测试时适应如何进入架构层？"),
            ("Q2", "长序列内存机制能否在单一架构内比较？"),
            ("Q3", "多专家组合相对单一专家是否有实证优势？"),
            ("Q4", "能否构建统一的 3R 模型聚合管理平台？"),
        ]
        for i, (q, text) in enumerate(qs):
            y = 150 + i * 62
            add_label(slide, 235, y, 75, 42, q)
            add_box(slide, 330, y, 555, 42, text, fill=WHITE if i % 2 == 0 else LIGHT_BLUE, line=LINE, font=18, bold=(i % 2 == 1), color=DARK_BLUE if i % 2 else TEXT, align=2)
        add_box(slide, 90, 430, 780, 40, "本课题围绕几何校验、长序列记忆、多专家编排和统一平台展开。", fill=BLUE, line=BLUE, font=18, bold=True, color=WHITE, align=2)
    elif idx == 6:
        add_bar(slide, "本课题以候选架构和聚合管理平台为两大支柱，支撑后续对照评估。")
        big_card(slide, 100, 165, 310, 170, "支柱 A：候选架构 X", "六模块设计\n校验、记忆、编排\n跨模块信号契约", accent=BLUE)
        add_arrow(slide, 430, 230, 50, 30)
        big_card(slide, 550, 165, 310, 170, "支柱 B：聚合管理平台", "7 模型统一接入\n调度执行\n跨模型对比", accent=rgb(38, 145, 95))
        add_box(slide, 120, 385, 720, 48, "架构消融依赖平台调度执行，平台为架构提供标准化实验环境。", fill=LIGHT_BLUE, line=LINE, font=20, bold=True, color=DARK_BLUE, align=2)
    elif idx == 8:
        add_bar(slide, "候选架构由感知、记忆、永久性、校验、编排、总线六个模块组成。")
        add_box(slide, 350, 235, 260, 82, "MemoryBus\nCR-1 至 CR-6 类型化交接", fill=BLUE, line=BLUE, font=20, bold=True, color=WHITE, align=2)
        modules = [
            (70, 150, 250, 62, "Perceiver", "视觉骨干 / 特征 token"),
            (640, 150, 250, 62, "SpatialMemory", "三分支注意力 / 空间锚点"),
            (70, 340, 250, 62, "Critic", "几何冲突检测 / 修复动作"),
            (640, 340, 250, 62, "Composer", "7 专家池 / 能力路由"),
            (365, 365, 230, 62, "Permanence", "对象身份 / 动静分离"),
        ]
        for x, y, w, h, head, body in modules:
            add_label(slide, x, y, w, 28, head)
            add_box(slide, x, y + 30, w, h - 30, body, fill=WHITE, line=LINE, font=15, color=TEXT, align=2)
        for x1, y1, x2, y2 in [(315, 200, 350, 260), (645, 200, 610, 260), (315, 370, 350, 292), (645, 370, 610, 292), (480, 365, 480, 317)]:
            ln = slide.Shapes.AddLine(x1, y1, x2, y2)
            ln.Line.ForeColor.RGB = MID_BLUE
            ln.Line.Weight = 2
        add_box(slide, 120, 440, 720, 38, "六个模块通过信号契约协同，输入图像序列，输出点图、动态掩码和中间证据。", fill=LIGHT_BLUE, line=LINE, font=17, bold=True, color=DARK_BLUE, align=2)
    elif idx == 9:
        add_bar(slide, "记忆模块在单一架构内覆盖递推状态、空间指针和混合记忆三类机制。")
        big_card(slide, 60, 165, 240, 145, "压缩分支", "递推状态\nCUT3R 类机制", accent=BLUE)
        add_arrow(slide, 310, 220, 38, 24)
        big_card(slide, 360, 165, 240, 145, "选择分支", "空间锚点存储\nK=256", accent=MID_BLUE)
        add_arrow(slide, 610, 220, 38, 24)
        big_card(slide, 660, 165, 240, 145, "滑窗分支", "Mamba-Transformer\n混合记忆", accent=BLUE)
        add_box(slide, 95, 365, 770, 46, "缓存治理接口已保留，动态剪枝和帧预算控制将在后续对照中验证。", fill=LIGHT_BLUE, line=LINE, font=18, bold=True, color=DARK_BLUE, align=2)
    elif idx == 10:
        add_bar(slide, "校验模块将几何验证作为架构内置组件，通过三类信号检测冲突并触发修复。")
        big_card(slide, 75, 175, 210, 130, "输入信号", "点图对\n置信度\n共视关系", accent=BLUE)
        add_arrow(slide, 300, 225, 42, 26)
        big_card(slide, 365, 155, 230, 170, "几何 Critic", "Sampson 残差\n深度一致性\n共视冲突\n冲突评分", accent=MID_BLUE)
        add_arrow(slide, 610, 225, 42, 26)
        big_card(slide, 675, 175, 210, 130, "修复动作", "不修复\n局部重跑\n全窗口重跑\n路由切换", accent=BLUE)
        add_box(slide, 95, 385, 770, 46, "六类失败模式对应 30 项阈值标定方案，当前处于待执行状态。", fill=LIGHT_BLUE, line=LINE, font=18, bold=True, color=DARK_BLUE, align=2)
    elif idx == 11:
        add_bar(slide, "编排模块通过能力描述符在七个异构专家之间按输入条件路由。")
        experts = [
            ("MASt3R", "静态对"), ("Fast3R", "多视图"), ("Spann3R", "流式"), ("CUT3R", "动态容忍"),
            ("MoGe-2", "单目"), ("DepthAnything", "深度先验"), ("Test3R", "校验"),
        ]
        for i, (name, role) in enumerate(experts):
            x = 48 + (i % 4) * 215
            y = 150 + (i // 4) * 75
            add_box(slide, x, y, 180, 50, f"{name}\n{role}", fill=WHITE if i % 2 == 0 else LIGHT_BLUE, line=LINE, font=15, bold=True, color=DARK_BLUE, align=2)
        add_box(slide, 145, 330, 210, 58, "能力匹配度跨度", fill=LIGHT_BLUE, line=LINE, font=18, bold=True, color=DARK_BLUE, align=2)
        add_arrow(slide, 365, 348, 40, 24)
        add_box(slide, 415, 330, 210, 58, "成本调整解析平局", fill=LIGHT_BLUE, line=LINE, font=18, bold=True, color=DARK_BLUE, align=2)
        add_arrow(slide, 635, 348, 40, 24)
        add_box(slide, 685, 330, 150, 58, "失败退化", fill=LIGHT_BLUE, line=LINE, font=18, bold=True, color=DARK_BLUE, align=2)
        add_box(slide, 95, 430, 770, 38, "后续实验将比较多专家组合与单一专家在不同输入条件下的表现。", fill=BLUE, line=BLUE, font=17, bold=True, color=WHITE, align=2)
    elif idx == 12:
        add_bar(slide, "六条信号校验规则规约跨模块协作；当前已完成实现里程碑 1-18。")
        contracts = [("CR-1", "路由切换约束"), ("CR-2", "静态写入抑制"), ("CR-3", "漂移信号传播"), ("CR-4", "平局窗口处理"), ("CR-5", "证据标签传播"), ("CR-6", "前向引用协议")]
        for i, (h, b) in enumerate(contracts):
            x = 50 + (i % 3) * 290
            y = 145 + (i // 3) * 70
            add_label(slide, x, y, 78, 42, h)
            add_box(slide, x + 85, y, 185, 42, b, fill=WHITE if i % 2 == 0 else LIGHT_BLUE, line=LINE, font=16, color=DARK_BLUE, align=2)
        rows = [
            ("实现进展", "v0.3 前后向流水线、多窗口流式更新、MASt3R/Spann3R 适配器"),
            ("KITTI 验证", "点图 L2 = 20.47；集成证据，非训练后质量"),
        ]
        for i, (h, b) in enumerate(rows):
            y = 310 + i * 58
            add_label(slide, 85, y, 135, 42, h)
            add_box(slide, 240, y, 635, 42, b, fill=WHITE if i == 0 else LIGHT_BLUE, line=LINE, font=16, bold=(i == 1), color=DARK_BLUE, align=2)
    elif idx == 14:
        add_bar(slide, "现有 3R 方向缺乏面向前馈式模型的统一聚合对比平台。")
        rows = [
            ("Nerfstudio", "面向 NeRF 范式，pipeline 抽象与 3R 不兼容"),
            ("商业产品", "闭源，算法细节和评测过程难以审计"),
            ("官方 demo", "孤岛运行，跨模型对比需要手动调度"),
            ("本研究平台", "模型注册、执行合同和对比框架统一管理"),
        ]
        for i, (h, b) in enumerate(rows):
            y = 150 + i * 62
            add_label(slide, 70, y, 145, 44, h)
            add_box(slide, 240, y, 635, 44, b, fill=LIGHT_BLUE if i == 3 else WHITE, line=LINE, font=17, bold=(i == 3), color=DARK_BLUE if i == 3 else TEXT, align=2)
        add_box(slide, 120, 430, 720, 38, "多模型实验需要统一运行入口、输出结构和结果归集方式。", fill=BLUE, line=BLUE, font=17, bold=True, color=WHITE, align=2)
    elif idx == 15:
        add_bar(slide, "平台采用桌面前端、本地后端、远端调度、模型执行器四层分离架构。")
        layers = [
            ("桌面应用层", "Tauri 2 + React：命令中心、任务工作台、样本矩阵"),
            ("本地后端层", "FastAPI：模型注册、任务队列、合同验证"),
            ("远端调度层", "SSH / SCP：启动任务、监听日志、同步产物"),
            ("模型执行层", "7 个执行器：封装各模型推理差异"),
        ]
        for i, (h, body) in enumerate(layers):
            y = 145 + i * 65
            fill = BLUE if i == 0 else MID_BLUE if i == 2 else LIGHT_BLUE
            color = WHITE if i in (0, 2) else DARK_BLUE
            add_box(slide, 100, y, 760, 46, f"{h}    {body}", fill=fill, line=LINE, font=18, bold=True, color=color, align=2)
            if i < len(layers) - 1:
                add_arrow(slide, 460, y + 49, 34, 20)
        add_box(slide, 115, 435, 730, 38, "新模型接入主要通过新增执行器和注册记录完成。", fill=LIGHT_BLUE, line=LINE, font=17, bold=True, color=DARK_BLUE, align=2)
    elif idx == 16:
        add_bar(slide, "三文件合同将异构模型的执行差异封装在执行器层。")
        contracts = [("job.json", "模型名\n输入路径\n参数配置"), ("status.json", "排队\n上传\n运行\n回传\n完成"), ("scene_meta.json", "产物类型\n数量分组\n运行时间")]
        for i, (h, b) in enumerate(contracts):
            big_card(slide, 80 + i * 285, 155, 225, 145, h, b, accent=BLUE if i == 0 else MID_BLUE)
        add_box(slide, 65, 350, 830, 45, "桌面提交 → 后端验证 → SSH 推送 → 远端执行 → SCP 回传 → 结果展示", fill=LIGHT_BLUE, line=LINE, font=18, bold=True, color=DARK_BLUE, align=2)
        add_box(slide, 115, 430, 730, 38, "6 个已验证模型在同一输入上的对比流程保持一致。", fill=BLUE, line=BLUE, font=17, bold=True, color=WHITE, align=2)
    elif idx == 17:
        add_bar(slide, "6 个模型已通过端到端集成验证，跨模型对比矩阵已生成。")
        bullets(slide, 60, 145, 255, ["7 个模型执行器已完成", "6 个端到端验证通过", "Align3R 待权重上传", "统一输入：13 帧 1080p"], "接入状态")
        add_picture(slide, ASSETS / "timing_bars.png", 345, 135, 520, 270)
        add_box(slide, 70, 435, 820, 38, "耗时数据用于说明调度和归集需求，不作为模型质量排序。", fill=LIGHT_BLUE, line=LINE, font=17, bold=True, color=DARK_BLUE, align=2)
    elif idx == 18:
        add_bar(slide, "平台为架构消融实验提供调度执行基础设施，架构需求驱动平台功能演化。")
        big_card(slide, 80, 165, 300, 155, "候选架构需求", "多专家对照\n模块消融\n新模型接入\n结果复核", accent=BLUE)
        add_arrow(slide, 400, 220, 50, 30)
        big_card(slide, 520, 165, 320, 155, "平台支撑方式", "统一提交\n统一输出合同\n论文表格导出\n同协议对比", accent=rgb(38, 145, 95))
        add_box(slide, 95, 390, 770, 48, "短期补齐合同覆盖，中期完善对比视图与报告导出，长期扩展 API 层。", fill=LIGHT_BLUE, line=LINE, font=18, bold=True, color=DARK_BLUE, align=2)
    elif idx == 20:
        add_bar(slide, "架构侧设计了三组消融实验和两组评测，所有实验均为待执行状态。")
        rows = [
            ("架构层消融", "三分支注意力移除、多专家与单专家对照、Test3R-alone 对照"),
            ("记忆机制消融", "12 项测试；四类候选变体 × 五类 fixture regime"),
            ("校验标定", "30 项阈值；六类失败模式 × 5 sub-signal"),
            ("长序列评测", "4 变体 × 4 度量 × 窗口 {10, 20, 50, 100}"),
        ]
        for i, (h, b) in enumerate(rows):
            y = 145 + i * 64
            add_label(slide, 65, y, 160, 46, h)
            add_box(slide, 250, y, 630, 46, b, fill=WHITE if i % 2 == 0 else LIGHT_BLUE, line=LINE, font=16, color=DARK_BLUE, align=2)
        add_box(slide, 115, 430, 730, 38, "预估 GPU 预算约 1377 小时，执行需算力授权。", fill=BLUE, line=BLUE, font=17, bold=True, color=WHITE, align=2)
    elif idx == 21:
        add_bar(slide, "平台侧评测独立于架构算法评测，聚焦统一合同覆盖与对比能力。")
        cards = [("合同覆盖率", "目标 7/7\n当前 6/7"), ("对比矩阵", "平台内聚合\n报告导出"), ("接入成本", "一个执行器\n一条注册记录"), ("API 对接", "REST 端点\n设计完整性")]
        for i, (h, b) in enumerate(cards):
            big_card(slide, 60 + i * 220, 175, 180, 150, h, b, accent=BLUE if i < 2 else MID_BLUE)
        add_box(slide, 95, 395, 770, 46, "平台评测可与架构实验并行推进，形成互补证据。", fill=LIGHT_BLUE, line=LINE, font=19, bold=True, color=DARK_BLUE, align=2)
    elif idx == 22:
        add_bar(slide, "四个创新点分别对应四个研究问题，均以候选方案和对照实验证据为边界。")
        ips = [("IP1", "校验作为架构组件", "对应 Q1"), ("IP2", "异构多专家组合", "对应 Q3"), ("IP3", "长序列内存机制统一", "对应 Q2"), ("IP4", "统一聚合管理平台", "对应 Q4")]
        for i, (ip, text, q) in enumerate(ips):
            x = 70 + (i % 2) * 430
            y = 165 + (i // 2) * 115
            add_label(slide, x, y, 90, 52, ip)
            add_box(slide, x + 110, y, 260, 52, f"{text}\n{q}", fill=WHITE if i % 2 == 0 else LIGHT_BLUE, line=LINE, font=17, bold=True, color=DARK_BLUE, align=2)
        add_box(slide, 115, 420, 730, 38, "创新点表述限定为候选方案与实验证据，不写总体优越性声明。", fill=BLUE, line=BLUE, font=17, bold=True, color=WHITE, align=2)
    elif idx == 23:
        add_bar(slide, "当前已形成架构文档、原型实现、KITTI 集成验证、平台代码和综述材料。")
        big_card(slide, 70, 155, 380, 200, "支柱 A：候选架构", "7 份正式文档\n跨模块信号契约 v2.1\n里程碑 1-18 通过\nKITTI 点图 L2 = 20.47", accent=BLUE)
        big_card(slide, 510, 155, 380, 200, "支柱 B：聚合平台", "约 15000 行代码\n7 模型接入\n6 个端到端验证\n18 页中文综述支撑", accent=rgb(38, 145, 95))
        add_box(slide, 115, 415, 730, 38, "KITTI 数值为集成证据，后续消融和训练需算力授权后启动。", fill=LIGHT_BLUE, line=LINE, font=17, bold=True, color=DARK_BLUE, align=2)
    elif idx == 24:
        add_bar(slide, "后续工作分短期、中期、长期三阶段推进，每项里程碑独立决策。")
        add_picture(slide, ASSETS / "timeline_gantt.png", 90, 130, 780, 320)
        add_box(slide, 80, 455, 800, 42, "短期完稿与补充验证，中期执行标定和对比，长期推进训练、渲染和论文撰写。", fill=LIGHT_BLUE, line=LINE, font=17, bold=True, color=DARK_BLUE, align=2)
    elif idx == 25:
        add_bar(slide, "所有已识别风险均有方案级缓解路径，但仍需后续实验逐步收敛。")
        risks = [
            ("P0", "消融待执行 / 训练后质量未知", "算力授权门控 + 证据边界声明"),
            ("P1", "域外检测缺口 / 渲染许可链", "标定方案规划 + 渲染器执行门控"),
            ("P2", "远端环境漂移 / 对比公平性", "环境锁定 + 统一预处理"),
        ]
        for i, (p, r, m) in enumerate(risks):
            y = 165 + i * 78
            add_label(slide, 75, y, 90, 52, p)
            add_box(slide, 190, y, 315, 52, r, fill=WHITE, line=LINE, font=16, color=TEXT, align=2)
            add_box(slide, 525, y, 335, 52, m, fill=LIGHT_BLUE, line=LINE, font=16, bold=True, color=DARK_BLUE, align=2)
        add_box(slide, 115, 430, 730, 38, "风险管理随实证推进动态更新，不在开题阶段提前关闭。", fill=BLUE, line=BLUE, font=17, bold=True, color=WHITE, align=2)


def slide_thanks(slide):
    texts = [s for s in slide.Shapes if shape_text(s)]
    texts.sort(key=lambda s: (s.Top, s.Left))
    for s in texts:
        try:
            s.TextFrame.TextRange.Text = ""
        except Exception:
            pass
    add_box(slide, 90, 165, 780, 70, "总结与致谢\n敬请批评指正！", fill=rgb(255, 255, 255), line=rgb(255, 255, 255), font=34, bold=True, color=BLUE, align=2)
    add_box(slide, 155, 280, 650, 116, "本研究关注前馈式三维重建在长序列、动态干扰和多模型对比方面的架构层问题。\n拟设计包含显式空间记忆、几何校验和专家编排的候选架构，并构建统一的 3R 模型聚合管理平台。", fill=rgb(255, 255, 255), line=rgb(255, 255, 255), font=18, bold=True, color=TEXT, align=2)


def export_contact():
    def key(p):
        ds = "".join(ch for ch in p.stem if ch.isdigit())
        return int(ds) if ds else 0
    imgs = sorted(PREVIEW.glob("*.png"), key=key)
    thumbs = []
    for i, p in enumerate(imgs, 1):
        im = Image.open(p).convert("RGB")
        im.thumbnail((320, 180))
        c = Image.new("RGB", (320, 210), "white")
        c.paste(im, ((320 - im.width) // 2, 0))
        d = ImageDraw.Draw(c)
        d.text((6, 186), f"Slide {i:02d}", fill=(0, 0, 0))
        thumbs.append(c)
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 210), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 320, (i // cols) * 210))
    sheet.save(CONTACT)


def main():
    make_assets()
    if OUT.exists():
        OUT.unlink()
    if PREVIEW.exists():
        shutil.rmtree(PREVIEW)
    PREVIEW.mkdir(parents=True, exist_ok=True)

    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    pres = app.Presentations.Add()
    for src in SLIDE_SOURCES:
        pres.Slides.InsertFromFile(str(REF), pres.Slides.Count, src, src)
    while pres.Slides.Count > len(SLIDE_SOURCES):
        pres.Slides(1).Delete()

    for i, title in enumerate(TITLES, 1):
        slide = pres.Slides(i)
        if i == 1:
            slide_cover(slide)
        elif i == 2:
            set_title(slide, title)
            slide_outline(slide)
        elif i == 26:
            slide_thanks(slide)
        else:
            set_title(slide, title)
            build_slide(slide, i)

    pres.SaveAs(str(OUT))
    pres.Export(str(PREVIEW), "PNG")
    try:
        pres.Close()
    except Exception:
        pass
    try:
        app.Quit()
    except Exception:
        pass
    export_contact()
    print(OUT)
    print(PREVIEW)
    print(CONTACT)


if __name__ == "__main__":
    main()
