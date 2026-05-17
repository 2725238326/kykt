from pathlib import Path
import shutil

import win32com.client
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\kykt\Dream\planning\proposal_dream3r")
WORK = ROOT / "deliverables" / "ppt_work"
REF = Path(r"E:\Work\HSY\气动系统泄漏检测迁移学习.pptx")
OUT = WORK / "proposal_dream3r_opening_report_reference_mode_v3.pptx"
PREVIEW = WORK / "previews_reference_mode_v3"
CONTACT = WORK / "contact_sheet_reference_mode_v3.png"
AI = ROOT / "ppt_assets" / "ai"
ASSETS = WORK / "reference_mode_assets"


SLIDE_SOURCES = [1, 2, 3, 4, 6, 5, 5, 10, 10, 5, 10, 10, 10, 8, 7, 6, 5, 5, 4, 12]

TITLES = [
    "",
    "汇报提纲",
    "研究背景：3R 方向快速演进",
    "问题提出：失败模式与未解问题",
    "研究现状：四轴覆盖矩阵",
    "工具链空白：支柱 B 的动机",
    "研究目标：四个目标与两大支柱",
    "候选架构 X：整体设计",
    "校验模块：几何一致性反馈",
    "编排模块：多专家路由机制",
    "记忆模块：长序列状态管理",
    "聚合管理平台：四层架构",
    "统一执行合同：跨模型抽象",
    "跨模型实测：耗时差异明显",
    "实验设计：三层证据链",
    "创新点：问题驱动的四项设计",
    "支柱 A 进展：候选架构入口",
    "支柱 B 进展：平台工程基础",
    "风险分析：边界与对策",
    "",
]


def rgb(r, g, b):
    return r + g * 256 + b * 65536


BLUE = rgb(0, 83, 155)
DARK_BLUE = rgb(0, 61, 130)
MID_BLUE = rgb(40, 111, 185)
LIGHT_BLUE = rgb(232, 242, 251)
GRAY = rgb(245, 247, 250)
LINE = rgb(70, 120, 180)
ORANGE = rgb(235, 138, 46)
TEXT = rgb(35, 50, 75)
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
    d.text((95, 680), "读法：耗时差异用于调度与实验管理，不直接等同于模型质量排序。", font=f_head, fill=(0, 83, 155))
    im.save(ASSETS / "timing_bars.png")


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


def add_box(slide, x, y, w, h, text="", fill=WHITE, line=LINE, font=16, bold=False, color=TEXT, align=1):
    shp = slide.Shapes.AddShape(1, x, y, w, h)
    shp.Fill.ForeColor.RGB = fill
    shp.Line.ForeColor.RGB = line
    shp.Line.Weight = 1
    if text:
        tr = shp.TextFrame.TextRange
        tr.Text = text
        tr.Font.Name = "微软雅黑"
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
    return add_box(slide, 34, y, 890, 34, text, fill=BLUE, line=BLUE, font=16, bold=True, color=WHITE, align=2)


def add_label(slide, x, y, w, h, text):
    return add_box(slide, x, y, w, h, text, fill=BLUE, line=BLUE, font=17, bold=True, color=WHITE, align=2)


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


def bullets(slide, x, y, w, items, head=None):
    if head:
        add_label(slide, x, y, w, 30, head)
        y += 34
    for it in items:
        add_box(slide, x, y, w, 42, it, fill=WHITE, line=LINE, font=15, color=TEXT)
        y += 48


def slide_cover(slide):
    texts = [s for s in slide.Shapes if shape_text(s)]
    texts.sort(key=lambda s: (s.Top, s.Left))
    repl = [
        "面向前馈式三维重建的\n候选架构设计与统一聚合管理平台",
        "硕士学位论文开题报告",
        "汇报人：XXX",
        "2026年5月",
    ]
    for i, s in enumerate(texts):
        try:
            s.TextFrame.TextRange.Text = repl[i] if i < len(repl) else ""
        except Exception:
            pass


def slide_outline(slide):
    clear_body(slide)
    add_bar(slide, "20 页版本按 15-20 分钟组织，后续可压缩为答辩短版")
    items = ["01 研究背景与问题", "02 研究现状与目标", "03 候选架构与平台设计", "04 实验计划与进展", "05 风险边界与总结"]
    y = 155
    for i, item in enumerate(items, 1):
        add_label(slide, 250, y, 70, 38, f"{i:02d}")
        add_box(slide, 330, y, 370, 38, item[3:], fill=WHITE, line=LINE, font=18, bold=True, color=BLUE)
        y += 55


def build_slide(slide, idx):
    clear_body(slide)
    if idx == 3:
        add_bar(slide, "DUSt3R 之后，前馈式三维重建从单模型能力扩展到长序列、动态场景与测试时机制。")
        names = ["2024\nDUSt3R", "2024\nMASt3R", "2025\nFast3R", "2024-25\nMonST3R", "2025\nSpann3R / CUT3R", "2025\nTest3R / TTT3R"]
        desc = ["pose-free\npointmap", "3D grounding\nmatching", "many-view\nsingle forward", "video / dynamic\n4D", "long-sequence\nmemory", "test-time\nmechanisms"]
        x0, gap = 50, 145
        for i, (n, d) in enumerate(zip(names, desc)):
            x = x0 + i * gap
            add_box(slide, x, 165, 110, 50, n, fill=LIGHT_BLUE, line=LINE, font=13, bold=True, color=BLUE, align=2)
            add_box(slide, x, 225, 110, 54, d, fill=WHITE, line=LINE, font=12, color=TEXT, align=2)
        add_box(slide, 70, 340, 820, 70, "研究缺口：比较口径、复现入口与跨模型实验记录尚未统一。", fill=WHITE, line=LINE, font=22, bold=True, color=BLUE, align=2)
    elif idx == 4:
        add_bar(slide, "本研究不押注单一分支，而是围绕失败模式建立候选架构与统一评测入口。")
        cards = [("几何验证", "尺度漂移、重投影误差\n缺少统一反馈机制"), ("长序列内存", "跨片段状态衰减\n一致性难保持"), ("多专家组合", "模型能力互补\n但缺少编排接口"), ("统一平台", "入口、格式、日志\n难以横向比较")]
        for i, (h, b) in enumerate(cards):
            x = 55 + (i % 2) * 455
            y = 165 + (i // 2) * 130
            add_label(slide, x, y, 125, 82, h)
            add_box(slide, x + 130, y, 300, 82, b, fill=WHITE, line=LINE, font=18, color=TEXT, align=2)
        add_box(slide, 85, 435, 790, 45, "研究边界：候选架构不是最终方案，目标是形成可对照、可复用的实验入口。", fill=LIGHT_BLUE, line=LINE, font=18, bold=True, color=BLUE, align=2)
    elif idx == 5:
        add_bar(slide, "覆盖矩阵显示现有方法在几何校验、长序列内存、多专家组合和统一平台方面覆盖不均。")
        add_picture(slide, ASSETS / "coverage_matrix.png", 115, 142, 730, 330)
    elif idx == 6:
        add_bar(slide, "当前工具链主要瓶颈不在模型数量，而在统一入口、统一格式和统一实验记录不足。")
        rows = [("模型入口分散", "不同仓库、环境与脚本割裂"), ("输出格式异构", "pointmap、mesh、动态轨迹难以横向比较"), ("日志口径不一", "耗时、状态、失败原因缺少统一记录")]
        for i, (h, b) in enumerate(rows):
            y = 155 + i * 88
            add_label(slide, 80, y, 150, 54, h)
            add_box(slide, 245, y, 620, 54, b, fill=WHITE, line=LINE, font=18, color=TEXT, align=2)
        add_box(slide, 100, 430, 760, 44, "支柱 B 的作用：降低比较成本，并支撑支柱 A 的可验证性。", fill=BLUE, line=BLUE, font=18, bold=True, color=WHITE, align=2)
    elif idx == 7:
        add_bar(slide, "两大支柱不是并列堆砌：平台提供实验入口，候选架构提出可检验假设。")
        bullets(slide, 60, 155, 250, ["可检验的候选架构 X", "校验 / 编排 / 记忆模块", "消融实验与对照验证"], "支柱 A")
        add_picture(slide, AI / "F04_two_pillars_3840x2160.png", 335, 165, 290, 205)
        bullets(slide, 650, 155, 250, ["统一执行合同", "7 模型接入", "远端调度与日志记录"], "支柱 B")
        add_box(slide, 100, 430, 760, 40, "三条红线：候选不等于结论；不押注单一分支；只提供比较数据与评估入口。", fill=LIGHT_BLUE, line=LINE, font=17, bold=True, color=BLUE, align=2)
    elif idx == 8:
        add_bar(slide, "候选架构 X 由前馈模型、校验反馈、多专家编排和记忆管理组成，后续通过消融实验验证。")
        add_picture(slide, AI / "F05_candidate_architecture_x_3840x2160.png", 80, 145, 800, 305)
    elif idx == 9:
        add_bar(slide, "校验模块将几何一致性信号反馈到前馈流程，用于发现失败案例并支撑修正策略。")
        add_picture(slide, AI / "F06_critic_module_flow_3840x2160.png", 80, 145, 800, 305)
    elif idx == 10:
        add_bar(slide, "编排模块不预设某个模型最优，而是依据任务与场景特征进行可配置组合。")
        add_picture(slide, AI / "F07_routing_decision_flow_3840x2160.png", 80, 145, 560, 285)
        bullets(slide, 670, 150, 220, ["任务输入：场景、序列、动态程度", "路由依据：失败信号与资源约束", "输出：模型选择与聚合记录"], "核心机制")
    elif idx == 11:
        add_bar(slide, "记忆模块面向长序列输入，统一短期状态、长期状态和检索记忆三类机制。")
        add_picture(slide, AI / "F08_memory_three_branch_3840x2160.png", 80, 145, 800, 305)
    elif idx == 12:
        add_bar(slide, "聚合管理平台采用桌面前端、本地后端、远端调度和模型执行层的分层结构。")
        add_picture(slide, AI / "F09_platform_four_layer_architecture_3840x2160.png", 80, 145, 800, 305)
        add_box(slide, 160, 455, 640, 28, "平台 UI 截图待版本稳定后补充，本页仅展示架构抽象。", fill=LIGHT_BLUE, line=LINE, font=14, color=BLUE, align=2)
    elif idx == 13:
        add_bar(slide, "统一执行合同规定输入、状态、日志、输出和错误返回，是跨模型比较的基础。")
        add_picture(slide, AI / "F10_execution_contract_sequence_3840x2160.png", 80, 145, 800, 305)
    elif idx == 14:
        add_bar(slide, "远端视频输入下不同模型推理耗时差异明显，说明统一调度和可观测记录具有必要性。")
        add_picture(slide, ASSETS / "timing_bars.png", 115, 135, 730, 335)
    elif idx == 15:
        add_bar(slide, "实验设计围绕场景覆盖、校验消融、长序列评测和平台复用能力展开。")
        add_picture(slide, AI / "F12_experiment_design_overview_3840x2160.png", 80, 145, 800, 305)
    elif idx == 16:
        add_bar(slide, "创新点均从研究问题出发，后续通过对照实验验证，不提前宣称性能结论。")
        items = [("IP1", "校验作为架构组件"), ("IP2", "异构多专家组合"), ("IP3", "长序列内存统一"), ("IP4", "统一聚合管理平台")]
        for i, (h, b) in enumerate(items):
            x = 60 + i * 220
            add_label(slide, x, 160, 180, 36, h)
            add_box(slide, x, 200, 180, 95, b, fill=WHITE, line=LINE, font=18, bold=True, color=BLUE, align=2)
        add_picture(slide, AI / "F13_question_innovation_mapping_3840x2160.png", 225, 325, 510, 120)
    elif idx == 17:
        add_bar(slide, "支柱 A 已完成候选架构拆分与实验入口梳理，后续重点是消融验证。")
        bullets(slide, 70, 150, 360, ["综述 18 页，44 篇引用", "KITTI 集成验证与实验规划", "Critic / Composer / Memory 模块拆分"], "已完成")
        bullets(slide, 520, 150, 360, ["统一 ablation 与 benchmark", "长序列与动态场景对照组", "避免把计划写成既有结果"], "待验证")
    elif idx == 18:
        add_bar(slide, "支柱 B 已具备多模型接入、统一执行合同和远端调度基础，UI 截图待稳定后补充。")
        bullets(slide, 70, 150, 360, ["7 个模型接入", "6 项验证通过", "约 15000 行代码"], "工程基础")
        bullets(slide, 520, 150, 360, ["文件上传、远端执行、日志记录、结果回传", "UI 暂不展示，避免未定稿界面", "后续补充稳定版截图"], "平台证据")
    elif idx == 19:
        add_bar(slide, "风险可控的前提是明确 claim 边界：候选方案必须由对照实验支持。")
        cards = [("实验风险", "性能波动或提升不稳定\n设置强基线与消融组"), ("工程风险", "模型适配与资源受限\n保留离线执行方案"), ("表达风险", "避免计划写成结果\n统一使用候选与评估表述"), ("进度风险", "平台与论文并行推进\n按阶段锁定交付物")]
        for i, (h, b) in enumerate(cards):
            x = 65 + (i % 2) * 430
            y = 160 + (i // 2) * 125
            add_label(slide, x, y, 130, 70, h)
            add_box(slide, x + 140, y, 265, 70, b, fill=WHITE, line=LINE, font=16, color=TEXT, align=2)


def slide_thanks(slide):
    texts = [s for s in slide.Shapes if shape_text(s)]
    texts.sort(key=lambda s: (s.Top, s.Left))
    for s in texts:
        try:
            s.TextFrame.TextRange.Text = ""
        except Exception:
            pass
    add_box(slide, 90, 185, 780, 80, "总结与致谢\n敬请批评指正！", fill=rgb(255, 255, 255), line=rgb(255, 255, 255), font=34, bold=True, color=BLUE, align=2)
    add_box(slide, 170, 310, 620, 56, "本研究以候选架构 X 与聚合管理平台为两大支柱，面向前馈式三维重建提供可对照、可复用的研究框架。", fill=rgb(255, 255, 255), line=rgb(255, 255, 255), font=18, bold=True, color=TEXT, align=2)


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
        elif i == 20:
            slide_thanks(slide)
        else:
            set_title(slide, title)
            build_slide(slide, i)

    pres.SaveAs(str(OUT))
    pres.Export(str(PREVIEW), "PNG")
    pres.Close()
    app.Quit()
    export_contact()
    print(OUT)
    print(PREVIEW)
    print(CONTACT)


if __name__ == "__main__":
    main()
