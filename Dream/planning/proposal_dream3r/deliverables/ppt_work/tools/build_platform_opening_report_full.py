from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw
import win32com.client


BASE = Path(r"E:\kykt\Dream\planning\proposal_dream3r\deliverables\ppt_work")
SRC = Path(r"E:\Work\HSY\学术风ppt模板-蓝色.pptx")
OUT = BASE / "platform_opening_report_full.pptx"
PREVIEW_DIR = BASE / "previews_platform_opening_report_full"
CONTACT = BASE / "contact_sheet_platform_opening_report_full.png"


def rgb(r: int, g: int, b: int) -> int:
    return r + g * 256 + b * 65536


DARK = rgb(0, 65, 125)
BLUE = rgb(0, 84, 160)
MID = rgb(35, 108, 180)
LIGHT = rgb(232, 241, 250)
PALE = rgb(246, 250, 253)
WHITE = rgb(255, 255, 255)
LINE = rgb(43, 107, 185)
TEXT = rgb(5, 35, 70)
GRAY = rgb(92, 105, 120)


def shape(slide, shape_id: int):
    for idx in range(1, slide.Shapes.Count + 1):
        shp = slide.Shapes(idx)
        if shp.Id == shape_id:
            return shp
    return None


def set_text(slide, shape_id: int, text: str, size: float | None = None, bold: bool | None = None, color: int | None = None):
    shp = shape(slide, shape_id)
    if shp is None:
        return
    tr = shp.TextFrame.TextRange
    tr.Text = text
    tr.Font.Name = "微软雅黑"
    if size is not None:
        tr.Font.Size = size
    if bold is not None:
        tr.Font.Bold = -1 if bold else 0
    if color is not None:
        tr.Font.Color.RGB = color


def add_box(slide, x, y, w, h, text="", fill=WHITE, line=LINE, size=15, bold=False, color=TEXT, align=2):
    shp = slide.Shapes.AddShape(1, x, y, w, h)
    shp.Fill.ForeColor.RGB = fill
    shp.Line.ForeColor.RGB = line
    shp.Line.Weight = 1
    if text:
        tr = shp.TextFrame.TextRange
        tr.Text = text
        tr.Font.Name = "微软雅黑"
        tr.Font.Size = size
        tr.Font.Bold = -1 if bold else 0
        tr.Font.Color.RGB = color
        tr.ParagraphFormat.Alignment = align
        shp.TextFrame.MarginLeft = 8
        shp.TextFrame.MarginRight = 8
        shp.TextFrame.MarginTop = 5
        shp.TextFrame.MarginBottom = 5
        try:
            shp.TextFrame.VerticalAnchor = 3
        except Exception:
            pass
    return shp


def add_text(slide, x, y, w, h, text, size=15, bold=False, color=TEXT, align=1):
    shp = slide.Shapes.AddTextbox(1, x, y, w, h)
    tr = shp.TextFrame.TextRange
    tr.Text = text
    tr.Font.Name = "微软雅黑"
    tr.Font.Size = size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = color
    tr.ParagraphFormat.Alignment = align
    shp.TextFrame.MarginLeft = 2
    shp.TextFrame.MarginRight = 2
    shp.TextFrame.MarginTop = 1
    shp.TextFrame.MarginBottom = 1
    return shp


def add_arrow(slide, x1, y1, x2, y2, color=LINE, weight=2):
    ln = slide.Shapes.AddLine(x1, y1, x2, y2)
    ln.Line.ForeColor.RGB = color
    ln.Line.Weight = weight
    try:
        ln.Line.EndArrowheadStyle = 3
    except Exception:
        pass
    return ln


def set_notes(slide, text: str):
    body = None
    for idx in range(1, slide.NotesPage.Shapes.Count + 1):
        shp = slide.NotesPage.Shapes(idx)
        try:
            if shp.PlaceholderFormat.Type == 2:
                body = shp
                break
        except Exception:
            pass
    if body is None:
        body = slide.NotesPage.Shapes.AddTextbox(1, 54, 346.5, 432, 283.5)
    body.TextFrame.TextRange.Text = text
    body.TextFrame.TextRange.Font.Name = "微软雅黑"
    body.TextFrame.TextRange.Font.Size = 11


def mask_body(slide):
    add_box(slide, 0, 84, 960, 456, "", fill=WHITE, line=WHITE)


def header(slide, no: str, title: str, claim: str | None = None):
    set_text(slide, 90, no, 18, True, DARK)
    set_text(slide, 92, title, 25, True, DARK)
    mask_body(slide)
    if claim:
        add_box(slide, 34, 96, 892, 42, claim, fill=BLUE, line=BLUE, size=17, bold=True, color=WHITE)


def three_cards(slide, y, titles, bodies, footer=None):
    xs = [58, 358, 658]
    for x, t, b in zip(xs, titles, bodies):
        add_box(slide, x, y, 245, 42, t, fill=BLUE, line=BLUE, size=18, bold=True, color=WHITE)
        add_box(slide, x, y + 48, 245, 132, b, fill=LIGHT, line=LINE, size=14, bold=True, color=TEXT)
    if footer:
        add_box(slide, 110, 430, 740, 42, footer, fill=LIGHT, line=LINE, size=16, bold=True)


def five_flow(slide, labels, y=245):
    x0 = 55
    gap = 174
    for i, label in enumerate(labels):
        x = x0 + i * gap
        add_box(slide, x, y, 126, 54, label, fill=LIGHT if i % 2 == 0 else WHITE, line=LINE, size=15, bold=True)
        if i < len(labels) - 1:
            add_arrow(slide, x + 130, y + 27, x + gap - 12, y + 27, MID, 2)


def table(slide, x, y, col_widths, rows, header_fill=BLUE):
    row_h = 36
    for r, row in enumerate(rows):
        cx = x
        for c, text in enumerate(row):
            fill = header_fill if r == 0 else (LIGHT if r % 2 == 1 else WHITE)
            color = WHITE if r == 0 else TEXT
            add_box(slide, cx, y + r * row_h, col_widths[c], row_h, text, fill=fill, line=LINE, size=13 if r else 14, bold=True, color=color)
            cx += col_widths[c]


def section(slide, no: str, title: str, subtitle: str, items: list[str]):
    add_box(slide, 0, 0, 960, 540, "", fill=WHITE, line=WHITE)
    add_box(slide, 0, 0, 960, 42, "", fill=BLUE, line=BLUE)
    add_box(slide, 0, 500, 960, 40, "", fill=BLUE, line=BLUE)
    add_box(slide, 170, 118, 110, 52, no, fill=BLUE, line=BLUE, size=22, bold=True, color=WHITE)
    add_text(slide, 315, 115, 560, 58, title, size=29, bold=True, color=DARK)
    add_text(slide, 170, 205, 690, 36, subtitle, size=18, bold=True, color=MID, align=2)
    for i, item in enumerate(items):
        add_box(slide, 250, 280 + i * 50, 460, 36, item, fill=LIGHT if i % 2 == 0 else WHITE, line=LINE, size=16, bold=True)


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    pres = app.Presentations.Add()

    pages = [3, 5, 15, 15, 37, 15, 30, 15, 25, 48, 15, 48, 15, 30, 48, 72]
    for p in pages:
        pres.Slides.InsertFromFile(str(SRC), pres.Slides.Count, p, p)

    # 1 cover
    s = pres.Slides(1)
    set_text(s, 7, "2026年春科研课堂开题报告", 16)
    set_text(s, 2, "面向前馈式三维重建模型的\n聚合管理平台设计与实现", 30, True)
    set_text(s, 4, "汇报人", 16, True)
    set_text(s, 5, "崔昊喆  纪博闻", 18, True)
    set_text(s, 8, "北京航空航天大学", 16)
    set_text(s, 10, "2026年6月", 16)
    set_notes(s, "开场：本次开题聚焦平台方向，目标是构建面向前馈式三维重建模型的统一实验、部署与调用平台。")

    # 2 outline
    s = pres.Slides(2)
    add_box(s, 355, 82, 520, 380, "", fill=PALE, line=rgb(160, 190, 225), size=1)
    items = [("01", "研究背景与问题提出"), ("02", "平台定位与研究目标"), ("03", "系统设计与关键实现"), ("04", "实验验证与进度安排")]
    for i, (no, txt) in enumerate(items):
        y = 116 + i * 70
        add_box(s, 385, y, 92, 46, no, fill=BLUE if i == 0 else WHITE, line=LINE, size=17, bold=True, color=WHITE if i == 0 else DARK)
        add_box(s, 492, y, 348, 46, txt, fill=WHITE, line=LINE, size=17, bold=True, color=DARK, align=1)
    set_notes(s, "目录分四部分：背景问题、平台定位、系统设计、验证计划。")

    # 3 background
    s = pres.Slides(3)
    header(s, "1.1", "研究背景", "前馈式 3R 模型降低了三维重建门槛，但工程使用仍然分散")
    add_text(s, 70, 170, 820, 38, "DUSt3R 以来，三维重建逐步从多阶段几何流程转向端到端前馈推理。", 17, True, DARK, 2)
    five_flow(s, ["输入图像", "模型推理", "点图输出", "结果转换", "应用使用"], y=245)
    add_box(s, 90, 350, 780, 58, "平台建设的出发点：把模型能力从论文 demo 和命令行脚本中抽取出来，形成可复现、可对比、可扩展的工程流程。", fill=LIGHT, line=LINE, size=16, bold=True)
    set_notes(s, "背景页只讲方向变化：3R 模型让重建更直接，但从论文到可用工具仍有工程距离。")

    # 4 problem
    s = pres.Slides(4)
    header(s, "1.2", "问题提出", "多模型实验的主要成本来自部署、调度和结果对齐")
    three_cards(
        s,
        178,
        ["部署成本高", "算力使用不便", "对比难复现"],
        [
            "模型环境、权重路径和运行脚本分散\n不同仓库缺少统一入口",
            "远端 GPU 任务依赖手动命令\n状态、日志、产物回传不集中",
            "输入输出格式差异明显\n横向比较需要大量人工整理",
        ],
        "平台要解决的不是单一模型效果，而是多模型实验流程的标准化。",
    )
    set_notes(s, "问题页强调真实痛点：部署、远端算力、结果对齐。避免把平台说成单纯 UI。")

    # 5 section
    s = pres.Slides(5)
    section(s, "02", "平台定位与研究目标", "从实验辅助工具，扩展为模型能力的统一入口。", ["统一部署", "统一调度", "统一对比", "标准化输出"])
    set_notes(s, "第二部分说明平台定位和研究目标。")

    # 6 positioning
    s = pres.Slides(6)
    header(s, "2.1", "平台定位", "平台定位为前馈式 3R 模型的实验、部署与输出基础设施")
    three_cards(
        s,
        180,
        ["学习与实践", "实验与研究", "应用与扩展"],
        [
            "降低新模型运行门槛\n沉淀可复用配置与样例",
            "同一输入下运行多模型\n记录日志、耗时和输出产物",
            "将稳定能力封装为接口\n支撑后续应用调用",
        ],
        "定位边界：平台支撑模型实验和部署，不替代模型本身的算法贡献。",
    )
    set_notes(s, "这页讲平台三层价值：学习实践、实验研究、应用扩展。")

    # 7 objectives
    s = pres.Slides(7)
    header(s, "2.2", "研究目标", "围绕多模型实验流程，形成可运行、可扩展、可复核的平台原型")
    rows = [
        ["目标", "内容", "验收方式"],
        ["模型接入", "统一描述模型环境、命令和输出目录", "完成代表模型执行器"],
        ["任务调度", "支持本地提交、远端运行、日志监听和回传", "端到端任务可追踪"],
        ["结果管理", "归集点图、mask、日志和元数据", "同一输入可横向查看"],
        ["能力输出", "预留统一导出和 API 封装路径", "形成接口设计文档"],
    ]
    table(s, 70, 160, [130, 485, 250], rows)
    add_box(s, 130, 430, 700, 38, "研究目标强调流程闭环：接入、运行、归集、对比和输出。", fill=LIGHT, line=LINE, size=16, bold=True)
    set_notes(s, "目标页用可验收表述，不写宏大口号。")

    # 8 functions
    s = pres.Slides(8)
    header(s, "2.3", "核心功能设计", "平台围绕样本、模型、任务和结果四类对象组织功能")
    labels = ["样本管理", "模型注册", "任务工作台", "日志与状态", "结果归集"]
    five_flow(s, labels, y=185)
    three_cards(
        s,
        305,
        ["命令中心", "样本矩阵", "模型路线图"],
        [
            "选择模型、输入与参数\n提交任务并查看状态",
            "同一输入组织多模型结果\n支持对比和记录复用",
            "维护模型接入状态\n区分已验证与待补充模型",
        ],
    )
    set_notes(s, "功能页把平台对象讲清楚：样本、模型、任务、日志、结果。")

    # 9 architecture
    s = pres.Slides(9)
    header(s, "3.1", "技术架构", "桌面前端、本地后端、远端调度和执行器分层解耦")
    layers = [
        ("桌面前端", "Tauri 2 + React\n任务配置、结果查看、交互入口"),
        ("本地后端", "FastAPI\n模型注册、任务队列、合同校验"),
        ("远端调度", "SSH / SCP\n上传输入、启动任务、同步产物"),
        ("模型执行器", "封装各模型命令\n适配输出目录与元数据"),
    ]
    for i, (name, body) in enumerate(layers):
        y = 150 + i * 75
        add_box(s, 130, y, 170, 46, name, fill=BLUE if i % 2 == 0 else MID, line=BLUE, size=17, bold=True, color=WHITE)
        add_box(s, 320, y, 500, 46, body, fill=LIGHT if i % 2 == 0 else WHITE, line=LINE, size=14, bold=True, color=TEXT, align=1)
        if i < len(layers) - 1:
            add_arrow(s, 570, y + 48, 570, y + 70, MID, 2)
    add_box(s, 130, 455, 690, 38, "执行器封装模型差异，平台层保持统一提交、跟踪、归集和导出。", fill=LIGHT, line=LINE, size=16, bold=True)
    set_notes(s, "架构页讲四层分工。重点是执行器隔离模型差异，平台负责统一流程。")

    # 10 contract
    s = pres.Slides(10)
    header(s, "3.2", "统一执行合同", "三类状态文件把模型脚本纳入统一任务流程")
    rows = [
        ["文件", "作用", "平台使用方式"],
        ["job.json", "记录模型、输入、参数和输出目录", "任务提交与复现依据"],
        ["status.json", "记录运行状态、进度、错误信息", "状态刷新与失败诊断"],
        ["scene_meta.json", "记录结果产物、相机与场景信息", "结果展示与对比入口"],
    ]
    table(s, 75, 160, [165, 345, 345], rows)
    five_flow(s, ["提交", "验证", "执行", "回传", "解析"], y=350)
    add_box(s, 120, 445, 720, 38, "统一合同使新模型接入从“改平台代码”转为“实现执行器接口”。", fill=LIGHT, line=LINE, size=16, bold=True)
    set_notes(s, "合同页是平台的核心抽象。不要讲太多代码，只讲三类文件如何统一流程。")

    # 11 running workflow
    s = pres.Slides(11)
    header(s, "3.3", "实际运行流程", "一次任务覆盖输入准备、远端执行、日志监听和结果回传")
    five_flow(s, ["导入样本", "选择模型", "提交任务", "监听日志", "同步结果"], y=175)
    add_box(s, 70, 285, 390, 116, "本地侧\n- 组织图像序列、单图或图像对\n- 生成任务合同\n- 展示状态、日志和产物索引", fill=LIGHT, line=LINE, size=15, bold=True, align=1)
    add_box(s, 500, 285, 390, 116, "远端侧\n- 接收输入和配置\n- 执行模型推理脚本\n- 输出点图、mask、日志和元数据", fill=WHITE, line=LINE, size=15, bold=True, align=1)
    add_box(s, 130, 445, 700, 38, "流程目标：减少手动命令，保留可复核的运行记录。", fill=LIGHT, line=LINE, size=16, bold=True)
    set_notes(s, "运行流程页是展示平台实用性的关键页。中期报告时这里要换成真实截图。")

    # 12 model coverage
    s = pres.Slides(12)
    header(s, "3.4", "模型接入计划", "平台优先覆盖代表性 3R 模型，形成可比较的模型池")
    rows = [
        ["模型类型", "代表模型", "接入价值"],
        ["图像对重建", "DUSt3R / MASt3R", "提供基础重建能力"],
        ["多视图与流式", "Fast3R / Spann3R", "覆盖多图和长序列输入"],
        ["动态场景", "MonST3R / CUT3R", "补充动态与视频场景"],
        ["候选扩展", "Align3R 等", "后续根据权重和环境补充"],
    ]
    table(s, 80, 150, [185, 280, 390], rows)
    add_box(s, 110, 420, 740, 42, "接入重点不是简单堆模型，而是在统一条件下组织输入、输出和对比记录。", fill=LIGHT, line=LINE, size=16, bold=True)
    set_notes(s, "这页把模型池讲成平台验证对象，不夸大已完成状态。")

    # 13 evaluation
    s = pres.Slides(13)
    header(s, "4.1", "验证方案", "平台验证同时关注功能闭环、运行稳定性和对比可用性")
    three_cards(
        s,
        170,
        ["功能闭环", "稳定性验证", "对比能力"],
        [
            "模型注册、任务提交、日志刷新、结果回传均可运行",
            "远端任务异常、取消、失败恢复具有明确状态记录",
            "同一输入下保留耗时、产物和元数据，支持横向查看",
        ],
        "验证标准以可运行、可复现、可对比为主，不把平台结果包装成模型性能结论。",
    )
    set_notes(s, "验证方案页强调工程证据，避免把耗时或指标说成算法优势。")

    # 14 progress
    s = pres.Slides(14)
    header(s, "4.2", "当前进展", "平台已形成基础原型，后续重点补齐视图、记录和接口")
    rows = [
        ["方向", "当前基础", "下一步"],
        ["前端工作台", "已形成命令中心、任务工作台等界面框架", "优化交互和结果查看"],
        ["后端调度", "已设计任务合同与执行器接口", "完善状态记录和异常处理"],
        ["模型接入", "已有代表模型接入基础", "补齐权重、环境和验证记录"],
        ["输出能力", "已规划统一导出与 API 路径", "形成接口文档和演示样例"],
    ]
    table(s, 58, 150, [135, 390, 390], rows)
    add_box(s, 110, 430, 740, 38, "开题阶段重点确认方案边界；中期阶段以真实截图和运行记录作为主要证据。", fill=LIGHT, line=LINE, size=16, bold=True)
    set_notes(s, "进展页要诚实：能说基础原型，不要说所有功能已成熟。")

    # 15 schedule and risk
    s = pres.Slides(15)
    header(s, "4.3", "计划安排与风险应对", "后续工作按原型完善、模型验证和论文整理三阶段推进")
    rows = [
        ["阶段", "工作重点", "主要风险", "应对方式"],
        ["近期", "完善平台交互与任务记录", "UI 与真实流程脱节", "以端到端任务驱动界面"],
        ["中期", "补充模型接入和对比视图", "模型环境差异较大", "执行器隔离模型差异"],
        ["后期", "整理报告、截图和系统文档", "证据链不完整", "保留日志、产物和配置"],
    ]
    table(s, 55, 155, [120, 300, 250, 250], rows)
    add_box(s, 110, 420, 740, 42, "预期成果：可运行平台原型、代表模型接入样例、实验记录与系统说明文档。", fill=LIGHT, line=LINE, size=16, bold=True)
    set_notes(s, "计划页给出阶段和风险。中期 PPT 要把这里的计划换成已完成证据。")

    # 16 thanks
    s = pres.Slides(16)
    set_text(s, 7, "面向前馈式三维重建模型的聚合管理平台设计与实现", 16)
    set_text(s, 2, "感谢各位老师\n敬请批评指正！", 34, True)
    set_text(s, 4, "汇报人", 16, True)
    set_text(s, 5, "崔昊喆  纪博闻", 18, True)
    set_text(s, 8, "北京航空航天大学", 16)
    set_text(s, 10, "2026年6月", 16)
    set_notes(s, "结束语：欢迎老师对平台边界、验证方式和后续计划提出建议。")

    for si in range(1, pres.Slides.Count + 1):
        s = pres.Slides(si)
        for i in range(1, s.Shapes.Count + 1):
            shp = s.Shapes(i)
            try:
                if shp.HasTextFrame and shp.TextFrame.HasText:
                    tr = shp.TextFrame.TextRange
                    tr.Font.Name = "微软雅黑"
                    if tr.Font.Size < 11:
                        tr.Font.Size = 11
            except Exception:
                pass

    pres.SaveAs(str(OUT))
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pres.Export(str(PREVIEW_DIR), "PNG", 1920, 1080)
    try:
        pres.Close()
    except Exception:
        pass
    try:
        app.Quit()
    except Exception:
        pass

    def slide_no(path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else 0

    files = sorted(PREVIEW_DIR.glob("*.PNG"), key=slide_no)
    thumbs = []
    for idx, f in enumerate(files, start=1):
        im = Image.open(f).convert("RGB")
        im.thumbnail((320, 180), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (320, 202), "white")
        canvas.paste(im, (0, 22))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, 319, 201], outline=(210, 220, 230))
        d.text((8, 4), f"Slide {idx:02d}", fill=(0, 70, 130))
        thumbs.append(canvas)

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 202), (245, 247, 250))
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 320, (idx // cols) * 202))
    sheet.save(CONTACT)
    print(OUT)
    print(PREVIEW_DIR)
    print(CONTACT)


if __name__ == "__main__":
    main()
