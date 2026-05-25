from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw
import win32com.client


BASE = Path(r"E:\kykt\Dream\planning\proposal_dream3r\deliverables\ppt_work")
SRC = Path(r"E:\Work\HSY\学术风ppt模板-蓝色.pptx")
OUT = BASE / "platform_fill_template_from_blue.pptx"
PREVIEW_DIR = BASE / "previews_platform_fill_template"
CONTACT = BASE / "contact_sheet_platform_fill_template.png"


def rgb(r: int, g: int, b: int) -> int:
    return r + g * 256 + b * 65536


DARK = rgb(0, 65, 125)
BLUE = rgb(0, 84, 160)
LIGHT = rgb(232, 241, 250)
WHITE = rgb(255, 255, 255)
GRAY = rgb(245, 247, 250)
LINE = rgb(43, 107, 185)
TEXT = rgb(5, 35, 70)


def find_shape(slide, shape_id: int):
    for idx in range(1, slide.Shapes.Count + 1):
        shape = slide.Shapes(idx)
        if shape.Id == shape_id:
            return shape
    return None


def set_text(slide, shape_id: int, text: str, size: float | None = None, bold: bool | None = None):
    shape = find_shape(slide, shape_id)
    if shape is None:
        return
    tr = shape.TextFrame.TextRange
    tr.Text = text
    tr.Font.Name = "微软雅黑"
    if size is not None:
        tr.Font.Size = size
    if bold is not None:
        tr.Font.Bold = -1 if bold else 0


def add_box(slide, x, y, w, h, text, fill=LIGHT, line=LINE, size=16, bold=False, color=TEXT, align=2):
    shape = slide.Shapes.AddShape(1, x, y, w, h)
    shape.Fill.ForeColor.RGB = fill
    shape.Line.ForeColor.RGB = line
    shape.Line.Weight = 1.1
    shape.TextFrame.TextRange.Text = text
    shape.TextFrame.TextRange.Font.Name = "微软雅黑"
    shape.TextFrame.TextRange.Font.Size = size
    shape.TextFrame.TextRange.Font.Bold = -1 if bold else 0
    shape.TextFrame.TextRange.Font.Color.RGB = color
    shape.TextFrame.TextRange.ParagraphFormat.Alignment = align
    shape.TextFrame.MarginLeft = 8
    shape.TextFrame.MarginRight = 8
    shape.TextFrame.MarginTop = 5
    shape.TextFrame.MarginBottom = 5
    try:
        shape.TextFrame.VerticalAnchor = 3
    except Exception:
        pass
    return shape


def set_notes(slide, text: str) -> None:
    body = None
    for idx in range(1, slide.NotesPage.Shapes.Count + 1):
        shape = slide.NotesPage.Shapes(idx)
        try:
            if shape.PlaceholderFormat.Type == 2:
                body = shape
                break
        except Exception:
            pass
    if body is None:
        body = slide.NotesPage.Shapes.AddTextbox(1, 54, 346.5, 432, 283.5)
    body.TextFrame.TextRange.Text = text
    body.TextFrame.TextRange.Font.Name = "微软雅黑"
    body.TextFrame.TextRange.Font.Size = 11


def overlay_image_placeholders(slide, labels):
    for x, y, w, h, text in labels:
        add_box(slide, x, y, w, h, text, fill=WHITE, line=LINE, size=15, bold=True)


def main() -> None:
    shutil.copy2(SRC, OUT)
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    pres = app.Presentations.Open(str(OUT), WithWindow=False)

    keep = {3, 5, 15, 37, 30, 25, 48, 67, 72}
    for i in range(pres.Slides.Count, 0, -1):
        if i not in keep:
            pres.Slides(i).Delete()

    # 1. Cover
    s = pres.Slides(1)
    set_text(s, 7, "2026年春科研课堂开题报告", 16)
    set_text(s, 2, "面向前馈式三维重建模型的\n聚合管理平台设计与实现", 30, True)
    set_text(s, 4, "汇报人", 16, True)
    set_text(s, 5, "崔昊喆  纪博闻", 18, True)
    set_text(s, 8, "北京航空航天大学", 16)
    set_text(s, 10, "2026年5月", 16)
    set_notes(s, "填：题目、汇报人、日期。保持模板封面即可。")

    # 2. Outline
    s = pres.Slides(2)
    add_box(s, 355, 82, 520, 380, "", fill=rgb(247, 250, 253), line=rgb(160, 190, 225), size=1)
    add_box(s, 385, 115, 455, 48, "01    平台建设背景", fill=BLUE, line=BLUE, size=18, bold=True, color=WHITE)
    add_box(s, 385, 185, 455, 48, "02    平台定位与功能", fill=WHITE, line=rgb(220, 225, 230), size=18, bold=True, color=DARK)
    add_box(s, 385, 255, 455, 48, "03    技术架构与运行流程", fill=WHITE, line=rgb(220, 225, 230), size=18, bold=True, color=DARK)
    add_box(s, 385, 325, 455, 48, "04    当前进展与后续计划", fill=WHITE, line=rgb(220, 225, 230), size=18, bold=True, color=DARK)
    set_notes(s, "填：4 个目录项。建议不要超过 4 项，保持平台主线。")

    # 3. Background / three-column problem framing
    s = pres.Slides(3)
    set_text(s, 90, "1.1", 18, True)
    set_text(s, 92, "平台建设背景", 26, True)
    set_text(s, 13, "3R 模型从论文到实践仍存在部署、算力和对比三类工程门槛", 18, True)
    add_box(s, 175, 162, 150, 32, "门槛一", fill=WHITE, line=rgb(160, 190, 225), size=18, bold=True, color=DARK)
    add_box(s, 448, 162, 150, 32, "门槛二", fill=WHITE, line=rgb(160, 190, 225), size=18, bold=True, color=DARK)
    add_box(s, 722, 162, 150, 32, "门槛三", fill=WHITE, line=rgb(160, 190, 225), size=18, bold=True, color=DARK)
    for sid, text in [(39, "问 题"), (41, "现 状"), (52, "需 求")]:
        set_text(s, sid, text, 18, True)
    for sid, text in [
        (65, "部署门槛高：环境、依赖、权重路径和脚本分散"),
        (66, "算力使用散：上传、启动、日志、回传依赖手动命令"),
        (67, "对比成本高：输入输出格式不统一，结果难横向比较"),
        (72, "【填：你遇到的实际部署问题】\n例：模型 A/B/C 的环境差异"),
        (73, "【填：远端 GPU 使用痛点】\n例：任务状态、日志、回传不集中"),
        (74, "【填：对比与复现痛点】\n例：同一输入需要手动整理结果"),
        (75, "快速部署与统一入口"),
        (76, "远端任务调度与结果回传"),
        (77, "统一对比与报告导出"),
    ]:
        set_text(s, sid, text, 13 if sid in (72, 73, 74) else 15, sid in (75, 76, 77))
    set_notes(s, "填：只写真实痛点。最好写你实际跑模型时遇到的问题，不要写抽象口号。")

    # 4. Platform positioning / three cards
    s = pres.Slides(4)
    set_text(s, 90, "1.2", 18, True)
    set_text(s, 92, "平台定位", 26, True)
    add_box(s, 0, 92, 960, 448, "", fill=WHITE, line=WHITE, size=1)
    add_box(s, 45, 112, 870, 60, "平台面向前馈式三维重建模型的学习实践、实验研究和应用扩展", fill=WHITE, line=LINE, size=20, bold=True, align=1)
    add_box(s, 60, 205, 245, 55, "学习与实践", fill=BLUE, line=BLUE, color=WHITE, size=20, bold=True)
    add_box(s, 357, 205, 245, 55, "实验与研究", fill=BLUE, line=BLUE, color=WHITE, size=20, bold=True)
    add_box(s, 654, 205, 245, 55, "应用与扩展", fill=BLUE, line=BLUE, color=WHITE, size=20, bold=True)
    add_box(s, 60, 268, 245, 135, "【填：学习场景】\n新模型如何快速跑起来\n减少环境配置和命令记忆", fill=LIGHT, line=LINE, size=15, bold=True)
    add_box(s, 357, 268, 245, 135, "【填：实验场景】\n同一输入、多模型对比\n统一记录耗时、日志和结果", fill=LIGHT, line=LINE, size=15, bold=True)
    add_box(s, 654, 268, 245, 135, "【填：应用场景】\n稳定模型封装为 API\n供后续程序或服务调用", fill=LIGHT, line=LINE, size=15, bold=True)
    add_box(s, 110, 440, 740, 48, "【底部主句】降低部署门槛，统一远端算力使用和结果管理。", fill=LIGHT, line=LINE, size=17, bold=True)
    set_notes(s, "填：这页讲平台不是工具堆砌，而是三层定位：学习实践、实验研究、应用扩展。")

    # 5. Core functions / four cards
    s = pres.Slides(5)
    set_text(s, 90, "2.1", 18, True)
    set_text(s, 92, "核心功能模块", 26, True)
    add_box(s, 0, 92, 960, 448, "", fill=WHITE, line=WHITE, size=1)
    add_box(s, 55, 116, 850, 44, "平台围绕模型部署、任务运行、结果对比和能力输出组织功能", fill=BLUE, line=BLUE, color=WHITE, size=18, bold=True)
    overlay_image_placeholders(s, [
        (60, 190, 190, 115, "放图：命令中心 / 首页截图"),
        (275, 190, 190, 115, "放图：任务工作台截图"),
        (490, 190, 190, 115, "放图：样本矩阵 / 对比截图"),
        (705, 190, 190, 115, "放图：模型路线 / 注册截图"),
    ])
    add_box(s, 60, 316, 190, 82, "命令中心\n【填 2-3 个功能点】", fill=LIGHT, line=LINE, size=14, bold=True)
    add_box(s, 275, 316, 190, 82, "任务工作台\n【填 2-3 个功能点】", fill=LIGHT, line=LINE, size=14, bold=True)
    add_box(s, 490, 316, 190, 82, "样本矩阵\n【填 2-3 个功能点】", fill=LIGHT, line=LINE, size=14, bold=True)
    add_box(s, 705, 316, 190, 82, "模型注册与输出\n【填 2-3 个功能点】", fill=LIGHT, line=LINE, size=14, bold=True)
    add_box(s, 110, 440, 740, 48, "【底部主句】把模型部署、远端运行、日志查看和结果回传统一到一个入口。", fill=LIGHT, line=LINE, size=17, bold=True)
    set_notes(s, "填：每个功能模块放真实截图或留图位。不要写太多解释，截图比文字重要。")

    # 6. Technical architecture / left image + right explanation
    s = pres.Slides(6)
    set_text(s, 90, "2.2", 18, True)
    set_text(s, 92, "技术架构", 26, True)
    add_box(s, 35, 85, 900, 425, "", fill=WHITE, line=WHITE, size=1)
    add_box(s, 42, 120, 870, 42, "平台通过桌面前端、本地后端、远端调度和模型执行器连接实际算力", fill=BLUE, line=BLUE, color=WHITE, size=17, bold=True)
    overlay_image_placeholders(s, [
        (55, 185, 530, 205, "放图：四层技术架构图\nTauri / React / FastAPI / SSH-SCP / 执行器"),
        (610, 185, 285, 205, "放图：任务运行截图\n日志、状态、结果回传"),
    ])
    add_box(s, 610, 405, 135, 40, "桌面前端", fill=BLUE, line=BLUE, color=WHITE, size=15, bold=True)
    add_box(s, 760, 405, 135, 40, "本地后端", fill=BLUE, line=BLUE, color=WHITE, size=15, bold=True)
    add_box(s, 610, 452, 135, 40, "远端调度", fill=BLUE, line=BLUE, color=WHITE, size=15, bold=True)
    add_box(s, 760, 452, 135, 40, "模型执行器", fill=BLUE, line=BLUE, color=WHITE, size=15, bold=True)
    add_box(s, 55, 415, 530, 58, "【填：一句架构结论】例如：执行器封装模型差异，平台统一提交、日志监听和结果回传。", fill=LIGHT, line=LINE, size=15, bold=True)
    set_notes(s, "填：技术架构图优先用简洁四层图。右侧可以放真实运行截图。")

    # 7. Running flow / evidence
    s = pres.Slides(7)
    set_text(s, 90, "3.1", 18, True)
    set_text(s, 92, "实际运行流程", 26, True)
    add_box(s, 0, 92, 960, 448, "", fill=WHITE, line=WHITE, size=1)
    add_box(s, 45, 115, 870, 52, "一次任务应能完整覆盖：选择模型、上传输入、远端执行、日志监听、结果回传", fill=WHITE, line=LINE, size=20, bold=True)
    add_box(s, 55, 205, 390, 50, "运行流程", fill=BLUE, line=BLUE, color=WHITE, size=20, bold=True)
    add_box(s, 55, 265, 390, 112, "【填：任务如何从本地提交到远端 GPU】\n建议写 3-4 步，不要写长段", fill=WHITE, line=LINE, size=17, bold=True)
    add_box(s, 55, 405, 390, 50, "运行证据", fill=BLUE, line=BLUE, color=WHITE, size=20, bold=True)
    add_box(s, 55, 465, 390, 58, "【填：实际运行日志 / 结果回传说明】", fill=WHITE, line=LINE, size=16, bold=True)
    overlay_image_placeholders(s, [
        (500, 205, 400, 260, "放图：运行日志 / 任务状态 / 输出结果截图"),
    ])
    set_notes(s, "填：这一页是证明平台不是空想的关键页。必须尽量放真实运行截图。")

    # 8. Progress and roadmap
    s = pres.Slides(8)
    set_text(s, 90, "4.1", 18, True)
    set_text(s, 92, "当前进展与后续计划", 26, True)
    add_box(s, 0, 92, 960, 448, "", fill=WHITE, line=WHITE, size=1)
    add_box(s, 42, 115, 875, 44, "围绕快速部署、算力调度和结果对比，逐步完善平台功能", fill=BLUE, line=BLUE, color=WHITE, size=18, bold=True)
    add_box(s, 55, 190, 120, 66, "当前\n进展", fill=BLUE, line=BLUE, color=WHITE, size=18, bold=True)
    add_box(s, 185, 190, 430, 66, "【填：已完成】\n桌面端基础界面 / 本地后端 / 远端调度 / 已接入模型", fill=LIGHT, line=LINE, size=14, bold=True, align=1)
    add_box(s, 55, 285, 120, 66, "近期\n优化", fill=BLUE, line=BLUE, color=WHITE, size=18, bold=True)
    add_box(s, 185, 285, 430, 66, "【填：近期要改】\nUI 交互 / 任务记录 / 对比视图 / 报告导出", fill=LIGHT, line=LINE, size=14, bold=True, align=1)
    add_box(s, 55, 380, 120, 66, "后续\n扩展", fill=BLUE, line=BLUE, color=WHITE, size=18, bold=True)
    add_box(s, 185, 380, 430, 66, "【填：后续方向】\n算力队列 / 更多模型执行器 / API 服务化调用", fill=LIGHT, line=LINE, size=14, bold=True, align=1)
    overlay_image_placeholders(s, [
        (660, 190, 250, 115, "放图：当前 UI 截图"),
        (660, 335, 250, 115, "放图：耗时图 / 后续计划图"),
    ])
    set_notes(s, "填：用真实完成项，不要写已经实现但实际还没稳定的功能。")

    # 9. Thanks
    s = pres.Slides(9)
    set_text(s, 7, "面向前馈式三维重建模型的聚合管理平台设计与实现", 16)
    set_text(s, 2, "感谢各位老师\n敬请批评指正！", 34, True)
    set_text(s, 4, "汇报人", 16, True)
    set_text(s, 5, "崔昊喆  纪博闻", 18, True)
    set_text(s, 8, "北京航空航天大学", 16)
    set_text(s, 10, "2026年5月", 16)
    set_notes(s, "填：结束页一般不需要再加复杂内容。")

    pres.Save()
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pres.Export(str(PREVIEW_DIR), "PNG", 1920, 1080)
    pres.Close()
    app.Quit()

    def slide_no(path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else 0

    files = sorted(PREVIEW_DIR.glob("*.PNG"), key=slide_no)
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im.thumbnail((360, 203), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 223), "white")
        canvas.paste(im, (0, 20))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, 359, 222], outline=(210, 220, 230))
        d.text((8, 3), f"Slide {slide_no(f):02d}", fill=(0, 70, 130))
        thumbs.append(canvas)

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 360, rows * 223), (245, 247, 250))
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 360, (idx // cols) * 223))
    sheet.save(CONTACT)

    print(OUT)
    print(PREVIEW_DIR)
    print(CONTACT)


if __name__ == "__main__":
    main()
